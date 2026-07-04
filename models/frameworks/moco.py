"""
MoCo v2 framework module.

Adapts Moco-Imagenet-pred/moco/builder.py. Differences for pred_ssl:
  - The projector is a SEPARATE module (encoder_q's fc stays Identity), so the
    trainable trunk's state_dict is clean for the eval scripts.
  - To feed the relational head, the query encoder runs ONE batched forward of
    cat([v1, v2]) (with grad) and the pooled features are split into h1, h2. The
    InfoNCE query is the v1 half; the key is v2 through the no-grad momentum encoder.
    This is the ~1.5x backbone-compute cost of MoCo's asymmetry. When rel_lambda==0
    (baseline) the extra v2 forward is skipped entirely (faithful pure-MoCo control).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..projector import build_projector, projector_out_dim
from ..split import build_split
from ..types import ModelOutput


def _moco_v2_head(feat_dim, dim):
    return nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, dim))


class MoCoModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.K = cfg.get("moco_k", 16384)
        self.m = cfg.get("moco_m", 0.999)
        self.T = cfg.get("moco_t", 0.2)
        # In decoupled mode the relational pair is embedded separately (encode_features),
        # so the framework forward must NOT pay the extra cat([v1,v2]) query forward.
        self.pair_feats = cfg.get("rel_lambda", 0.0) > 0 and not cfg.get("rel_decoupled", False)
        native_dim = cfg.get("moco_dim", 128)
        dim = projector_out_dim(cfg, native_dim)   # queue width follows the head's output

        self.encoder_q, feat_dim = build_backbone(cfg["arch"], hook=True)
        self.encoder_k, _ = build_backbone(cfg["arch"], hook=False)
        self.feat_dim = feat_dim
        self.split = build_split(cfg, feat_dim)   # feat_split off -> identity
        d_in = self.split.ssl_dim
        self.projector_q = build_projector(cfg, d_in, lambda: _moco_v2_head(d_in, native_dim))
        self.projector_k = build_projector(cfg, d_in, lambda: _moco_v2_head(d_in, native_dim))

        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False
        for q, k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, self.K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data = k.data * self.m + q.data * (1.0 - self.m)
        for q, k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            k.data = k.data * self.m + q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bs = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + bs <= self.K:
            self.queue[:, ptr:ptr + bs] = keys.T
        else:
            rem = self.K - ptr
            self.queue[:, ptr:] = keys.T[:, :rem]
            self.queue[:, :bs - rem] = keys.T[:, rem:]
        self.queue_ptr[0] = (ptr + bs) % self.K

    def forward(self, v1, v2):
        N = v1.size(0)
        if self.pair_feats:
            feat = self.encoder_q(torch.cat([v1, v2], dim=0))  # (2N, feat_dim), fc=Identity
            h1, h2 = feat[:N], feat[N:]
            q = F.normalize(self.projector_q(self.split.ssl(h1)), dim=1)
        else:
            h1 = self.encoder_q(v1)                              # (N, feat_dim)
            h2 = None
            q = F.normalize(self.projector_q(self.split.ssl(h1)), dim=1)

        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.projector_k(self.split.ssl(self.encoder_k(v2))), dim=1)

        l_pos = (q * k).sum(dim=1, keepdim=True)                 # (N, 1)
        l_neg = q @ self.queue.clone().detach()                  # (N, K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(N, dtype=torch.long, device=q.device)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean().item() * 100.0
        self._dequeue_and_enqueue(k)
        return ModelOutput(ssl_loss=loss, ssl_acc=acc, h1=h1, h2=h2)
