"""
LooC framework module (v1: n_aug = 0).

Adapts Looc-Imagenet/looc/builder.py. With n_aug=0 / full_multiview=false, LooC
reduces to MoCo over a single embedding space (Z0): query view q (=v1) through the
trainable backbone_q, standard key k0 (=v2) through the no-grad momentum backbone_k.
The relational head sees the (q, k0) pair, captured via one batched forward of
backbone_q(cat([v1, v2])) (the k0 view normally only goes through backbone_k).

The full multi-view LooC (extra shared-augmentation embedding spaces) is deferred
to Phase 5 and raises NotImplementedError here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..projector import build_projector, projector_out_dim
from ..split import build_split
from ..types import ModelOutput


def _head(feat_dim, dim):
    return nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, dim))


class LooCModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        if cfg.get("n_aug", 0) != 0 or cfg.get("full_multiview", False):
            raise NotImplementedError(
                "v1 LooC supports n_aug=0 / full_multiview=false only (Phase 5).")
        self.K = cfg.get("K", 16384)
        self.m = cfg.get("m", 0.999)
        self.T = cfg.get("T", 0.2)
        # In decoupled mode the relational pair is embedded separately (encode_features),
        # so the framework forward must NOT pay the extra cat([v1,v2]) query forward.
        self.pair_feats = cfg.get("rel_lambda", 0.0) > 0 and not cfg.get("rel_decoupled", False)
        native_dim = cfg.get("dim", 128)
        dim = projector_out_dim(cfg, native_dim)   # queue width follows the head's output

        self.backbone_q, feat_dim = build_backbone(cfg["arch"], hook=True)
        self.backbone_k, _ = build_backbone(cfg["arch"], hook=False)
        self.feat_dim = feat_dim
        self.split = build_split(cfg, feat_dim)   # feat_split off -> identity
        d_in = self.split.ssl_dim
        self.head_q = build_projector(cfg, d_in, lambda: _head(d_in, native_dim))
        self.head_k = build_projector(cfg, d_in, lambda: _head(d_in, native_dim))

        for q, k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False
        for q, k in zip(self.head_q.parameters(), self.head_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, self.K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            k.data = k.data * self.m + q.data * (1.0 - self.m)
        for q, k in zip(self.head_q.parameters(), self.head_k.parameters()):
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
            feat = self.backbone_q(torch.cat([v1, v2], dim=0))   # (2N, feat_dim)
            h1, h2 = feat[:N], feat[N:]
            q = F.normalize(self.head_q(self.split.ssl(h1)), dim=1)
        else:
            h1 = self.backbone_q(v1)
            h2 = None
            q = F.normalize(self.head_q(self.split.ssl(h1)), dim=1)

        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.head_k(self.split.ssl(self.backbone_k(v2))), dim=1)

        l_pos = (q * k).sum(dim=1, keepdim=True)
        l_neg = q @ self.queue.clone().detach()
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(N, dtype=torch.long, device=q.device)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean().item() * 100.0
        self._dequeue_and_enqueue(k)
        return ModelOutput(ssl_loss=loss, ssl_acc=acc, h1=h1, h2=h2)
