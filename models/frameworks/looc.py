"""
LooC framework module (Xiao et al., "What Should Not Be Contrastive in Contrastive
Learning", ICLR 2021).

LooC is the *structural* answer to augmentation invariance, and the natural
competitor to this paper's *predictive* one: instead of adding a task, it splits
the embedding into one space per augmentation, each invariant to every
augmentation except its own. Concretely, on top of MoCo:

  - a shared backbone, and 1 + n_aug projection heads with 1 + n_aug queues;
  - space Z0 is the ordinary all-invariant contrastive space (key = an
    independently augmented view);
  - space Z_a is trained with a key that shares every augmentation parameter with
    the query except those of group a, so matching query to key in that space
    requires encoding a.

The total loss is the sum of the per-space InfoNCE terms, as in the paper.

``full_multiview=False`` keeps the earlier degenerate configuration in which LooC
reduces exactly to MoCo over a single space; it is retained only for backwards
compatibility with previously logged runs and should not be used as a LooC
baseline, since it implements none of the above.
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
        self.full_multiview = cfg.get("full_multiview", False)
        self.looc_augs = tuple(cfg.get("looc_augs", ("rotation", "color")))
        # number of embedding spaces: Z0 plus one per augmentation group
        self.n_spaces = 1 + (len(self.looc_augs) if self.full_multiview else 0)

        self.K = cfg.get("K", 16384)
        self.m = cfg.get("m", 0.999)
        self.T = cfg.get("T", 0.2)
        # In decoupled mode the relational pair is embedded separately, so the forward
        # must not pay the extra query forward for the (q, k0) pair.
        self.pair_feats = cfg.get("rel_lambda", 0.0) > 0 and not cfg.get("rel_decoupled", False)
        native_dim = cfg.get("dim", 128)
        dim = projector_out_dim(cfg, native_dim)

        self.backbone_q, feat_dim = build_backbone(cfg["arch"], hook=True)
        self.backbone_k, _ = build_backbone(cfg["arch"], hook=False)
        self.feat_dim = feat_dim
        self.split = build_split(cfg, feat_dim)
        d_in = self.split.ssl_dim

        self.heads_q = nn.ModuleList([
            build_projector(cfg, d_in, lambda: _head(d_in, native_dim))
            for _ in range(self.n_spaces)])
        self.heads_k = nn.ModuleList([
            build_projector(cfg, d_in, lambda: _head(d_in, native_dim))
            for _ in range(self.n_spaces)])

        for q, k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False
        for hq, hk in zip(self.heads_q, self.heads_k):
            for q, k in zip(hq.parameters(), hk.parameters()):
                k.data.copy_(q.data)
                k.requires_grad = False

        # one queue per embedding space
        for s in range(self.n_spaces):
            self.register_buffer(f"queue_{s}", F.normalize(torch.randn(dim, self.K), dim=0))
            self.register_buffer(f"queue_ptr_{s}", torch.zeros(1, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            k.data = k.data * self.m + q.data * (1.0 - self.m)
        for hq, hk in zip(self.heads_q, self.heads_k):
            for q, k in zip(hq.parameters(), hk.parameters()):
                k.data = k.data * self.m + q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, s):
        queue = getattr(self, f"queue_{s}")
        ptr_buf = getattr(self, f"queue_ptr_{s}")
        bs = keys.shape[0]
        ptr = int(ptr_buf)
        if ptr + bs <= self.K:
            queue[:, ptr:ptr + bs] = keys.T
        else:
            rem = self.K - ptr
            queue[:, ptr:] = keys.T[:, :rem]
            queue[:, :bs - rem] = keys.T[:, rem:]
        ptr_buf[0] = (ptr + bs) % self.K

    def _infonce(self, q, k, s):
        l_pos = (q * k).sum(dim=1, keepdim=True)
        l_neg = q @ getattr(self, f"queue_{s}").clone().detach()
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return self.criterion(logits, labels), logits, labels

    def forward(self, v1, v2, *extra_keys):
        """v1 = query, v2 = all-invariant key k0, extra_keys = one key per LooC space."""
        N = v1.size(0)
        if self.pair_feats:
            feat = self.backbone_q(torch.cat([v1, v2], dim=0))   # (2N, feat_dim)
            h1, h2 = feat[:N], feat[N:]
        else:
            h1 = self.backbone_q(v1)
            h2 = None
        z_q = self.split.ssl(h1)

        with torch.no_grad():
            self._momentum_update()
            key_views = (v2,) + tuple(extra_keys)
            keys = [F.normalize(self.heads_k[s](self.split.ssl(self.backbone_k(kv))), dim=1)
                    for s, kv in enumerate(key_views[:self.n_spaces])]

        loss = 0.0
        acc = 0.0
        for s, k in enumerate(keys):
            q = F.normalize(self.heads_q[s](z_q), dim=1)
            l, logits, labels = self._infonce(q, k, s)
            loss = loss + l                     # the paper sums the per-space losses
            if s == 0:                          # report the all-invariant space
                acc = (logits.argmax(dim=1) == labels).float().mean().item() * 100.0
            self._dequeue_and_enqueue(k, s)

        return ModelOutput(ssl_loss=loss, ssl_acc=acc, h1=h1, h2=h2)
