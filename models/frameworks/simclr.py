"""
SimCLR framework module.

Single trainable backbone; both views pass through it, so h1, h2 (post-avgpool,
with grad) are free. Adapts SimCLR-Imagenet/simclr/builder.py (2-layer projector
Linear(d,d)->ReLU->Linear(d,128)) and wraps NTXentLoss internally so the training
loop is framework-agnostic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..projector import build_projector
from ..split import build_split
from ..types import ModelOutput
from ...losses import NTXentLoss


def _contrastive_top1(z1, z2):
    """InfoNCE top-1: fraction of the 2N anchors whose nearest non-self view is its positive."""
    z = torch.cat([z1, z2], dim=0)
    N2 = z.size(0)
    N = N2 // 2
    sim = z @ z.t()
    sim.fill_diagonal_(-1e9)
    pos = torch.cat([
        torch.arange(N, N2, device=z.device),
        torch.arange(0, N, device=z.device),
    ])
    pred = sim.argmax(dim=1)
    return (pred == pos).float().mean().item() * 100.0


class SimCLRModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone, feat_dim = build_backbone(cfg["arch"])
        self.feat_dim = feat_dim
        self.split = build_split(cfg, feat_dim)   # feat_split off -> identity
        dim = cfg.get("simclr_dim", 128)
        d_in = self.split.ssl_dim
        self.projector = build_projector(cfg, d_in, lambda: nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, dim),
        ))
        self.criterion = NTXentLoss(temperature=cfg.get("temperature", 0.5))

    def _encode(self, x):
        self.backbone(x)                 # triggers the avgpool hook
        h = self.backbone._feat          # (N, feat_dim), requires grad
        z = F.normalize(self.projector(self.split.ssl(h)), dim=1)
        return h, z

    def forward(self, v1, v2):
        h1, z1 = self._encode(v1)
        h2, z2 = self._encode(v2)
        loss = self.criterion(torch.cat([z1, z2], dim=0))
        with torch.no_grad():
            acc = _contrastive_top1(z1, z2)
        return ModelOutput(ssl_loss=loss, ssl_acc=acc, h1=h1, h2=h2)
