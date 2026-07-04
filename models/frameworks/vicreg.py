"""
VICReg framework module (Bardes, Ponce & LeCun, 2022).

Single trainable backbone; both views pass through it, so h1, h2 (post-avgpool,
with grad) are free for the relational head — no momentum encoder, no queue, no
negatives (same cheap integration as SimCLR/BYOL). The backbone feeds a parameterized
"expander" MLP (3-layer, width 8192, BN by default — the canonical VICReg recipe) and
VICRegLoss is applied to the two expander outputs.

The expander is fully configurable: vicreg_expander_layers (count), vicreg_expander_dim
(hidden width) and vicreg_proj_dim (output) — the standalone home of the "size as a
parameter" feature until the shared native/custom projector lands across all frameworks.
"""

import torch.nn as nn

from ..backbones import build_backbone
from ..projector import build_projector
from ..split import build_split
from ..types import ModelOutput
from ...losses import VICRegLoss


def _build_expander(in_dim, hidden_dim, out_dim, num_layers, batch_norm=True):
    """VICReg expander: `num_layers` Linear layers, Linear->BN->ReLU between them.

    num_layers=3 -> Linear(in,h)->BN->ReLU -> Linear(h,h)->BN->ReLU -> Linear(h,out),
    the canonical VICReg expander. num_layers=1 -> a single Linear(in,out).
    """
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class VICRegModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone, feat_dim = build_backbone(cfg["arch"])
        self.feat_dim = feat_dim
        self.split = build_split(cfg, feat_dim)   # feat_split off -> identity
        d_in = self.split.ssl_dim
        # native: the configurable VICReg expander (vicreg_* knobs).
        # custom (proj_preset='custom'): the shared proj_* MLP, like every other framework.
        self.expander = build_projector(cfg, d_in, lambda: _build_expander(
            d_in,
            cfg.get("vicreg_expander_dim", 8192),
            cfg.get("vicreg_proj_dim", 8192),
            cfg.get("vicreg_expander_layers", 3),
        ))
        self.criterion = VICRegLoss(
            sim_coeff=cfg.get("vicreg_sim_coeff", 25.0),
            std_coeff=cfg.get("vicreg_std_coeff", 25.0),
            cov_coeff=cfg.get("vicreg_cov_coeff", 1.0),
        )

    def _encode(self, x):
        h = self.backbone(x)             # (N, feat_dim), fc=Identity -> pooled feature
        z = self.expander(self.split.ssl(h))  # (N, proj_dim), NOT normalized (VICReg)
        return h, z

    def forward(self, v1, v2):
        h1, z1 = self._encode(v1)
        h2, z2 = self._encode(v2)
        loss = self.criterion(z1, z2)
        return ModelOutput(ssl_loss=loss, ssl_acc=0.0, h1=h1, h2=h2)
