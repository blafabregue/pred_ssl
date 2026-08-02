"""
Barlow Twins framework module (Zbontar et al., 2021).

Deliberately built as VICReg's twin: single trainable backbone, both views through
it (so h1, h2 are free for the relational head), the same projector geometry and
the same LARS+warmup recipe. Only the loss differs -- cross-correlation to the
identity, rather than variance/invariance/covariance.

That symmetry is the point. The paper hypothesizes that redundancy-reduction
objectives are structurally opposed to the relational signal (a factor shared
across half the pairs produces exactly the low-variance, correlated direction such
objectives penalize). Barlow Twins is the falsification test: if the hypothesis
holds it should reproduce VICReg's rotation anomaly, and if it does not, the
explanation is wrong. Keeping every other component identical is what makes the
comparison informative.
"""

import torch.nn as nn

from ..backbones import build_backbone
from ..projector import build_projector
from ..split import build_split
from ..types import ModelOutput
from ...losses import BarlowTwinsLoss


def _build_projector(in_dim, hidden_dim, out_dim, num_layers, batch_norm=True):
    """Barlow Twins projector: Linear->BN->ReLU between layers, plain Linear last."""
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BarlowTwinsModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone, feat_dim = build_backbone(cfg["arch"])
        self.feat_dim = feat_dim
        self.split = build_split(cfg, feat_dim)   # feat_split off -> identity
        d_in = self.split.ssl_dim
        out_dim = cfg.get("barlow_proj_dim", 1024)
        self.projector = build_projector(cfg, d_in, lambda: _build_projector(
            d_in,
            cfg.get("barlow_expander_dim", 2048),
            out_dim,
            cfg.get("barlow_expander_layers", 3),
        ))
        self.criterion = BarlowTwinsLoss(out_dim, lambd=cfg.get("barlow_lambd", 0.0051))

    def _encode(self, x):
        h = self.backbone(x)                  # (N, feat_dim), fc=Identity
        z = self.projector(self.split.ssl(h))  # (N, out_dim), NOT normalized
        return h, z

    def forward(self, v1, v2):
        h1, z1 = self._encode(v1)
        h2, z2 = self._encode(v2)
        loss = self.criterion(z1, z2)
        return ModelOutput(ssl_loss=loss, ssl_acc=0.0, h1=h1, h2=h2)
