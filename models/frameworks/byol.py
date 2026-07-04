"""
BYOL framework module.

Adapts BYOL-Imagenet/byol/{builder,loss}.py. The online backbone already runs both
views, so h1, h2 (post-avgpool, with grad) are free — no extra forward. The target
network is a momentum-updated deepcopy with no predictor; the cosine tau schedule
(tau_base -> 1.0) matches main_byol.cosine_momentum_schedule and is advanced inside
forward (one step per batch). The online backbone uses hook=False so the deepcopy'd
target carries no forward hook.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..projector import build_projector, is_custom, projector_out_dim
from ..types import ModelOutput
from ...losses import BYOLLoss


def _build_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOLModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        proj_hidden = cfg.get("proj_hidden_dim", 4096)
        proj_dim = cfg.get("proj_dim", 256)
        self.tau_base = cfg.get("tau_base", 0.996)
        self.tau = self.tau_base
        self.total_steps = None
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

        self.online_backbone, feat_dim = build_backbone(cfg["arch"], hook=False)
        self.feat_dim = feat_dim
        self.online_projector = build_projector(
            cfg, feat_dim, lambda: _build_mlp(feat_dim, proj_hidden, proj_dim))
        # predictor maps the projector's OUTPUT back to itself; its hidden follows the
        # custom head width when overridden, else the native BYOL projector hidden.
        out_dim = projector_out_dim(cfg, proj_dim)
        pred_hidden = cfg.get("proj_hidden", proj_hidden) if is_custom(cfg) else proj_hidden
        self.online_predictor = _build_mlp(out_dim, pred_hidden, out_dim)

        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.criterion = BYOLLoss()

    def set_total_steps(self, n):
        """Optional hook (called by train.py) to enable the cosine tau schedule."""
        self.total_steps = n

    def _current_tau(self):
        if not self.total_steps:
            return self.tau_base
        step = int(self._step.item())
        return 1 - (1 - self.tau_base) * (math.cos(math.pi * step / self.total_steps) + 1) / 2

    @torch.no_grad()
    def _update_target(self):
        self.tau = self._current_tau()
        for o, t in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            t.data = self.tau * t.data + (1 - self.tau) * o.data
        for o, t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            t.data = self.tau * t.data + (1 - self.tau) * o.data

    def forward(self, v1, v2):
        h1 = self.online_backbone(v1)                      # (N, feat_dim), fc=Identity
        p1 = self.online_predictor(self.online_projector(h1))
        h2 = self.online_backbone(v2)
        p2 = self.online_predictor(self.online_projector(h2))

        with torch.no_grad():
            tz1 = self.target_projector(self.target_backbone(v1))
            tz2 = self.target_projector(self.target_backbone(v2))

        p1, p2 = F.normalize(p1, dim=1), F.normalize(p2, dim=1)
        tz1, tz2 = F.normalize(tz1, dim=1), F.normalize(tz2, dim=1)
        loss = self.criterion(p1, p2, tz1, tz2)

        if self.training:
            self._update_target()
            self._step += 1

        return ModelOutput(ssl_loss=loss, ssl_acc=0.0, h1=h1, h2=h2)
