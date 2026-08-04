"""
Positives-only framework: alignment plus a transformation-prediction regularizer.

The question it exists to answer: can predicting the augmentation replace ALL of
the usual anti-collapse machinery? The SSL term here is BYOL's regression term and
nothing else -- no negatives, no predictor, no momentum target, no stop-gradient,
no variance or covariance penalty. Its minimum is the constant representation. The
only thing standing against that is the auxiliary head the training loop adds on
top (AugSelf's omega regression, or ours over the nine factors), so this framework
is only meaningful when `rel_lambda > 0`.

Why the auxiliary term can work at all: at h1 = h2 = c the head sees a single
input and can emit a single vector, while its target omega1 - omega2 varies from
pair to pair. The auxiliary loss is therefore strictly positive at the collapsed
solution and its gradient points away from it. Complete collapse is blocked.

Why that is NOT sufficient, and what the experiment is actually testing: omega is
only 8-12 dimensional. Consider h = [content, omega-code] against h = [0,
omega-code]. Both predict omega perfectly, and both pay the SAME alignment cost.
The content is a flat direction -- nothing rewards keeping it, and weight decay
actively shrinks it. So the failure mode to watch is not the constant solution but
a low-dimensional one that encodes the augmentation and discards the image.

What pushes back is that predicting omega1 - omega2 needs content for SOME factors
and not others. Saying "view 2 is shifted right" or "rotated 90 degrees" requires
putting the two views in correspondence, which needs real features. Brightness or
blur differences are readable off low-order image statistics in a handful of
dimensions. Hence `regress_factors`, and hence the collapse diagnostics: the
per-epoch effective rank is the measurement this framework is built to produce.

Attachment points matter. Alignment is on the projector output z, the auxiliary
head reads the backbone feature h. The projector can absorb the invariance and
leave omega information in h, which relieves the tension between the two terms.
But if z collapses the alignment gradient vanishes and the run silently degrades
into auxiliary-only pretraining -- which is why train.py logs the two losses
separately, and why the lambda=0 control is not optional.
"""

import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..projector import build_projector
from ..split import build_split
from ..types import ModelOutput
from ...losses import AlignmentLoss


def _build_mlp(in_dim, hidden_dim, out_dim, batch_norm=True):
    layers = [nn.Linear(in_dim, hidden_dim)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers += [nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class PosOnlyModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        proj_hidden = cfg.get("proj_hidden_dim", 4096)
        proj_dim = cfg.get("proj_dim", 256)
        # BatchNorm in the projector is a KNOWN confound for collapse claims: the
        # "BYOL works without a momentum encoder" debate turned on whether BN alone
        # supplies an implicit contrastive signal. Exposed as a knob so the
        # lambda=0 control can be rerun without it and settle the attribution.
        proj_bn = cfg.get("align_proj_bn", True)

        self.backbone, feat_dim = build_backbone(cfg["arch"], hook=False)
        self.feat_dim = feat_dim
        self.split = build_split(cfg, feat_dim)   # feat_split off -> identity
        d_in = self.split.ssl_dim
        self.projector = build_projector(
            cfg, d_in, lambda: _build_mlp(d_in, proj_hidden, proj_dim, proj_bn))
        self.criterion = AlignmentLoss()

    def forward(self, v1, v2):
        h1 = self.backbone(v1)                   # (N, feat_dim), fc=Identity
        h2 = self.backbone(v2)
        z1 = F.normalize(self.projector(self.split.ssl(h1)), dim=1)
        z2 = F.normalize(self.projector(self.split.ssl(h2)), dim=1)
        return ModelOutput(ssl_loss=self.criterion(z1, z2), ssl_acc=0.0, h1=h1, h2=h2)
