"""
AugSelf's augmentation-parameter-difference head (Lee et al., 2021), for the
baseline comparison in the paper.

The paper defines one 3-layer MLP ``phi^aug`` per augmentation type, predicting
``omega_1^aug - omega_2^aug`` from the two views' representations. Its default
predicted set is A_AugSelf = {crop, color}, i.e. 4 + 4 parameters; we keep the
per-augmentation split so each group gets its own head, exactly as specified.

Contrast with models/rel_head.py: AugSelf's target is ANTI-symmetric under view
swap (omega_1 - omega_2 flips sign), so the input is the ordered concatenation
[h1, h2]; our relational head predicts a symmetric same/different relation and
therefore uses the symmetric combination [h1 + h2, |h1 - h2|].
"""

import torch
import torch.nn as nn

# (name, number of parameters) for the paper's default predicted set.
AUGSELF_GROUPS = (("crop", 4), ("color", 4))
AUGSELF_DIM = sum(n for _, n in AUGSELF_GROUPS)


def _mlp(in_dim, hidden, out_dim):
    """The paper's 3-layer MLP for omega_diff prediction."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class AugSelfHead(nn.Module):
    """One 3-layer MLP per augmentation group, over the concatenation [h1, h2]."""

    def __init__(self, feat_dim, hidden=2048, groups=AUGSELF_GROUPS):
        super().__init__()
        self.groups = tuple(groups)
        self.heads = nn.ModuleList([_mlp(2 * feat_dim, hidden, n)
                                    for _, n in self.groups])

    def forward(self, h1, h2):
        x = torch.cat([h1, h2], dim=1)                       # (N, 2*feat_dim), ordered
        return torch.cat([head(x) for head in self.heads], dim=1)   # (N, AUGSELF_DIM)
