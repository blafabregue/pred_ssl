"""
The relational/pairwise same-different head.

Takes the two views' post-avgpool backbone features (h1, h2) and predicts, per
augmentation factor, whether the same parameter value was applied to both views.

The input is the SYMMETRIC combination [h1 + h2, |h1 - h2|] (dim 2*feat_dim), so the
head is invariant to swapping the two views — correct, because "same/different" is a
symmetric relation. The MLP is the 3-layer LayerNorm stack from the existing
SimCLR-pred-3layers aug classifier, with the input dim doubled and one binary output
per factor (9 with crop included; train.py passes len(FACTORS)).
"""

import torch
import torch.nn as nn


class RelHead(nn.Module):

    def __init__(self, feat_dim, num_factors=9, hidden=2048):
        super().__init__()
        self.num_factors = num_factors
        self.mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_factors),
        )

    def forward(self, h1, h2):
        x = torch.cat([h1 + h2, (h1 - h2).abs()], dim=1)  # (N, 2*feat_dim)
        return self.mlp(x)                                 # (N, num_factors)
