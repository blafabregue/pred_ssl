"""
E-SSL's transformation-classification head (Dangovski et al., 2022).

A multi-layer perceptron mapping the backbone features of the *predictor view* to
transformation-class logits. Both properties matter for a faithful baseline:

  - the head is multi-layer (the paper uses a 2-layer MLP with 2048 hidden units,
    LayerNorm and ReLU), not the single linear layer of the Section 4 stress test;
  - it reads a separate small crop rather than the contrastive view.

Together those are the implicit shortcut guards the paper never labels as such,
and removing them is exactly what produces the collapse we characterize.

Contrast with models/rel_head.py: E-SSL's target is a property of ONE view, so the
head takes a single feature vector; our relational head takes both views.
"""

import torch.nn as nn


class ESSLHead(nn.Module):

    def __init__(self, feat_dim, num_classes=4, hidden=2048):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, h):
        return self.mlp(h)              # (N, num_classes)
