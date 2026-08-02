"""
E-SSL's transformation-classification head (Dangovski et al., 2022).

A multi-layer perceptron mapping the backbone features of the *predictor view* to
transformation-class logits. Both properties matter for a faithful baseline:

  - the head is multi-layer (the paper uses a 2-layer MLP with 2048 hidden units,
    LayerNorm and ReLU), not the single linear layer of the Section 4 stress test;
  - it reads a separate small crop rather than the contrastive view.

Together those are the implicit shortcut guards the paper never labels as such,
and removing them is exactly what produces the collapse we characterize.

With several factors (the paper's formulation extended by us to the full factor
set, reported as extended_essl) the head emits one classification group per factor
from the same shared trunk, mirroring how our relational head emits one logit per
factor from a shared MLP -- so the two differ in the target, not in capacity
allocation.

Contrast with models/rel_head.py: E-SSL's target is a property of ONE view, so the
head takes a single feature vector; our relational head takes both views.
"""

import torch.nn as nn


class ESSLHead(nn.Module):

    def __init__(self, feat_dim, num_classes=(4,), hidden=2048):
        super().__init__()
        if isinstance(num_classes, int):        # single-factor convenience
            num_classes = (num_classes,)
        self.num_classes = tuple(num_classes)
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.classifiers = nn.ModuleList([nn.Linear(hidden, c)
                                          for c in self.num_classes])

    def forward(self, h):
        """List of (N, C_f) logit tensors, one per predicted factor."""
        z = self.trunk(h)
        return [clf(z) for clf in self.classifiers]
