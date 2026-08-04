"""Common model-forward output contract shared by all frameworks."""

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    ssl_loss: torch.Tensor   # scalar, the framework's own SSL loss (computed inside forward)
    ssl_acc: float           # diagnostic SSL accuracy (e.g. InfoNCE top-1); 0.0 for BYOL
    h1: torch.Tensor         # (N, feat_dim) post-avgpool features of view1, trainable backbone
    h2: torch.Tensor         # (N, feat_dim) post-avgpool features of view2, trainable backbone
    # Raw value of a framework-internal decorrelation regularizer, for logging only:
    # it is ALREADY weighted into ssl_loss. 0.0 when the framework has none. Optional
    # and last, so the frameworks that predate it are unaffected.
    decov_loss: float = 0.0
