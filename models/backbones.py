"""
ResNet backbones with an avgpool forward hook (the framework-agnostic mechanism
for capturing post-avgpool features h1, h2 used by the relational head).

Adapted from SimCLR-Imagenet-pred-3layers/simclr/builder.py: the encoder's `fc`
is replaced with Identity and a hook on `avgpool` stores the flattened pooled
feature on `encoder._feat`. The hook overwrites `_feat` on every forward, so the
caller must read it immediately after each forward (or read the backbone's direct
output, which equals `_feat` when fc=Identity).
"""

import torch.nn as nn
import torchvision.models as models

SUPPORTED = ("resnet18", "resnet50")


def build_backbone(arch="resnet50", hook=True):
    """Return (encoder, feat_dim).

    encoder.fc is Identity, so encoder(x) returns the (N, feat_dim) pooled feature
    directly. When hook=True an avgpool forward hook also stores that feature on
    encoder._feat (the framework-agnostic capture point used by SimCLR/MoCo/LooC).
    BYOL passes hook=False so its deepcopy'd target network carries no hook.
    feat_dim is 2048 for resnet50, 512 for resnet18.
    """
    if arch not in SUPPORTED:
        raise ValueError(f"arch must be one of {SUPPORTED}, got {arch}")
    encoder = models.__dict__[arch]()
    feat_dim = encoder.fc.in_features  # before replacing
    encoder.fc = nn.Identity()
    encoder._feat = None

    if hook:
        def _hook(module, inp, out):
            encoder._feat = out.flatten(1)

        encoder.avgpool.register_forward_hook(_hook)
    return encoder, feat_dim
