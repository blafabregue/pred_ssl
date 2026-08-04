"""
Tests for the AugSelf baseline (Lee et al., NeurIPS 2021) used as the closest-prior
comparison: parameter extraction, head shape/asymmetry, loss, and config wiring.

Fidelity points checked against the paper:
  - omega is normalized to [0,1]: crop (y_center, x_center, h, w) over the image size,
    colour (brightness, contrast, saturation, hue) over their sampling ranges;
  - the head is one 3-layer MLP per augmentation group over the ORDERED [h1, h2]
    (the target omega1 - omega2 is antisymmetric, unlike our symmetric relation);
  - the augmentation distribution is the standard one, so the comparison against
    `baseline` isolates the auxiliary loss.

Run:  python -m pytest pred_ssl/tests/test_augself.py -q
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.data.transforms import AugSelfTransform, build_transform  # noqa: E402
from pred_ssl.losses import AugSelfLoss  # noqa: E402
from pred_ssl.models.augself_head import AUGSELF_DIM, AugSelfHead  # noqa: E402
from pred_ssl.relctl.config import _deep_merge, _load_yaml  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _img(seed=0, size=256):
    arr = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# Transform: shapes, normalization, independence
# ---------------------------------------------------------------------------

def test_transform_returns_views_and_omegas():
    tf = AugSelfTransform()
    v1, v2, o1, o2 = tf(_img())
    assert v1.shape == (3, 224, 224) and v2.shape == (3, 224, 224)
    assert o1.shape == (AUGSELF_DIM,) and o2.shape == (AUGSELF_DIM,)
    assert o1.dtype == torch.float32


def test_omega_is_normalized_to_unit_interval():
    tf = AugSelfTransform()
    for seed in range(30):
        _, _, o1, o2 = tf(_img(seed=seed))
        for o in (o1, o2):
            assert torch.isfinite(o).all()
            assert (o >= -1e-6).all() and (o <= 1 + 1e-6).all(), o


def test_omega_normalization_is_correct():
    # crop entries are (y_center, x_center, h, w) / image size
    tf = AugSelfTransform()
    params = {"crop": (0, 0, 128, 64), "brightness": 1.0, "contrast": 0.6,
              "saturation": 1.4, "hue": 0.0}
    o = tf._omega(params, (256, 256))          # (width, height)
    assert abs(o[0] - 64 / 256) < 1e-6         # y_center = (0 + 128/2)/256
    assert abs(o[1] - 32 / 256) < 1e-6         # x_center = (0 + 64/2)/256
    assert abs(o[2] - 128 / 256) < 1e-6        # h
    assert abs(o[3] - 64 / 256) < 1e-6         # w
    # colour: brightness 1.0 is the midpoint of [0.6, 1.4] -> 0.5; contrast 0.6 -> 0;
    # saturation 1.4 -> 1; hue 0.0 is the midpoint of [-0.1, 0.1] -> 0.5
    assert abs(o[4] - 0.5) < 1e-6 and abs(o[5] - 0.0) < 1e-6
    assert abs(o[6] - 1.0) < 1e-6 and abs(o[7] - 0.5) < 1e-6


def test_views_are_independently_augmented():
    # AugSelf uses the standard pipeline: parameters are NOT shared across views,
    # so omega1 == omega2 should essentially never happen.
    tf = AugSelfTransform()
    identical = sum(torch.allclose(o1, o2)
                    for o1, o2 in (tf(_img(seed=s))[2:] for s in range(20)))
    assert identical == 0


# ---------------------------------------------------------------------------
# Head: shape and the antisymmetry that distinguishes it from our relational head
# ---------------------------------------------------------------------------

def test_head_shape_and_view_order_sensitivity():
    head = AugSelfHead(feat_dim=32, hidden=16).eval()
    h1, h2 = torch.randn(4, 32), torch.randn(4, 32)
    out = head(h1, h2)
    assert out.shape == (4, AUGSELF_DIM)
    # unlike RelHead, AugSelf's head must NOT be symmetric: its target flips sign
    assert not torch.allclose(out, head(h2, h1), atol=1e-4)


def test_head_has_one_mlp_per_group_with_three_layers():
    head = AugSelfHead(feat_dim=32, hidden=16)
    assert len(head.heads) == 2                                  # crop, color
    for mlp in head.heads:
        assert sum(1 for m in mlp if isinstance(m, torch.nn.Linear)) == 3


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_loss_applies_tanh_before_the_squared_error():
    # AugSelf's released SSObjective computes F.mse_loss(torch.tanh(p), d). The tanh
    # is load-bearing: it bounds the prediction to the target's range, which is what
    # keeps the objective stable. Omitting it diverges within a few iterations.
    crit = AugSelfLoss()
    o1, o2 = torch.rand(8, AUGSELF_DIM), torch.rand(8, AUGSELF_DIM)
    target = o1 - o2
    # a perfect prediction is atanh(target), not target itself
    assert crit(torch.atanh(target.clamp(-0.999, 0.999)), o1, o2).item() < 1e-5
    assert crit(target, o1, o2).item() > 0.0
    # at the origin tanh is the identity, so the loss is the plain MSE there
    pred = torch.zeros(8, AUGSELF_DIM, requires_grad=True)
    loss = crit(pred, o1, o2)
    assert abs(loss.item() - (target ** 2).mean().item()) < 1e-6
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_loss_is_bounded_for_arbitrarily_large_predictions():
    # the property the tanh buys: a drifting output cannot blow the loss up
    crit = AugSelfLoss()
    o1, o2 = torch.rand(8, AUGSELF_DIM), torch.rand(8, AUGSELF_DIM)
    huge = torch.full((8, AUGSELF_DIM), 1e6)
    assert crit(huge, o1, o2).item() < 4.0        # (tanh in [-1,1], target in [-1,1])


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

def test_augself_experiment_config_and_transform_selection():
    cfg = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "framework", "simclr.yaml")))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "experiment", "augself.yaml")))
    assert cfg["augself"] is True
    assert cfg["rel_lambda"] == 0.5          # the paper's IN-100 value
    assert cfg["aug_sharing"] is False       # standard, non-shared augmentation
    assert isinstance(build_transform(cfg), AugSelfTransform)
    # the default experiments keep augself off
    base = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    assert base["augself"] is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
