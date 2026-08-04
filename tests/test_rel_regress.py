"""
Tests for the relpred_regress ablation: same views, same factors, same head
capacity as relpred, with an l2 target on the normalized parameter difference
instead of per-factor same/different.

The ablation only means something if it changes ONE thing, so the tests pin both
sides of that: the view generation must be identical to relpred's, and the head
must use the ordered pair (a signed difference is antisymmetric, so a symmetric
head could not represent it and would fail for the wrong reason).

Run:  python -m pytest pred_ssl/tests/test_rel_regress.py -q
"""

import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.data.transforms import (  # noqa: E402
    FACTORS, REGRESS_DIMS, REGRESS_TOTAL, RelPairTransform, build_transform,
    expand_mask, normalize_params, sample_factor_params)
from pred_ssl.losses import RelRegressLoss  # noqa: E402
from pred_ssl.models.rel_head import RelHead  # noqa: E402
from pred_ssl.relctl.config import _deep_merge, _load_yaml  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _img(seed=0, size=256):
    arr = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------

def test_dims_cover_every_factor_with_crop_carrying_four():
    assert set(REGRESS_DIMS) == set(FACTORS)
    assert REGRESS_DIMS["crop"] == 4
    assert REGRESS_TOTAL == len(FACTORS) - 1 + 4 == 12


def test_normalized_params_lie_in_unit_interval():
    rng = random.Random(0)
    for _ in range(50):
        p1, _, _ = sample_factor_params(rng=rng)
        v = normalize_params(p1)
        assert v.shape == (REGRESS_TOTAL,)
        assert np.isfinite(v).all()
        assert (v >= -1e-6).all() and (v <= 1 + 1e-6).all(), v


def test_shared_factors_give_exactly_zero_difference():
    # p_same=1 shares every parameter, so the regression target must vanish
    p1, p2, labels = sample_factor_params(p_same=1.0, rng=random.Random(3))
    assert labels.sum() == len(FACTORS)
    d = normalize_params(p1) - normalize_params(p2)
    assert np.allclose(d, 0.0)


def test_different_factors_give_nonzero_difference():
    p1, p2, labels = sample_factor_params(p_same=0.0, rng=random.Random(4))
    d = normalize_params(p1) - normalize_params(p2)
    assert np.abs(d).sum() > 0


def test_transform_returns_difference_and_expanded_mask():
    tf = RelPairTransform(regress=True)
    v1, v2, target, mask = tf(_img())
    assert v1.shape == (3, 224, 224)
    assert target.shape == (REGRESS_TOTAL,) and mask.shape == (REGRESS_TOTAL,)
    assert torch.isfinite(target).all()
    assert ((target >= -1) & (target <= 1)).all()


def test_expand_mask_repeats_crop_entries():
    m = np.ones(len(FACTORS), dtype=np.float32)
    m[FACTORS.index("saturation")] = 0.0
    e = expand_mask(m)
    assert e.shape == (REGRESS_TOTAL,)
    assert e.sum() == REGRESS_TOTAL - 1          # one masked scalar
    assert e[-4:].tolist() == [1.0, 1.0, 1.0, 1.0]   # crop expands to four


# ---------------------------------------------------------------------------
# The head must be ordered, not symmetric
# ---------------------------------------------------------------------------

def test_regression_head_is_view_order_sensitive():
    head = RelHead(32, num_factors=REGRESS_TOTAL, hidden=16, symmetric=False).eval()
    h1, h2 = torch.randn(4, 32), torch.randn(4, 32)
    assert not torch.allclose(head(h1, h2), head(h2, h1), atol=1e-4), \
        "a signed difference is antisymmetric; the head must see the ordered pair"


def test_default_head_stays_symmetric_for_relpred():
    head = RelHead(32, num_factors=9, hidden=16).eval()
    h1, h2 = torch.randn(4, 32), torch.randn(4, 32)
    assert torch.allclose(head(h1, h2), head(h2, h1), atol=1e-6)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def test_masked_l2_ignores_masked_entries():
    crit = RelRegressLoss()
    pred = torch.zeros(4, REGRESS_TOTAL)
    target = torch.ones(4, REGRESS_TOTAL)
    mask = torch.ones(4, REGRESS_TOTAL)
    assert abs(crit(pred, target, mask).item() - 1.0) < 1e-6
    mask[:, 0] = 0.0                      # masking a wrong entry must lower nothing else
    assert abs(crit(pred, target, mask).item() - 1.0) < 1e-6
    pred2 = target.clone()
    assert crit(pred2, target, mask).item() == 0.0


def test_loss_is_differentiable_and_finite_with_full_mask_zero():
    crit = RelRegressLoss()
    pred = torch.zeros(4, REGRESS_TOTAL, requires_grad=True)
    loss = crit(pred, torch.ones(4, REGRESS_TOTAL), torch.zeros(4, REGRESS_TOTAL))
    assert torch.isfinite(loss)           # clamped denominator, no division by zero
    loss.backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# Config: only the target differs from relpred
# ---------------------------------------------------------------------------

def _resolve(experiment):
    cfg = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "framework", "simclr.yaml")))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "experiment", experiment + ".yaml")))
    return cfg


def test_only_the_target_differs_from_relpred():
    r, g = _resolve("relpred"), _resolve("relpred_regress")
    for key in ("aug_sharing", "p_same", "delta", "rel_lambda", "rel_head_hidden",
                "crop_scale", "color_strength", "blur_mode"):
        assert r[key] == g[key], f"{key} differs: relpred={r[key]} regress={g[key]}"
    assert g["rel_regress"] is True and not r.get("rel_regress", False)
    assert g["grad_clip"] > 0, "an l2 target does not saturate; clipping is needed"
    assert isinstance(build_transform(g), RelPairTransform)
    assert build_transform(g).regress is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
