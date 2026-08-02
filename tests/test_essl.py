"""
Tests for the E-SSL baseline (Dangovski et al., ICLR 2022).

E-SSL appears twice in this work and the two roles must not be confused:
  - in the shortcut-collapse study it is deliberately WEAKENED (linear predictor on
    the full-resolution view) to show what its safeguards protect against;
  - as a main-table baseline it must be faithful, i.e. keep both safeguards.
These tests pin the faithful version: a separate SMALLER predictor crop, and a
multi-layer head.

Run:  python -m pytest pred_ssl/tests/test_essl.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.data.transforms import ESSLTransform, build_transform  # noqa: E402
from pred_ssl.models.essl_head import ESSLHead  # noqa: E402
from pred_ssl.relctl.config import _deep_merge, _load_yaml  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _img(seed=0, size=256):
    arr = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# The safeguards: a separate crop, and a smaller one
# ---------------------------------------------------------------------------

def test_predictor_view_is_separate_and_smaller():
    tf = ESSLTransform(crop_size=112, out_size=224)
    v1, v2, pred, label = tf(_img())
    assert v1.shape == (3, 224, 224) and v2.shape == (3, 224, 224)
    assert pred.shape == (3, 112, 112), "the predictor crop must be smaller"
    # and it is a different draw, not one of the contrastive views downsampled
    assert not torch.allclose(pred, torch.nn.functional.interpolate(
        v1[None], size=112, mode="bilinear", align_corners=False)[0], atol=1e-3)


def test_label_is_a_valid_rotation_class_and_varies():
    tf = ESSLTransform()
    labels = [int(tf(_img(seed=s))[3][0]) for s in range(40)]
    assert all(0 <= k < ESSLTransform.N_ROT for k in labels)
    assert len(set(labels)) > 1, "the transformation must actually vary"


# ---------------------------------------------------------------------------
# extended_essl: our extension to the full factor set
# ---------------------------------------------------------------------------

EXT = ["rotation", "hflip", "brightness", "contrast", "saturation", "hue",
       "grayscale", "blur"]


def test_extended_emits_one_label_per_factor_in_range():
    tf = ESSLTransform(factors=EXT, bins=4)
    assert tf.num_classes == [4, 2, 4, 4, 4, 4, 2, 4]
    for s in range(20):
        _, _, _, labels = tf(_img(seed=s))
        assert labels.shape == (len(EXT),)
        for k, n in zip(labels.tolist(), tf.num_classes):
            assert 0 <= k < n


def test_binning_is_monotone_and_covers_the_range():
    from pred_ssl.data.transforms import essl_label
    # brightness in [0.6, 1.4] with 4 bins: the ends must land in the end bins
    assert essl_label("brightness", 0.6, 4) == 0
    assert essl_label("brightness", 1.4, 4) == 3
    lows = [essl_label("brightness", v, 4) for v in (0.65, 0.9, 1.15, 1.35)]
    assert lows == sorted(lows) and len(set(lows)) == 4
    # blur reserves bin 0 for "not applied"
    assert essl_label("blur", 0.0, 4) == 0
    assert essl_label("blur", 2.0, 4) == 3


def test_crop_has_no_per_view_target():
    from pred_ssl.data.transforms import essl_num_classes
    # not an omission: crop parameters are not identifiable from the cropped view
    with pytest.raises(ValueError):
        essl_num_classes("crop", 4)


def test_extended_config_is_ours_not_theirs():
    cfg = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "framework", "simclr.yaml")))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "experiment", "extended_essl.yaml")))
    assert cfg["essl"] is True and len(cfg["essl_factors"]) == 8
    assert "crop" not in cfg["essl_factors"]
    tf = build_transform(cfg)
    assert isinstance(tf, ESSLTransform) and len(tf.factors) == 8
    # the published variant stays single-factor
    pub = _load_yaml(os.path.join(CFG_DIR, "experiment", "essl.yaml"))
    assert pub.get("essl_factors", ["rotation"]) == ["rotation"]


def test_crop_size_is_configurable():
    for size in (64, 96, 112):
        _, _, pred, _ = ESSLTransform(crop_size=size)(_img())
        assert pred.shape == (3, size, size)


def test_contrastive_pair_is_the_standard_pipeline():
    # E-SSL leaves the contrastive branch untouched, so the two views must be a
    # plain independent pair (no shared-parameter structure).
    tf = ESSLTransform()
    v1, v2, _, _ = tf(_img())
    assert not torch.allclose(v1, v2)


# ---------------------------------------------------------------------------
# The head: multi-layer, single view
# ---------------------------------------------------------------------------

def test_head_is_multi_layer_and_single_view():
    head = ESSLHead(feat_dim=64, num_classes=4, hidden=32)
    linears = [m for m in head.trunk if isinstance(m, torch.nn.Linear)]
    assert len(linears) + len(head.classifiers) >= 2, \
        "E-SSL's predictor is multi-layer, not a linear probe"
    out = head(torch.randn(8, 64))
    assert len(out) == 1 and out[0].shape == (8, 4)


def test_head_emits_one_group_per_factor():
    head = ESSLHead(feat_dim=64, num_classes=[4, 2, 4], hidden=32)
    out = head(torch.randn(8, 64))
    assert [o.shape for o in out] == [(8, 4), (8, 2), (8, 4)]


def test_head_trains_under_cross_entropy():
    head = ESSLHead(feat_dim=64, num_classes=4, hidden=32)
    h = torch.randn(8, 64, requires_grad=True)
    loss = torch.nn.CrossEntropyLoss()(head(h)[0], torch.randint(0, 4, (8,)))
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

def test_essl_experiment_config():
    cfg = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "framework", "simclr.yaml")))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "experiment", "essl.yaml")))
    assert cfg["essl"] is True
    assert cfg["rel_lambda"] == 0.4          # the paper's auxiliary weight
    assert cfg["aug_sharing"] is False
    assert cfg["essl_crop_size"] < 224       # the safeguard
    assert isinstance(build_transform(cfg), ESSLTransform)
    # defaults keep it off, and it is exclusive with augself
    base = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    assert base["essl"] is False and base["augself"] is False
    assert not cfg.get("augself", False)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
