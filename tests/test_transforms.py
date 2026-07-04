"""
Phase 0 unit tests for the parameterized per-factor sharing loader.

Run (from the repo root, with deps available):
    python -m pytest pred_ssl/tests/test_transforms.py -q
or:
    python pred_ssl/tests/test_transforms.py
"""

import os
import random
import sys

import numpy as np
import torch
from PIL import Image

# Allow running both as `pytest pred_ssl/tests/...` and as a plain script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.data.transforms import (  # noqa: E402
    DEFAULT_DELTA,
    FACTORS,
    IDX,
    NUM_FACTORS,
    RelPairTransform,
    apply_pipeline,
    build_transform,
    compute_mask,
    sample_crop_box,
    sample_factor_params,
)

CONT = ("brightness", "contrast", "saturation", "hue")
DISCRETE = ("rotation", "hflip", "grayscale")


def _noise_image(seed=0, size=256):
    arr = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# (1) "same" -> identical parameters
# ---------------------------------------------------------------------------

def test_same_implies_identical_params():
    rng = random.Random(1)
    for _ in range(20000):
        p1, p2, labels = sample_factor_params(rng=rng)
        for f in FACTORS:
            if labels[IDX[f]] == 1.0:
                assert p1[f] == p2[f], f"'same' but params differ for {f}: {p1[f]} vs {p2[f]}"


# ---------------------------------------------------------------------------
# (2) "different" -> guaranteed-different (delta gap for continuous)
# ---------------------------------------------------------------------------

def test_different_is_guaranteed_different():
    from pred_ssl.data.transforms import _box_iou
    rng = random.Random(2)
    for _ in range(20000):
        p1, p2, labels = sample_factor_params(rng=rng)
        for f in DISCRETE:
            if labels[IDX[f]] == 0.0:
                assert p1[f] != p2[f], f"'different' but {f} equal: {p1[f]}"
        for f in CONT:
            if labels[IDX[f]] == 0.0:
                assert abs(p1[f] - p2[f]) >= DEFAULT_DELTA[f] - 1e-9, (
                    f"{f} gap {abs(p1[f]-p2[f]):.4f} < delta {DEFAULT_DELTA[f]}")
        if labels[IDX["blur"]] == 0.0:
            s1, s2 = p1["blur"], p2["blur"]
            applied_diff = (s1 > 0) != (s2 > 0)
            sigma_diff = (s1 > 0 and s2 > 0 and abs(s1 - s2) >= DEFAULT_DELTA["blur"] - 1e-9)
            assert applied_diff or sigma_diff, f"blur not guaranteed-different: {s1} vs {s2}"
        if labels[IDX["crop"]] == 0.0:
            iou = _box_iou(p1["crop"], p2["crop"])
            assert iou <= DEFAULT_DELTA["crop"] + 1e-9, (
                f"crop IoU {iou:.3f} > delta {DEFAULT_DELTA['crop']}")
        else:
            assert p1["crop"] == p2["crop"]


# ---------------------------------------------------------------------------
# (3) label vector: shape/dtype/domain + seed reproducibility
# ---------------------------------------------------------------------------

def test_label_vector_shape_and_reproducibility():
    p1a, p2a, la = sample_factor_params(rng=random.Random(123))
    p1b, p2b, lb = sample_factor_params(rng=random.Random(123))
    assert la.shape == (NUM_FACTORS,)
    assert la.dtype == np.float32
    assert set(np.unique(la)).issubset({0.0, 1.0})
    assert np.array_equal(la, lb), "same seed must reproduce identical labels"
    assert p1a == p1b and p2a == p2b, "same seed must reproduce identical params"
    # label bit is exactly the same/different relationship of the params
    for f in FACTORS:
        same = (p1a[f] == p2a[f])
        assert bool(la[IDX[f]]) == same, f"label/param mismatch for {f}"


# ---------------------------------------------------------------------------
# (4) mask rule: saturation/hue masked iff either view grayscale
# ---------------------------------------------------------------------------

def test_mask_rule_direct():
    def mk(g1, g2):
        return compute_mask({"grayscale": g1}, {"grayscale": g2})

    assert np.array_equal(mk(False, False), np.ones(NUM_FACTORS, dtype=np.float32))
    for g1, g2 in [(True, False), (False, True), (True, True)]:
        m = mk(g1, g2)
        assert m[IDX["saturation"]] == 0.0 and m[IDX["hue"]] == 0.0
        for f in FACTORS:
            if f not in ("saturation", "hue"):
                assert m[IDX[f]] == 1.0, f"{f} should never be masked"


def test_mask_rule_over_sampling():
    rng = random.Random(4)
    for _ in range(5000):
        p1, p2, _ = sample_factor_params(rng=rng)
        m = compute_mask(p1, p2)
        gray = p1["grayscale"] or p2["grayscale"]
        assert (m[IDX["saturation"]] == 0.0) == gray
        assert (m[IDX["hue"]] == 0.0) == gray
        assert m[IDX["brightness"]] == 1.0 and m[IDX["contrast"]] == 1.0


# ---------------------------------------------------------------------------
# (5) rendered views: shape, dtype, finiteness; labels/mask tensors
# ---------------------------------------------------------------------------

def test_render_shapes_and_finiteness():
    img = _noise_image()
    tf = RelPairTransform()
    for _ in range(10):
        v1, v2, labels, mask = tf(img)
        for v in (v1, v2):
            assert v.shape == (3, 224, 224)
            assert v.dtype == torch.float32
            assert torch.isfinite(v).all()
        assert labels.shape == (NUM_FACTORS,) and mask.shape == (NUM_FACTORS,)
        assert torch.isfinite(labels).all() and torch.isfinite(mask).all()


# ---------------------------------------------------------------------------
# (6) crop independence: identical params but independent crops -> views differ
# ---------------------------------------------------------------------------

def test_crop_is_independent():
    img = _noise_image(seed=7)
    # p_same=1.0 forces all 9 factors identical; an EXPLICIT crop_box argument
    # overrides params["crop"], so forcing different boxes must change the views.
    p1, p2, labels = sample_factor_params(p_same=1.0, rng=random.Random(7))
    assert labels.sum() == NUM_FACTORS  # all "same"
    box1 = (10, 10, 150, 150)
    box2 = (60, 40, 160, 170)
    v1 = apply_pipeline(img, p1, crop_box=box1)
    v2 = apply_pipeline(img, p2, crop_box=box2)
    assert not torch.allclose(v1, v2), "different crop boxes should yield different views"


# ---------------------------------------------------------------------------
# (7) STRONGEST: identical params AND identical crop -> byte-identical views
# ---------------------------------------------------------------------------

def test_determinism_same_params_same_crop():
    img = _noise_image(seed=11)
    p1, p2, labels = sample_factor_params(p_same=1.0, rng=random.Random(11))
    assert labels.sum() == NUM_FACTORS
    assert p1 == p2
    box = (12, 20, 180, 180)
    v1 = apply_pipeline(img, p1, crop_box=box)
    v2 = apply_pipeline(img, p2, crop_box=box)
    assert torch.allclose(v1, v2, atol=1e-6), "identical params + crop must be deterministic"


def test_shared_crop_renders_identical_views():
    # With p_same=1.0 the crop box in params is shared too, so rendering WITHOUT an
    # explicit crop_box must already be byte-identical across the two views.
    img = _noise_image(seed=17)
    p1, p2, labels = sample_factor_params(p_same=1.0, rng=random.Random(17))
    assert p1["crop"] == p2["crop"]
    v1 = apply_pipeline(img, p1)
    v2 = apply_pipeline(img, p2)
    assert torch.allclose(v1, v2, atol=1e-6), "shared crop param must yield identical views"


def test_crop_boxes_within_image():
    from pred_ssl.data.transforms import _box_iou  # noqa: F401 (import sanity)
    rng = random.Random(19)
    for w, h in [(256, 256), (300, 200), (200, 300), (64, 64)]:
        for _ in range(2000):
            p1, p2, _ = sample_factor_params(rng=rng, img_size=(w, h))
            for box in (p1["crop"], p2["crop"]):
                i, j, bh, bw = box
                assert 0 <= i and 0 <= j and bh > 0 and bw > 0
                assert i + bh <= h and j + bw <= w, f"box {box} exceeds image ({w}x{h})"


# ---------------------------------------------------------------------------
# (8) blur binary mode also yields guaranteed-different
# ---------------------------------------------------------------------------

def test_blur_binary_mode():
    rng = random.Random(8)
    for _ in range(5000):
        p1, p2, labels = sample_factor_params(blur_mode="binary", rng=rng)
        if labels[IDX["blur"]] == 0.0:
            assert (p1["blur"] > 0) != (p2["blur"] > 0)
        else:
            assert p1["blur"] == p2["blur"]


# ---------------------------------------------------------------------------
# (9) sanity: same/different roughly balanced at p_same=0.5
# ---------------------------------------------------------------------------

def test_same_fraction_is_balanced():
    rng = random.Random(9)
    n = 20000
    counts = np.zeros(NUM_FACTORS)
    for _ in range(n):
        _, _, labels = sample_factor_params(rng=rng)
        counts += labels
    frac = counts / n
    assert np.all(np.abs(frac - 0.5) < 0.03), f"same-fraction off-balance: {frac}"


# ---------------------------------------------------------------------------
# (10) decoupled transform: 6-tuple (independent SSL pair + separate rel pair)
# ---------------------------------------------------------------------------

def test_build_transform_selects_decoupled():
    from pred_ssl.data.transforms import DecoupledRelTransform, RelPairTransform as _RP
    # rel_decoupled wins over aug_sharing
    tf = build_transform({"rel_decoupled": True, "aug_sharing": True})
    assert isinstance(tf, DecoupledRelTransform)
    assert isinstance(build_transform({"aug_sharing": True}), _RP)


def test_decoupled_render_six_tuple():
    img = _noise_image(seed=13)
    tf = build_transform({"rel_decoupled": True})
    for _ in range(10):
        out = tf(img)
        assert len(out) == 6, "decoupled transform must return (v1,v2,u1,u2,labels,mask)"
        v1, v2, u1, u2, labels, mask = out
        for v in (v1, v2, u1, u2):
            assert v.shape == (3, 224, 224) and v.dtype == torch.float32
            assert torch.isfinite(v).all()
        assert labels.shape == (NUM_FACTORS,) and mask.shape == (NUM_FACTORS,)
        # labels describe the rel pair (u1,u2); mask follows the grayscale rule
        assert set(torch.unique(labels).tolist()).issubset({0.0, 1.0})


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
