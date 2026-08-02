"""
Parameterized augmentation + per-factor sharing for relational/pairwise SSL.

The core mechanism (the genuinely new part of pred_ssl): for each of 9 augmentation
factors INDEPENDENTLY, with probability ``p_same`` apply the IDENTICAL parameter
value to both views (label = "same" = 1); otherwise apply a GUARANTEED-different
value (label = "different" = 0). CROP is one of the factors: "same" applies the
identical crop box to both views; "different" guarantees the two boxes overlap by
at most ``delta["crop"]`` IoU (so the difference is perceptible, mirroring the
minimum-gap rule of the continuous factors).

Factor order (the canonical label-vector index order):
    0 rotation, 1 hflip, 2 brightness, 3 contrast, 4 saturation, 5 hue,
    6 grayscale, 7 blur, 8 crop

Each sample returns ``(view1, view2, labels[9], mask[9])`` where ``mask`` zeroes out
the saturation/hue factors whenever EITHER view is grayscale (those factors are
unobservable on a desaturated image).

Augmentations are applied with ``torchvision.transforms.functional`` so each sampled
parameter is explicit and shareable. The fixed color-op order (brightness → contrast
→ saturation → hue) is required so that an identical parameter produces byte-identical
pixels across views (torchvision's ColorJitter randomizes sub-op order, which would
break that guarantee — hence we do NOT reuse it).
"""

import math
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import ImageFilter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACTORS = [
    "rotation", "hflip", "brightness", "contrast",
    "saturation", "hue", "grayscale", "blur", "crop",
]
IDX = {name: i for i, name in enumerate(FACTORS)}
NUM_FACTORS = len(FACTORS)

ROT_ANGLES = [0, 90, 180, 270]
SIGMA_RANGE = (0.1, 2.0)

# ImageNet normalization (matches every existing loader / eval script in the repo).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Minimum gap that makes a continuous "different" label perceptible / learnable.
# For crop the gap is expressed the other way around: "different" boxes must overlap
# by AT MOST delta["crop"] IoU (must stay >= crop_scale[0], the worst-case reachable
# IoU when one box covers the whole image).
DEFAULT_DELTA = {
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.05,
    "blur": 0.4,
    "crop": 0.4,
}

# Per-factor "applied" probabilities for the binary factors. These control the
# augmentation strength (kept close to standard SSL) while the same/different coin
# stays balanced at p_same. Rotation uses a uniform draw over the 4 angle buckets.
DEFAULT_P = {
    "hflip": 0.5,
    "grayscale": 0.2,   # matches RandomGrayscale(p=0.2) prevalence in standard SSL
    "blur": 0.5,        # matches RandomApply([GaussianBlur], p=0.5)
}


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_diff_continuous(lo, hi, v1, delta, rng, max_tries=100):
    """Sample a value in [lo, hi] that differs from v1 by at least ``delta``.

    Rejection sampling with a fallback to the farther endpoint (guarantees a
    >= delta gap for the ranges used here).
    """
    for _ in range(max_tries):
        v2 = rng.uniform(lo, hi)
        if abs(v2 - v1) >= delta:
            return v2
    # Fallback: pick the endpoint farther from v1 (range width >> delta, so safe).
    return lo if (v1 - lo) >= (hi - v1) else hi


def _sample_blur_pair(same, rng, delta_blur, blur_mode, p_blur):
    """Return (sigma1, sigma2). sigma == 0.0 means "blur not applied"."""
    lo, hi = SIGMA_RANGE
    applied1 = rng.random() < p_blur
    s1 = rng.uniform(lo, hi) if applied1 else 0.0
    if same:
        return s1, s1
    if blur_mode == "binary":
        # Flip the applied state only.
        s2 = 0.0 if applied1 else rng.uniform(lo, hi)
        return s1, s2
    # blur_mode == "sigma": guaranteed-different sigma value
    if not applied1:
        s2 = rng.uniform(lo, hi)            # 0.0 vs >0 -> different
    elif rng.random() < 0.5:
        s2 = 0.0                            # applied vs not -> different
    else:
        s2 = _sample_diff_continuous(lo, hi, s1, delta_blur, rng)
    return s1, s2


def _box_iou(b1, b2):
    """IoU of two (i, j, h, w) crop boxes."""
    i1, j1, h1, w1 = b1
    i2, j2, h2, w2 = b2
    inter_h = max(0, min(i1 + h1, i2 + h2) - max(i1, i2))
    inter_w = max(0, min(j1 + w1, j2 + w2) - max(j1, j2))
    inter = inter_h * inter_w
    union = h1 * w1 + h2 * w2 - inter
    return inter / union if union > 0 else 0.0


def _sample_crop_from_size(width, height, scale, ratio, rng):
    """RandomResizedCrop.get_params reimplemented for a bare (width, height).

    Same algorithm/distribution as torchvision (10 area/aspect tries, then the
    aspect-clamped center crop), but needing no image object so the pair can be
    sampled alongside the other factor parameters.
    """
    area = width * height
    log_lo, log_hi = math.log(ratio[0]), math.log(ratio[1])
    for _ in range(10):
        target_area = area * rng.uniform(scale[0], scale[1])
        aspect = math.exp(rng.uniform(log_lo, log_hi))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            i = rng.randint(0, height - h)
            j = rng.randint(0, width - w)
            return i, j, h, w
    # Fallback: center crop clamped to the aspect-ratio bounds.
    in_ratio = width / height
    if in_ratio < ratio[0]:
        w = width
        h = int(round(w / ratio[0]))
    elif in_ratio > ratio[1]:
        h = height
        w = int(round(h * ratio[1]))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


def _sample_crop_pair(same, rng, img_size, scale, ratio, max_iou, max_tries=100):
    """Return (box1, box2). same -> identical boxes; different -> IoU <= max_iou."""
    width, height = img_size
    b1 = _sample_crop_from_size(width, height, scale, ratio, rng)
    if same:
        return b1, b1
    for _ in range(max_tries):
        b2 = _sample_crop_from_size(width, height, scale, ratio, rng)
        if _box_iou(b1, b2) <= max_iou:
            return b1, b2
    # Fallback: a minimal-area square in the corner farthest from b1's center.
    # Worst case (b1 == full image) its IoU is crop_scale[0], hence the constraint
    # max_iou >= crop_scale[0] documented on DEFAULT_DELTA["crop"].
    side = max(1, int(round(math.sqrt(scale[0] * width * height))))
    h, w = min(side, height), min(side, width)
    ci, cj = b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0
    i = 0 if ci >= height / 2.0 else height - h
    j = 0 if cj >= width / 2.0 else width - w
    return b1, (i, j, h, w)


def sample_factor_params(
    p_same=0.5,
    color_strength=1.0,
    delta=None,
    blur_mode="sigma",
    p=None,
    rng=None,
    img_size=(256, 256),
    crop_scale=(0.2, 1.0),
    crop_ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """Sample the per-factor parameters for the two views.

    ``img_size`` is the (width, height) the crop boxes are sampled against — the
    PRE-crop image size (rotation preserves the canvas, so one box is valid for
    both views regardless of their rotation angles).

    Returns:
        params_v1, params_v2: dicts keyed by factor name (crop -> an (i, j, h, w) box).
        labels: float32[9], 1.0 == "same", 0.0 == "different" (FACTORS order).
    """
    if rng is None:
        rng = random
    if delta is None:
        delta = DEFAULT_DELTA
    if p is None:
        p = DEFAULT_P

    s = color_strength
    cj_lo, cj_hi = 1.0 - 0.4 * s, 1.0 + 0.4 * s
    hue_lim = 0.1 * s

    p1, p2 = {}, {}
    labels = np.zeros(NUM_FACTORS, dtype=np.float32)

    def coin():
        return rng.random() < p_same

    # --- rotation (discrete) ---
    same = coin()
    a1 = rng.choice(ROT_ANGLES)
    a2 = a1 if same else rng.choice([a for a in ROT_ANGLES if a != a1])
    p1["rotation"], p2["rotation"] = a1, a2
    labels[IDX["rotation"]] = float(same)

    # --- hflip (binary) ---
    same = coin()
    f1 = rng.random() < p["hflip"]
    f2 = f1 if same else (not f1)
    p1["hflip"], p2["hflip"] = f1, f2
    labels[IDX["hflip"]] = float(same)

    # --- brightness / contrast / saturation (continuous, multiplicative, identity=1) ---
    for name in ("brightness", "contrast", "saturation"):
        same = coin()
        v1 = rng.uniform(cj_lo, cj_hi)
        v2 = v1 if same else _sample_diff_continuous(cj_lo, cj_hi, v1, delta[name], rng)
        p1[name], p2[name] = v1, v2
        labels[IDX[name]] = float(same)

    # --- hue (continuous, additive, identity=0) ---
    same = coin()
    v1 = rng.uniform(-hue_lim, hue_lim)
    v2 = v1 if same else _sample_diff_continuous(-hue_lim, hue_lim, v1, delta["hue"], rng)
    p1["hue"], p2["hue"] = v1, v2
    labels[IDX["hue"]] = float(same)

    # --- grayscale (binary) ---
    same = coin()
    g1 = rng.random() < p["grayscale"]
    g2 = g1 if same else (not g1)
    p1["grayscale"], p2["grayscale"] = g1, g2
    labels[IDX["grayscale"]] = float(same)

    # --- blur (continuous sigma, 0 == not applied) ---
    same = coin()
    s1, s2 = _sample_blur_pair(same, rng, delta["blur"], blur_mode, p["blur"])
    p1["blur"], p2["blur"] = s1, s2
    labels[IDX["blur"]] = float(same)

    # --- crop (box; "different" == IoU <= delta["crop"]) ---
    same = coin()
    b1, b2 = _sample_crop_pair(same, rng, img_size, crop_scale, crop_ratio,
                               delta["crop"])
    p1["crop"], p2["crop"] = b1, b2
    labels[IDX["crop"]] = float(same)

    return p1, p2, labels


def compute_mask(params_v1, params_v2):
    """saturation/hue are unobservable when either view is grayscale -> mask them."""
    mask = np.ones(NUM_FACTORS, dtype=np.float32)
    if params_v1["grayscale"] or params_v2["grayscale"]:
        mask[IDX["saturation"]] = 0.0
        mask[IDX["hue"]] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def sample_crop_box(img, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    """Independent RandomResizedCrop parameters (i, j, h, w)."""
    return transforms.RandomResizedCrop.get_params(img, list(scale), list(ratio))


def apply_pipeline(
    img,
    params,
    crop_box=None,
    scale=(0.2, 1.0),
    out_size=224,
    mean=MEAN,
    std=STD,
):
    """Render one view deterministically from explicit per-factor params.

    Order: rotation -> crop -> hflip -> brightness -> contrast ->
    saturation -> hue -> grayscale -> blur -> ToTensor -> Normalize.
    Crop-box precedence: the explicit ``crop_box`` argument (tests) wins, then
    ``params["crop"]`` (the shared/different pair from sample_factor_params), and
    only if both are absent is an independent box sampled here.
    """
    # rotation (before crop, on the full image) -- matches existing RandomRotation90
    angle = params["rotation"]
    if angle != 0:
        img = img.rotate(angle)

    # crop
    if crop_box is None:
        crop_box = params.get("crop")
    if crop_box is None:
        crop_box = sample_crop_box(img, scale=scale)
    i, j, h, w = crop_box
    img = TF.resized_crop(img, i, j, h, w, [out_size, out_size])

    # hflip
    if params["hflip"]:
        img = TF.hflip(img)

    # color jitter as explicit ops in a FIXED order
    img = TF.adjust_brightness(img, params["brightness"])
    img = TF.adjust_contrast(img, params["contrast"])
    img = TF.adjust_saturation(img, params["saturation"])
    img = TF.adjust_hue(img, params["hue"])

    # grayscale
    if params["grayscale"]:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)

    # blur (sigma == 0 -> skip)
    sigma = params["blur"]
    if sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    t = TF.to_tensor(img)
    t = TF.normalize(t, mean, std)
    return t


# ---------------------------------------------------------------------------
# Auxiliary augmentations reused by the standard (baseline) transform
# ---------------------------------------------------------------------------

class RandomRotation90:
    """Random rotation from {90, 180, 270} (never 0). Copied from the existing loaders."""

    def __call__(self, img):
        return img.rotate(random.choice([90, 180, 270]))


class GaussianBlur:
    """SimCLR / MoCo v2 style Gaussian blur. Copied from the existing loaders."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


# ---------------------------------------------------------------------------
# Transform objects
# ---------------------------------------------------------------------------

class RelPairTransform:
    """Returns (view1, view2, labels[9], mask[9]) with per-factor sharing."""

    def __init__(
        self,
        p_same=0.5,
        color_strength=1.0,
        delta=None,
        blur_mode="sigma",
        crop_scale=(0.2, 1.0),
        out_size=224,
        p=None,
        mean=MEAN,
        std=STD,
    ):
        self.p_same = p_same
        self.color_strength = color_strength
        self.delta = delta if delta is not None else dict(DEFAULT_DELTA)
        self.blur_mode = blur_mode
        self.crop_scale = tuple(crop_scale)
        self.out_size = out_size
        self.p = p if p is not None else dict(DEFAULT_P)
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = img.convert("RGB")
        p1, p2, labels = sample_factor_params(
            p_same=self.p_same,
            color_strength=self.color_strength,
            delta=self.delta,
            blur_mode=self.blur_mode,
            p=self.p,
            img_size=img.size,
            crop_scale=self.crop_scale,
        )
        v1 = apply_pipeline(img, p1, crop_box=None, scale=self.crop_scale,
                            out_size=self.out_size, mean=self.mean, std=self.std)
        v2 = apply_pipeline(img, p2, crop_box=None, scale=self.crop_scale,
                            out_size=self.out_size, mean=self.mean, std=self.std)
        mask = compute_mask(p1, p2)
        return v1, v2, torch.from_numpy(labels), torch.from_numpy(mask)


# --- E-SSL per-view targets -------------------------------------------------
# Factors whose value is a discrete class, so E-SSL's formulation applies directly.
ESSL_DISCRETE = {"rotation": len(ROT_ANGLES), "hflip": 2, "grayscale": 2}
# Continuous factors have no discrete class, so per-view prediction requires binning
# (E-SSL does this for blur: their "four-fold blur" cuts a continuous sigma into 4).
# The crop is deliberately absent: its parameters are not identifiable from the
# cropped view alone -- one would need the original image as a reference -- so no
# per-view target exists for it. Only pair-based formulations can cover it.
ESSL_CONTINUOUS = ("brightness", "contrast", "saturation", "hue", "blur")


def essl_num_classes(factor, bins):
    """Number of per-view classes E-SSL would predict for a factor."""
    if factor in ESSL_DISCRETE:
        return ESSL_DISCRETE[factor]
    if factor in ESSL_CONTINUOUS:
        return bins
    raise ValueError(f"factor '{factor}' has no per-view E-SSL target "
                     f"(the crop is not identifiable from a single view)")


def essl_label(factor, value, bins, color_strength=1.0):
    """The per-view class of a factor's realized parameter."""
    if factor == "rotation":
        return ROT_ANGLES.index(value)
    if factor in ("hflip", "grayscale"):
        return int(bool(value))
    s = color_strength
    if factor == "blur":
        # bin 0 = not applied, then the applied sigma range split over the rest
        if value <= 0:
            return 0
        lo, hi = SIGMA_RANGE
        k = int((value - lo) / (hi - lo) * (bins - 1))
        return 1 + min(max(k, 0), bins - 2)
    lo, hi = (-0.1 * s, 0.1 * s) if factor == "hue" else (1 - 0.4 * s, 1 + 0.4 * s)
    k = int((value - lo) / (hi - lo) * bins)
    return min(max(k, 0), bins - 1)


class ESSLTransform:
    """E-SSL view generation (Dangovski et al., 2022), with its safeguards intact.

    E-SSL leaves the contrastive pipeline untouched and adds a *separate* predictor
    view: a smaller crop of the same image, to which one transformation is applied,
    and whose class the auxiliary head must name. The separate small crop is the
    design choice that keeps the task out of shortcut range -- it is too small to
    carry the seam and frequency artifacts that Section 4 shows a full-resolution
    predictor exploits -- so reproducing it faithfully is the whole point of using
    E-SSL as a baseline rather than as the stress test of Section 4.

    Returns ``(view1, view2, pred_view, labels)`` where ``labels`` holds one class per
    predicted factor. With the default ``essl_factors=("rotation",)`` this is E-SSL as
    published; naming more factors gives the ``extended_essl`` variant of this paper,
    which is our extension rather than theirs -- it requires binning the continuous
    factors, a choice their formulation forces but never had to make.
    """

    N_ROT = 4

    def __init__(self, crop_size=112, factors=("rotation",), bins=4,
                 color_strength=1.0, crop_scale=(0.2, 1.0),
                 out_size=224, p=None, mean=MEAN, std=STD):
        self.factors = tuple(factors)
        self.bins = bins
        self.num_classes = [essl_num_classes(f, bins) for f in self.factors]
        self.crop_size = crop_size
        self.p = p if p is not None else dict(DEFAULT_P)
        self.color_strength = color_strength
        self.crop_scale = tuple(crop_scale)
        self.out_size = out_size
        self.mean, self.std = mean, std
        self.ssl_transform = StandardTwoViewTransform(
            use_rotation=False, use_color=True, color_strength=color_strength,
            crop_scale=crop_scale, out_size=out_size, mean=mean, std=std)

    def _pred_params(self, rng):
        s = self.color_strength
        lo, hi = 1.0 - 0.4 * s, 1.0 + 0.4 * s
        return {
            "rotation": 0,                      # applied separately, it is the label
            "hflip": rng.random() < self.p["hflip"],
            "brightness": rng.uniform(lo, hi), "contrast": rng.uniform(lo, hi),
            "saturation": rng.uniform(lo, hi), "hue": rng.uniform(-0.1 * s, 0.1 * s),
            "grayscale": rng.random() < self.p["grayscale"],
            "blur": rng.uniform(*SIGMA_RANGE) if rng.random() < self.p["blur"] else 0.0,
        }

    def __call__(self, img):
        img = img.convert("RGB")
        v1, v2, _, _ = self.ssl_transform(img)
        rng = random
        # the predictor's own small crop, augmented independently of the SSL pair
        params = self._pred_params(rng)
        params["crop"] = _sample_crop_from_size(img.size[0], img.size[1],
                                                self.crop_scale,
                                                (3.0 / 4.0, 4.0 / 3.0), rng)
        # the transformation(s) to be predicted are drawn here so their realized
        # values are known exactly
        params["rotation"] = ROT_ANGLES[rng.randrange(self.N_ROT)]
        pred = apply_pipeline(img, params, scale=self.crop_scale,
                              out_size=self.crop_size, mean=self.mean, std=self.std)
        labels = torch.tensor(
            [essl_label(f, params[f], self.bins, self.color_strength)
             for f in self.factors], dtype=torch.long)
        return v1, v2, pred, labels


# LooC groups several of our fine-grained factors into one "augmentation" -- the
# paper treats colour jittering as a single augmentation, whereas we parameterize
# its four components separately.
LOOC_GROUPS = {
    "rotation":  ("rotation",),
    "color":     ("brightness", "contrast", "saturation", "hue"),
    "hflip":     ("hflip",),
    "grayscale": ("grayscale",),
    "blur":      ("blur",),
    "crop":      ("crop",),
}


def resample_different(params, factor_names, color_strength=1.0, delta=None,
                       blur_mode="sigma", p=None, img_size=(256, 256),
                       crop_scale=(0.2, 1.0), crop_ratio=(3.0 / 4.0, 4.0 / 3.0),
                       rng=None):
    """Copy ``params`` with the named factors redrawn guaranteed-different.

    This is LooC's key-view construction: a key that shares every augmentation
    parameter with the query except the one its embedding space is meant to remain
    sensitive to.
    """
    if rng is None:
        rng = random
    if delta is None:
        delta = DEFAULT_DELTA
    if p is None:
        p = DEFAULT_P
    out = dict(params)
    s = color_strength
    cj_lo, cj_hi = 1.0 - 0.4 * s, 1.0 + 0.4 * s
    hue_lim = 0.1 * s

    for name in factor_names:
        v = params[name]
        if name == "rotation":
            out[name] = rng.choice([a for a in ROT_ANGLES if a != v])
        elif name in ("hflip", "grayscale"):
            out[name] = not v
        elif name in ("brightness", "contrast", "saturation"):
            out[name] = _sample_diff_continuous(cj_lo, cj_hi, v, delta[name], rng)
        elif name == "hue":
            out[name] = _sample_diff_continuous(-hue_lim, hue_lim, v, delta["hue"], rng)
        elif name == "blur":
            _, out[name] = _sample_blur_pair(False, rng, delta["blur"], blur_mode,
                                             p["blur"])
        elif name == "crop":
            _, out[name] = _sample_crop_pair(False, rng, img_size, crop_scale,
                                             crop_ratio, delta["crop"])
        else:
            raise ValueError(f"unknown factor: {name}")
    return out


class LooCTransform:
    """Multi-view generation for LooC (Xiao et al., 2021).

    LooC keeps one embedding space per augmentation, each invariant to every
    augmentation *except* its own. That requires one key view per space:

      - ``key0``  (all-invariant space): augmented independently of the query, the
        standard contrastive pair;
      - ``key_a`` (space of augmentation ``a``): shares every parameter with the
        query except those of group ``a``, which are redrawn guaranteed-different,
        so the only way to match query and key in that space is to encode ``a``.

    Returns ``(query, key0, key_a1, ..., key_an, labels[F], mask[F])``. The labels
    describe the ``(query, key0)`` pair, so the relational head sees the same kind
    of pair as under every other framework; with ``aug_sharing=False`` that pair is
    independently augmented, which is LooC's own setting.
    """

    def __init__(self, looc_augs=("rotation", "color"), aug_sharing=True,
                 p_same=0.5, color_strength=1.0, delta=None, blur_mode="sigma",
                 crop_scale=(0.2, 1.0), out_size=224, p=None, mean=MEAN, std=STD):
        for g in looc_augs:
            if g not in LOOC_GROUPS:
                raise ValueError(f"unknown LooC augmentation group '{g}'; "
                                 f"choose from {sorted(LOOC_GROUPS)}")
        self.looc_augs = tuple(looc_augs)
        self.aug_sharing = aug_sharing
        self.p_same = p_same
        self.color_strength = color_strength
        self.delta = delta if delta is not None else dict(DEFAULT_DELTA)
        self.blur_mode = blur_mode
        self.crop_scale = tuple(crop_scale)
        self.out_size = out_size
        self.p = p if p is not None else dict(DEFAULT_P)
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = img.convert("RGB")
        kw = dict(color_strength=self.color_strength, delta=self.delta,
                  blur_mode=self.blur_mode, p=self.p, img_size=img.size,
                  crop_scale=self.crop_scale)
        # query and the all-invariant key: per-factor sharing when the relational
        # head is active, independent draws otherwise (LooC's own setting).
        pq, pk0, labels = sample_factor_params(
            p_same=self.p_same if self.aug_sharing else 0.0, **kw)
        if not self.aug_sharing:
            _, pk0, _ = sample_factor_params(p_same=0.0, **kw)
            labels = np.zeros(NUM_FACTORS, dtype=np.float32)

        views = [
            apply_pipeline(img, pq, scale=self.crop_scale, out_size=self.out_size,
                           mean=self.mean, std=self.std),
            apply_pipeline(img, pk0, scale=self.crop_scale, out_size=self.out_size,
                           mean=self.mean, std=self.std),
        ]
        # one leave-one-out key per LooC space
        for g in self.looc_augs:
            pk = resample_different(pq, LOOC_GROUPS[g], **kw)
            views.append(apply_pipeline(img, pk, scale=self.crop_scale,
                                        out_size=self.out_size,
                                        mean=self.mean, std=self.std))
        mask = compute_mask(pq, pk0)
        return (*views, torch.from_numpy(labels), torch.from_numpy(mask))


class AugSelfTransform:
    """Standard independent two-view augmentation + each view's augmentation parameters.

    Faithful data side of AugSelf (Lee et al., 2021): the augmentation pipeline is the
    STANDARD one (identical distribution to ``StandardTwoViewTransform``, so the
    comparison against the baseline isolates the auxiliary loss), and what the transform
    additionally returns is the parameter vector of each view, normalized to [0, 1] as
    the paper specifies:

      - crop  ``(y_center, x_center, h, w)`` divided by the original image size;
      - color ``(brightness, contrast, saturation, hue)`` mapped from their sampling
        ranges onto [0, 1].

    The head then regresses ``omega_1 - omega_2`` under an l2 loss. Grayscale and blur
    are still applied (as in the reference pipeline) but not predicted: the paper's
    default predicted set is A_AugSelf = {crop, color}.

    Returns ``(view1, view2, omega1[8], omega2[8])``.
    """

    N_PARAMS = 8   # 4 crop + 4 color

    def __init__(self, color_strength=1.0, crop_scale=(0.2, 1.0), out_size=224,
                 p=None, mean=MEAN, std=STD):
        self.color_strength = color_strength
        self.crop_scale = tuple(crop_scale)
        self.out_size = out_size
        self.p = p if p is not None else dict(DEFAULT_P)
        self.mean = mean
        self.std = std

    def _sample_view(self, img, rng):
        """Independent parameters for one view (standard pipeline, no sharing)."""
        s = self.color_strength
        cj_lo, cj_hi = 1.0 - 0.4 * s, 1.0 + 0.4 * s
        hue_lim = 0.1 * s
        params = {
            "rotation": 0,
            "hflip": rng.random() < self.p["hflip"],
            "brightness": rng.uniform(cj_lo, cj_hi),
            "contrast": rng.uniform(cj_lo, cj_hi),
            "saturation": rng.uniform(cj_lo, cj_hi),
            "hue": rng.uniform(-hue_lim, hue_lim),
            "grayscale": rng.random() < self.p["grayscale"],
            "blur": rng.uniform(*SIGMA_RANGE) if rng.random() < self.p["blur"] else 0.0,
        }
        params["crop"] = _sample_crop_from_size(img.size[0], img.size[1],
                                                self.crop_scale,
                                                (3.0 / 4.0, 4.0 / 3.0), rng)
        return params

    def _omega(self, params, img_size):
        """Normalized parameter vector omega in [0, 1]^8 (crop then color)."""
        width, height = img_size
        i, j, h, w = params["crop"]
        s = self.color_strength
        cj_lo, cj_hi = 1.0 - 0.4 * s, 1.0 + 0.4 * s
        hue_lim = 0.1 * s
        rng_cj = cj_hi - cj_lo
        return np.array([
            (i + h / 2.0) / height,          # y_center
            (j + w / 2.0) / width,           # x_center
            h / height,                      # height
            w / width,                       # width
            (params["brightness"] - cj_lo) / rng_cj,
            (params["contrast"] - cj_lo) / rng_cj,
            (params["saturation"] - cj_lo) / rng_cj,
            (params["hue"] + hue_lim) / (2 * hue_lim),
        ], dtype=np.float32)

    def __call__(self, img):
        img = img.convert("RGB")
        rng = random
        p1, p2 = self._sample_view(img, rng), self._sample_view(img, rng)
        v1 = apply_pipeline(img, p1, scale=self.crop_scale, out_size=self.out_size,
                            mean=self.mean, std=self.std)
        v2 = apply_pipeline(img, p2, scale=self.crop_scale, out_size=self.out_size,
                            mean=self.mean, std=self.std)
        return (v1, v2,
                torch.from_numpy(self._omega(p1, img.size)),
                torch.from_numpy(self._omega(p2, img.size)))


class StandardTwoViewTransform:
    """Standard independent two-view augmentation (the existing baseline pipeline).

    Returns (view1, view2, zeros(8), zeros(8)) so the training loop's tuple shape is
    uniform with RelPairTransform. Used for the primary baseline (aug_sharing=false).
    """

    def __init__(self, use_rotation=False, use_color=True, color_strength=1.0,
                 crop_scale=(0.2, 1.0), out_size=224, mean=MEAN, std=STD):
        aug = []
        if use_rotation:
            aug.append(transforms.RandomApply([RandomRotation90()], p=0.5))
        aug.extend([
            transforms.RandomResizedCrop(out_size, scale=tuple(crop_scale)),
            transforms.RandomHorizontalFlip(),
        ])
        if use_color:
            s = color_strength
            aug.append(transforms.RandomApply(
                [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)], p=0.8))
            aug.append(transforms.RandomGrayscale(p=0.2))
        aug.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        aug.append(transforms.ToTensor())
        aug.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(aug)

    def __call__(self, img):
        img = img.convert("RGB")
        v1 = self.transform(img)
        v2 = self.transform(img)
        return v1, v2, torch.zeros(NUM_FACTORS), torch.zeros(NUM_FACTORS)


class DecoupledRelTransform:
    """Decoupled relpred: the SSL pair and the relational pair are INDEPENDENT draws.

    Returns ``(view1, view2, rel1, rel2, labels[8], mask[8])`` where:
      - ``(view1, view2)`` is a STANDARD independent two-view augmentation — the
        unchanged contrastive signal (same distribution as the baseline);
      - ``(rel1, rel2)`` is a SEPARATE per-factor shared/different pair that feeds
        ONLY the relational head, and ``labels``/``mask`` describe *that* pair.

    This removes the confound in ``RelPairTransform`` (where one shared/different pair
    drives both the contrastive loss and the head, so ``p_same`` makes positives more
    alike and weakens the SSL signal). The cost is two extra backbone forwards/step.
    """

    def __init__(self, ssl_transform, rel_transform):
        self.ssl_transform = ssl_transform
        self.rel_transform = rel_transform

    def __call__(self, img):
        v1, v2, _, _ = self.ssl_transform(img)
        u1, u2, labels, mask = self.rel_transform(img)
        return v1, v2, u1, u2, labels, mask


def build_transform(cfg):
    """Select the pretraining transform from a config dict."""
    if cfg.get("framework") == "looc" and cfg.get("full_multiview", False):
        return LooCTransform(
            looc_augs=cfg.get("looc_augs", ("rotation", "color")),
            aug_sharing=cfg.get("aug_sharing", True) and cfg.get("rel_lambda", 0.0) > 0,
            p_same=cfg.get("p_same", 0.5),
            color_strength=cfg.get("color_strength", 1.0),
            delta=cfg.get("delta"),
            blur_mode=cfg.get("blur_mode", "sigma"),
            crop_scale=cfg.get("crop_scale", (0.2, 1.0)),
        )
    if cfg.get("essl", False):
        return ESSLTransform(
            crop_size=cfg.get("essl_crop_size", 112),
            factors=cfg.get("essl_factors", ("rotation",)),
            bins=cfg.get("essl_bins", 4),
            color_strength=cfg.get("color_strength", 1.0),
            crop_scale=cfg.get("crop_scale", (0.2, 1.0)),
        )
    if cfg.get("augself", False):
        return AugSelfTransform(
            color_strength=cfg.get("color_strength", 1.0),
            crop_scale=cfg.get("crop_scale", (0.2, 1.0)),
        )
    if cfg.get("rel_decoupled", False):
        return DecoupledRelTransform(
            StandardTwoViewTransform(
                use_rotation=cfg.get("use_rotation", False),
                use_color=cfg.get("use_color", True),
                color_strength=cfg.get("color_strength", 1.0),
                crop_scale=cfg.get("crop_scale", (0.2, 1.0)),
            ),
            RelPairTransform(
                p_same=cfg.get("p_same", 0.5),
                color_strength=cfg.get("color_strength", 1.0),
                delta=cfg.get("delta"),
                blur_mode=cfg.get("blur_mode", "sigma"),
                crop_scale=cfg.get("crop_scale", (0.2, 1.0)),
            ),
        )
    if cfg.get("aug_sharing", True):
        return RelPairTransform(
            p_same=cfg.get("p_same", 0.5),
            color_strength=cfg.get("color_strength", 1.0),
            delta=cfg.get("delta"),
            blur_mode=cfg.get("blur_mode", "sigma"),
            crop_scale=cfg.get("crop_scale", (0.2, 1.0)),
        )
    return StandardTwoViewTransform(
        use_rotation=cfg.get("use_rotation", False),
        use_color=cfg.get("use_color", True),
        color_strength=cfg.get("color_strength", 1.0),
        crop_scale=cfg.get("crop_scale", (0.2, 1.0)),
    )


def worker_init_fn(worker_id):
    """Decorrelate + reproduce the per-factor sharing coins across DataLoader workers."""
    info = torch.utils.data.get_worker_info()
    base = (info.seed if info is not None else 0) % (2 ** 31)
    seed = (base + worker_id) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
