"""
Tests for the positives-only framework and its collapse diagnostics.

This experiment is unusual in that its interesting outcome may be a FAILURE: a
representation that satisfies both losses while encoding nothing. So the tests pin
the two things that make such a result readable rather than just puzzling.

  1. The setup really is positives-only -- no negatives, no predictor, no momentum
     target, no stop-gradient. If any of those crept in, a non-collapse would prove
     nothing, and the whole point is what happens without them.

  2. The diagnostics can actually see both failure modes. std_ratio catches the
     constant representation; erank catches the low-rank one that encodes only the
     augmentation parameters. A test that only checked complete collapse would
     pass on exactly the outcome we are trying to detect.

Run:  python -m pytest pred_ssl/tests/test_posonly.py -q
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.collapse import collapse_stats, format_stats  # noqa: E402
from pred_ssl.data.transforms import (  # noqa: E402
    FACTORS, GEOMETRIC_FACTORS, PHOTOMETRIC_FACTORS, REGRESS_DIMS, REGRESS_TOTAL,
    RelPairTransform, build_transform, factor_select_mask)
from pred_ssl.losses import AlignmentLoss  # noqa: E402
from pred_ssl.models.frameworks import (  # noqa: E402
    backbone_state_dict, build_model, encode_features)
from pred_ssl.relctl.config import _deep_merge, _load_yaml  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _img(seed=0, size=256):
    arr = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


def _cfg(experiment="posonly_all", framework="posonly"):
    cfg = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "framework", framework + ".yaml")))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "experiment", experiment + ".yaml")))
    cfg["arch"] = "resnet18"
    return cfg


# ---------------------------------------------------------------------------
# The framework is genuinely positives-only
# ---------------------------------------------------------------------------

def test_model_has_no_anti_collapse_machinery():
    model = build_model(_cfg())
    names = [n for n, _ in model.named_modules()]
    for forbidden in ("predictor", "target_backbone", "target_projector", "queue"):
        assert not any(forbidden in n for n in names), f"{forbidden} defeats the point"
    assert not hasattr(model, "queue")
    # every parameter must receive gradient: no stop-gradient branch anywhere
    assert all(p.requires_grad for p in model.parameters())


def test_forward_shapes_and_gradient_reaches_the_backbone():
    model = build_model(_cfg()).train()
    v1, v2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    out = model(v1, v2)
    assert out.h1.shape == (4, 512) and out.h2.shape == (4, 512)
    assert torch.isfinite(out.ssl_loss)
    out.ssl_loss.backward()
    grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_alignment_loss_is_symmetric_and_minimized_by_agreement():
    crit = AlignmentLoss()
    z1 = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    z2 = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    assert abs(crit(z1, z2).item() - crit(z2, z1).item()) < 1e-6
    assert crit(z1, z1).item() < 1e-5                  # perfect agreement -> 0
    assert crit(z1, -z1).item() > 3.9                  # opposite -> 4


def test_alignment_alone_is_minimized_by_the_constant_solution():
    # the property that makes the auxiliary term load-bearing rather than optional
    crit = AlignmentLoss()
    const = torch.nn.functional.normalize(torch.ones(8, 16), dim=1)
    assert crit(const, const).item() < 1e-5


def test_framework_is_registered_for_eval_and_feature_extraction():
    cfg = _cfg()
    model = build_model(cfg)
    sd = backbone_state_dict(model, "posonly")
    assert sd and all(k.startswith("backbone.") for k in sd)
    assert not any(k.startswith("backbone.fc.") for k in sd)
    f = encode_features(model.eval(), "posonly", torch.randn(2, 3, 32, 32))
    assert f.shape == (2, 512)


def test_projector_batchnorm_is_switchable():
    # the BN confound: a non-collapse with BN on is not attributable to the head
    has_bn = lambda m: any(isinstance(x, torch.nn.BatchNorm1d)  # noqa: E731
                           for x in m.projector.modules())
    assert has_bn(build_model(_cfg()))
    cfg = _cfg()
    cfg["align_proj_bn"] = False
    assert not has_bn(build_model(cfg))


# ---------------------------------------------------------------------------
# Collapse diagnostics: both failure modes must be visible
# ---------------------------------------------------------------------------

def _normed(x):
    return torch.nn.functional.normalize(x, dim=1)


def test_complete_collapse_drives_std_ratio_to_zero():
    feats = _normed(torch.ones(64, 32) + 1e-6 * torch.randn(64, 32))
    assert collapse_stats(feats)["std_ratio"] < 0.05


def test_erank_does_not_see_complete_collapse():
    # the trap in the other direction: effective rank is scale-invariant, so a point
    # collapse plus isotropic numerical noise has a FLAT spectrum and a high erank.
    # Reading erank alone would call this healthy -- which is why std_ratio is logged
    # beside it, and why this behaviour is pinned rather than left to be rediscovered.
    feats = _normed(torch.ones(64, 32) + 1e-6 * torch.randn(64, 32))
    s = collapse_stats(feats)
    assert s["erank"] > 10.0 and s["std_ratio"] < 0.05


def test_healthy_features_score_near_one_on_both():
    feats = _normed(torch.randn(4096, 32))
    s = collapse_stats(feats)
    assert 0.8 < s["std_ratio"] < 1.2
    assert s["erank_ratio"] > 0.9


def test_dimensional_collapse_is_caught_by_erank_and_missed_by_std_ratio():
    # THE case the pairing exists for: a representation confined to a few
    # directions -- what "encode omega, drop the image" looks like -- still has a
    # healthy per-dimension spread once normalized.
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(32, 32))[0][:, :3]     # 3 of 32 directions
    feats = _normed(torch.randn(4096, 3) @ basis.t())
    s = collapse_stats(feats)
    assert s["erank"] < 4.0, "erank must expose the low-rank structure"
    assert s["std_ratio"] > 0.5, "std_ratio alone would call this healthy"


def test_erank_grows_with_the_number_of_active_directions():
    torch.manual_seed(1)
    q = torch.linalg.qr(torch.randn(64, 64))[0]
    eranks = [collapse_stats(_normed(torch.randn(4096, k) @ q[:, :k].t()))["erank"]
              for k in (2, 8, 32)]
    assert eranks[0] < eranks[1] < eranks[2]


def test_stats_are_finite_on_degenerate_input():
    for feats in (torch.zeros(4, 8), torch.ones(1, 8)):
        s = collapse_stats(feats)
        assert all(np.isfinite(v) for v in s.values())
    assert "erank" in format_stats(collapse_stats(_normed(torch.randn(16, 8))))


# ---------------------------------------------------------------------------
# The factor subset: the axis of the study
# ---------------------------------------------------------------------------

def test_factor_partition_covers_everything_exactly_once():
    assert set(GEOMETRIC_FACTORS) | set(PHOTOMETRIC_FACTORS) == set(FACTORS)
    assert not set(GEOMETRIC_FACTORS) & set(PHOTOMETRIC_FACTORS)


def test_empty_selection_means_all_factors():
    assert factor_select_mask().sum() == REGRESS_TOTAL
    assert factor_select_mask([]).sum() == REGRESS_TOTAL


def test_selection_masks_exactly_the_named_factors():
    m = factor_select_mask(["crop", "rotation"])
    assert m.sum() == REGRESS_DIMS["crop"] + REGRESS_DIMS["rotation"]
    # the crop occupies the last four scalars and must be kept whole
    assert m[-4:].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert factor_select_mask(PHOTOMETRIC_FACTORS)[-4:].tolist() == [0.0] * 4


def test_unknown_factor_is_rejected():
    try:
        factor_select_mask(["crop", "nonsense"])
    except ValueError as e:
        assert "nonsense" in str(e)
    else:
        raise AssertionError("an unknown factor must not pass silently")


def test_transform_applies_the_selection_to_the_mask():
    tf = RelPairTransform(regress=True, regress_factors=GEOMETRIC_FACTORS)
    _, _, target, mask = tf(_img())
    assert target.shape == (REGRESS_TOTAL,) and mask.shape == (REGRESS_TOTAL,)
    # photometric entries are never supervised, whatever the perceptibility margins say
    photometric = factor_select_mask(PHOTOMETRIC_FACTORS)
    assert (mask.numpy() * photometric).sum() == 0.0


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

def test_variants_differ_only_in_the_predicted_factor_set():
    geom, color, allf = _cfg("posonly_geom"), _cfg("posonly_color"), _cfg("posonly_all")
    for key in ("rel_lambda", "aug_sharing", "rel_regress", "framework",
                "crop_scale", "color_strength", "blur_mode", "p_same", "delta"):
        assert geom[key] == color[key] == allf[key], f"{key} differs across variants"
    assert set(geom["regress_factors"]) == set(GEOMETRIC_FACTORS)
    assert set(color["regress_factors"]) == set(PHOTOMETRIC_FACTORS)
    assert set(allf["regress_factors"]) == set(FACTORS)
    # p_same=0 draws every factor different, so the auxiliary term is never handed a
    # pair with a near-zero target -- it is the only thing holding the representation
    # up. aug_sharing selects the relational loader, it does not share anything here.
    assert geom["p_same"] == 0.0
    assert geom["aug_sharing"] is True
    assert geom["rel_lambda"] > 0, "posonly with lambda=0 is the collapse control"


def test_build_transform_honours_the_configured_factor_set():
    tf = build_transform(_cfg("posonly_geom"))
    assert isinstance(tf, RelPairTransform) and tf.regress is True
    assert tf._factor_sel.sum() == sum(REGRESS_DIMS[f] for f in GEOMETRIC_FACTORS)


def test_baseline_on_posonly_is_the_collapse_control():
    cfg = _cfg("baseline")
    assert cfg["rel_lambda"] == 0.0, "the control must have no auxiliary term at all"


def test_collapse_line_reaches_the_curves(tmp_path):
    # the diagnostics are only useful if they come back out of the log
    from pred_ssl.scripts.plot_curves import parse_curves
    log = tmp_path / "t.log"
    log.write_text(
        "Epoch [1/2]  Loss: 1.03  SSL_Loss: 0.68  Pred_Loss: 0.69  Pred_Acc: 0.00%  LR: 0.004\n"
        "  KNN_Acc: 41.00%  (epoch 1)\n"
        "  " + format_stats(collapse_stats(_normed(torch.randn(64, 16)))) + "  (epoch 1)\n")
    r = parse_curves(str(log))["pretrain"][1]
    assert r["knn_acc"] == "41.00"
    assert float(r["std_ratio"]) > 0 and float(r["erank"]) > 1
    assert 0.0 <= float(r["erank_ratio"]) <= 1.0


def test_posonly_is_registered_in_the_matrix_and_the_knob_registry():
    from pred_ssl.relctl.knobs import EXPERIMENTS, FRAMEWORKS
    from pred_ssl.scripts.experiments import VARIANTS
    assert "posonly" in FRAMEWORKS
    for v in ("posonly_geom", "posonly_color", "posonly_all"):
        assert v in VARIANTS and v in EXPERIMENTS


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
