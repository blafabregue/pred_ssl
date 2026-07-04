"""
Tests for the relctl control panel — pure stdlib + pyyaml, no torch/GPU/data needed.

Covers the load-bearing logic: the knob catalog stays in sync with the configs
(--validate), config merge / base_lr matches train.py's convention, edit validation
guards the known foot-guns, and the plan builder emits the expected commands.
"""

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pred_ssl.relctl.__main__ import validate
from pred_ssl.relctl.actions import build_plan
from pred_ssl.relctl.config import ConfigModel, ValidationError


def test_validate_catalog_in_sync():
    assert validate(REPO_ROOT) == 0


def test_merge_and_base_lr_matches_convention():
    m = ConfigModel(REPO_ROOT)
    # simclr scales lr by batch
    assert m.value("lr") == 0.3 and m.value("lr_scale_by_batch") is True
    assert abs(m.base_lr() - 0.3) < 1e-9
    m.set("batch_size", "512")
    assert abs(m.base_lr() - 0.6) < 1e-9          # 0.3 * 512/256
    # moco does not scale, uses step schedule
    m2 = ConfigModel(REPO_ROOT)
    m2.set_framework("moco")
    assert m2.value("lr") == 0.03 and m2.value("lr_scale_by_batch") is False
    assert m2.value("lr_schedule") == "step"
    assert abs(m2.base_lr() - 0.03) < 1e-9


def test_experiment_couples_rel_lambda_and_sharing():
    m = ConfigModel(REPO_ROOT)
    m.set_experiment("baseline")
    assert m.value("rel_lambda") == 0.0 and m.value("aug_sharing") is False
    m.set_experiment("relpred")
    assert m.value("rel_lambda") == 0.5 and m.value("aug_sharing") is True
    m.set_experiment("relpred_lambda0")
    assert m.value("rel_lambda") == 0.0 and m.value("aug_sharing") is True


def test_validation_guards():
    m = ConfigModel(REPO_ROOT)
    m.set_framework("looc")
    with pytest.raises(ValidationError):
        m.set("n_aug", "2")                       # NotImplementedError guard
    with pytest.raises(ValidationError):
        m.set("full_multiview", "true")
    with pytest.raises(ValidationError):
        m.set("crop_scale", "0.9 0.2")            # lo < hi
    with pytest.raises(ValidationError):
        m.set("blur_mode", "nope")                # enum
    with pytest.raises(ValidationError):
        m.set_delta_key("hue", "0")               # must be > 0


def test_overlay_excludes_wrong_framework_and_identity_flags():
    m = ConfigModel(REPO_ROOT)
    m.set_framework("simclr")
    m.set("temperature", "0.7")                   # simclr-only knob
    m.set_framework("moco")                        # now temperature is the wrong framework
    m.set("moco_t", "0.15")
    m.set("epochs", "100")                         # identity flag -> never in overlay
    ov = m.overlay_dict()
    assert "moco_t" in ov
    assert "temperature" not in ov                 # stale wrong-fw edit dropped
    assert "epochs" not in ov                      # identity flag stays a CLI flag


def test_delta_overlay_keeps_all_keys():
    m = ConfigModel(REPO_ROOT)
    m.set_delta_key("hue", "0.08")
    ov = m.overlay_dict()
    assert set(ov["delta"]) == {"brightness", "contrast", "saturation", "hue", "blur"}
    assert ov["delta"]["hue"] == 0.08


def test_plan_pipeline_command():
    m = ConfigModel(REPO_ROOT)
    m.set_framework("simclr")
    m.set_experiment("relpred")
    m.action = "pipeline"
    p = build_plan(m)
    assert p.background is True
    cmd = p.commands[0]
    assert "run_pipeline.sh" in cmd
    assert "FRAMEWORK=simclr" in cmd and "EXPERIMENT=relpred" in cmd and "MODE=all" in cmd
    assert p.log.endswith("simclr_relpred.log")


def test_plan_single_eval_and_overlay_flag():
    m = ConfigModel(REPO_ROOT)
    m.action = "eval_rotation"
    m.eval_ckpt = "pred_ssl/checkpoints/simclr_relpred/checkpoint_0500.pth.tar"
    p = build_plan(m)
    cmd = p.commands[0]
    assert "linear_probe" in cmd and "--eval-rotation" in cmd
    assert "checkpoint_0500" in cmd
    # an overlay path is only injected when a YAML-only knob is dirty
    m.set("p_same", "0.7")
    m.action = "pretrain"
    p2 = build_plan(m, overlay_path="pred_ssl/.relctl/overlays/x.yaml")
    assert any("CONFIG_OVERLAY=" in c for c in p2.commands)


def test_plan_matrix_is_not_background():
    m = ConfigModel(REPO_ROOT)
    m.action = "matrix"
    m.include_ablation = True
    p = build_plan(m)
    assert p.background is False
    assert "launch_matrix.sh" in p.commands[0] and "INCLUDE_ABLATION=1" in p.commands[0]
