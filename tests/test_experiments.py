"""
CPU checks for the SLURM experiment matrix + the relpred_proj3 variant config.

Run:  python -m pytest pred_ssl/tests/test_experiments.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.scripts import experiments  # noqa: E402
from pred_ssl.relctl.config import _deep_merge, _load_yaml  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _run_matrix(**env):
    old = {k: os.environ.get(k) for k in ("FRAMEWORKS", "VARIANTS", "SEEDS", "ARCH", "EPOCHS")}
    try:
        for k in old:
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = v
        return experiments.matrix()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_default_matrix_shape():
    m = _run_matrix()
    # Tie to the actual default lists so the test tracks matrix growth (extra
    # variants/seeds) instead of a hard-coded count.
    expected = (len(experiments.DEFAULT_FRAMEWORKS)
                * len(experiments.DEFAULT_VARIANTS)
                * len(experiments.DEFAULT_SEEDS))
    assert len(m) == expected
    assert len({e["tag"] for e in m}) == len(m)     # tags are unique
    e = m[0]
    assert set(e) >= {"tag", "framework", "experiment", "arch", "seed", "epochs", "save_dir", "log"}
    # every variant maps to a real experiment config file
    for var, (exp, _desc) in experiments.VARIANTS.items():
        assert os.path.isfile(os.path.join(CFG_DIR, "experiment", exp + ".yaml")), var


def test_matrix_env_override_and_tags():
    m = _run_matrix(FRAMEWORKS="simclr moco", VARIANTS="baseline relpred_proj3", SEEDS="1 2", ARCH="resnet18")
    assert len(m) == 2 * 2 * 2
    tags = {e["tag"] for e in m}
    assert "simclr_baseline_resnet18_s1" in tags
    assert "moco_relpred_proj3_resnet18_s2" in tags
    assert all(e["arch"] == "resnet18" for e in m)


def _resolve(framework, experiment):
    cfg = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "framework", framework + ".yaml")))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "experiment", experiment + ".yaml")))
    return cfg


def test_relpred_proj3_config():
    cfg = _resolve("simclr", "relpred_proj3")
    assert cfg["rel_lambda"] == 0.5 and cfg["aug_sharing"] is True   # it IS relpred
    assert cfg["proj_preset"] == "custom" and cfg["proj_layers"] == 3  # + 3-layer head
    # inherits the base projector width/output/BN
    assert cfg["proj_hidden"] == 2048 and cfg["proj_out"] == 256 and cfg["proj_bn"] is True


def test_relpred_proj6_config_and_head_depth():
    import torch
    from pred_ssl.models.projector import build_projector
    cfg = _resolve("simclr", "relpred_proj6")
    assert cfg["rel_lambda"] == 0.5 and cfg["aug_sharing"] is True
    assert cfg["proj_preset"] == "custom" and cfg["proj_layers"] == 6
    # the built head really has 6 Linear layers and the right in/out dims
    head = build_projector(cfg, 2048, lambda: None)
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 6
    assert linears[0].in_features == 2048 and linears[-1].out_features == 256
    assert head(torch.randn(4, 2048)).shape == (4, 256)


def test_split_variants_are_opt_in_not_default():
    # kept runnable (config + VARIANTS entry) but out of the default matrix
    for v in ("relpred_split", "relpred_split_80_10_10", "relpred_split_45_45_10"):
        assert v in experiments.VARIANTS
        assert v not in experiments.DEFAULT_VARIANTS
    m = _run_matrix(VARIANTS="relpred_split", FRAMEWORKS="simclr", SEEDS="1")
    assert len(m) == 1 and m[0]["experiment"] == "relpred_split"


def test_baseline_and_relpred_configs():
    base = _resolve("vicreg", "baseline")
    assert base["rel_lambda"] == 0.0 and base["proj_preset"] == "native"
    rel = _resolve("vicreg", "relpred")
    assert rel["rel_lambda"] == 0.5 and rel["proj_preset"] == "native"
