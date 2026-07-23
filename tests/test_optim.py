"""
Tests for the LARS optimizer, the LR warmup schedule, and the VICReg config.

VICReg's un-normalized loss diverges to NaN under plain SGD at lr 0.3 (observed on
the cluster: SSL loss 105 -> nan within 20 steps). The fix is LARS + LR warmup +
the paper's small weight decay; these tests lock the pieces in place.

Run:  python -m pytest pred_ssl/tests/test_optim.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.optim import LARS, lars_param_groups  # noqa: E402
from pred_ssl.train import adjust_learning_rate  # noqa: E402
from pred_ssl.relctl.config import _deep_merge, _load_yaml  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


# ---------------------------------------------------------------------------
# LARS
# ---------------------------------------------------------------------------

def test_lars_param_groups_split_by_ndim():
    w = torch.nn.Parameter(torch.randn(8, 4))    # weight (ndim 2) -> adapted
    b = torch.nn.Parameter(torch.randn(8))       # bias   (ndim 1) -> excluded
    groups = lars_param_groups([w, b], weight_decay=1e-4)
    adapted, excluded = groups
    assert adapted["params"] == [w] and adapted["weight_decay"] == 1e-4
    assert adapted["lars_exclude"] is False
    assert excluded["params"] == [b] and excluded["weight_decay"] == 0.0
    assert excluded["lars_exclude"] is True


def test_lars_minimizes_quadratic_and_stays_finite():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(16, 16))
    target = torch.randn(16, 16)
    opt = LARS(lars_param_groups([w], weight_decay=0.0), lr=0.1, momentum=0.9)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = ((w - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert losses[-1] < losses[0]                # it actually descends


def test_lars_trust_ratio_scales_large_gradients():
    # A huge gradient on a small-norm weight must NOT produce a huge step: LARS
    # rescales by ||w||/||g||, so the update magnitude is ~eta*||w||, not lr*||g||.
    w = torch.nn.Parameter(torch.full((10, 10), 0.01))
    opt = LARS(lars_param_groups([w], weight_decay=0.0), lr=1.0, momentum=0.0, eta=0.001)
    w.grad = torch.full((10, 10), 1e4)           # enormous gradient
    before = w.detach().clone()
    opt.step()
    step = (w.detach() - before).norm().item()
    assert step < 1.0 and torch.isfinite(w).all()   # bounded, not lr*||g|| ~ 1e5


# ---------------------------------------------------------------------------
# Warmup schedule
# ---------------------------------------------------------------------------

class _Opt:
    def __init__(self):
        self.param_groups = [{"lr": 0.0}]


def _lr_at(epoch, cfg, base_lr=1.0):
    o = _Opt()
    return adjust_learning_rate(o, epoch, cfg, base_lr)


def test_warmup_is_linear_then_cosine():
    cfg = {"lr_schedule": "cosine", "epochs": 100, "warmup_epochs": 10}
    assert abs(_lr_at(0, cfg) - 0.1) < 1e-9      # (0+1)/10
    assert abs(_lr_at(4, cfg) - 0.5) < 1e-9      # (4+1)/10
    assert abs(_lr_at(9, cfg) - 1.0) < 1e-9      # end of warmup -> full lr
    assert abs(_lr_at(10, cfg) - 1.0) < 1e-9     # cosine starts at its peak (t=0)
    # decay begins one epoch later and is still near the peak
    assert _lr_at(11, cfg) < 1.0 and _lr_at(11, cfg) > 0.9
    assert _lr_at(99, cfg) < 0.05                # near the end of cosine


def test_no_warmup_matches_plain_cosine():
    import math
    cfg = {"lr_schedule": "cosine", "epochs": 100, "warmup_epochs": 0}
    for e in (0, 25, 50, 99):
        expected = 0.5 * (1 + math.cos(math.pi * e / 100))
        assert abs(_lr_at(e, cfg) - expected) < 1e-9


# ---------------------------------------------------------------------------
# VICReg config
# ---------------------------------------------------------------------------

def _resolve(framework, experiment):
    cfg = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "framework", framework + ".yaml")))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "experiment", experiment + ".yaml")))
    return cfg


def test_vicreg_uses_lars_warmup_and_small_wd():
    cfg = _resolve("vicreg", "relpred")
    assert cfg["optimizer"] == "lars"
    assert cfg["warmup_epochs"] >= 1
    assert cfg["weight_decay"] <= 1e-5           # paper's 1e-6, not the SGD 1e-4
    # other frameworks stay on plain SGD
    assert _resolve("simclr", "relpred")["optimizer"] == "sgd"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
