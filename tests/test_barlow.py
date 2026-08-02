"""
Tests for the Barlow Twins framework (Zbontar et al., 2021).

Barlow Twins is included as a falsification test of the paper's hypothesis that
redundancy-reduction objectives are structurally opposed to the relational signal.
For that test to be informative it must differ from VICReg ONLY in the loss, so
these tests check the loss against the reference formulation AND check that the
surrounding configuration matches VICReg's.

Run:  python -m pytest pred_ssl/tests/test_barlow.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.losses import BarlowTwinsLoss, _off_diagonal  # noqa: E402
from pred_ssl.models.frameworks import (TRAINABLE_BACKBONE_ATTR,  # noqa: E402
                                        backbone_state_dict, build_model)
from pred_ssl.models.rel_head import RelHead  # noqa: E402
from pred_ssl.losses import RelPairLoss  # noqa: E402
from pred_ssl.relctl.config import _deep_merge, _load_yaml  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _cfg(arch="resnet18", **over):
    cfg = {"arch": arch, "framework": "barlow", "rel_lambda": 0.5,
           "rel_head_hidden": 64, "feat_split": False,
           "barlow_expander_layers": 3, "barlow_expander_dim": 128,
           "barlow_proj_dim": 64, "barlow_lambd": 0.0051}
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# Loss: agreement with the reference formulation
# ---------------------------------------------------------------------------

def _reference_loss(z1, z2, bn, lambd):
    """Verbatim structure of facebookresearch/barlowtwins."""
    c = (bn(z1).T @ bn(z2)) / z1.size(0)
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    off_diag = _off_diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag


def test_loss_matches_reference_formulation():
    torch.manual_seed(0)
    for N, D in [(16, 8), (64, 32), (128, 64)]:
        crit = BarlowTwinsLoss(D, lambd=0.0051).double()
        z1 = torch.randn(N, D, dtype=torch.float64)
        z2 = z1 + 0.3 * torch.randn(N, D, dtype=torch.float64)
        ours = crit(z1, z2)
        ref = _reference_loss(z1, z2, crit.bn, 0.0051)
        assert torch.allclose(ours, ref, rtol=1e-10), (N, D, ours.item(), ref.item())


def test_identical_embeddings_minimize_the_loss():
    # perfectly correlated, decorrelated dimensions -> diagonal 1, off-diagonal ~0
    torch.manual_seed(0)
    D = 32
    crit = BarlowTwinsLoss(D, lambd=0.0051).double()
    z = torch.randn(512, D, dtype=torch.float64)
    identical = crit(z, z.clone()).item()
    unrelated = crit(z, torch.randn(512, D, dtype=torch.float64)).item()
    assert identical < unrelated
    assert identical < 1.0            # near the optimum


def test_loss_is_differentiable():
    crit = BarlowTwinsLoss(16)
    z1 = torch.randn(32, 16, requires_grad=True)
    z2 = torch.randn(32, 16, requires_grad=True)
    crit(z1, z2).backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all()


# ---------------------------------------------------------------------------
# Framework integration
# ---------------------------------------------------------------------------

def test_forward_exposes_both_view_features():
    model = build_model(_cfg()).train()
    out = model(torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64))
    assert torch.isfinite(out.ssl_loss)
    assert out.h1.shape == (4, model.feat_dim) and out.h2 is not None
    assert out.h1.requires_grad and out.h2.requires_grad


def test_relational_head_trains_the_backbone():
    model = build_model(_cfg()).train()
    head = RelHead(model.feat_dim, num_factors=9, hidden=32)
    crit = RelPairLoss()
    out = model(torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64))
    labels = (torch.rand(4, 9) > 0.5).float()
    rel, _, _ = crit(head(out.h1, out.h2), labels, torch.ones(4, 9))
    (out.ssl_loss + 0.5 * rel).backward()
    trunk = getattr(model, TRAINABLE_BACKBONE_ATTR["barlow"])
    assert any(p.grad is not None for p in trunk.parameters())
    assert any(p.grad is not None for p in head.parameters())


def test_checkpoint_loads_into_plain_resnet():
    import torchvision.models as models
    for arch in ("resnet18", "resnet50"):
        model = build_model(_cfg(arch=arch))
        sd = backbone_state_dict(model, "barlow")
        stripped = {k[len("backbone."):]: v for k, v in sd.items()}
        msg = models.__dict__[arch]().load_state_dict(stripped, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        assert msg.unexpected_keys == []


def test_projector_depth_and_output_dim():
    model = build_model(_cfg())
    linears = [m for m in model.projector if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    assert linears[-1].out_features == 64


# ---------------------------------------------------------------------------
# The comparison must isolate the loss: everything else matches VICReg
# ---------------------------------------------------------------------------

def _resolve(framework):
    cfg = _load_yaml(os.path.join(CFG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "framework", framework + ".yaml")))
    _deep_merge(cfg, _load_yaml(os.path.join(CFG_DIR, "experiment", "relpred.yaml")))
    return cfg


def test_barlow_mirrors_vicreg_except_the_loss():
    b, v = _resolve("barlow"), _resolve("vicreg")
    for key in ("optimizer", "lr", "lr_schedule", "lr_scale_by_batch",
                "warmup_epochs", "weight_decay", "batch_size", "epochs"):
        assert b[key] == v[key], f"{key}: barlow={b[key]} vicreg={v[key]}"
    # identical projector geometry, so only the objective differs
    assert b["barlow_expander_layers"] == v["vicreg_expander_layers"]
    assert b["barlow_expander_dim"] == v["vicreg_expander_dim"]
    assert b["barlow_proj_dim"] == v["vicreg_proj_dim"]


def test_barlow_output_dim_keeps_the_matrix_estimable():
    # same D/N reasoning as VICReg: the loss estimates a DxD cross-correlation
    cfg = _resolve("barlow")
    ratio = cfg["barlow_proj_dim"] / cfg["batch_size"]
    assert ratio <= 8, f"D/N = {ratio}; the off-diagonal term becomes sampling noise"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
