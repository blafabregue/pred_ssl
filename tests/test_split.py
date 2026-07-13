"""
Tests for the latent split (relpred_split): FeatSplit sizing/slicing, the
gradient-level separation property, per-framework integration, and SplitDecovLoss.

Run:  python -m pytest pred_ssl/tests/test_split.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.losses import RelPairLoss, SplitDecovLoss  # noqa: E402
from pred_ssl.models.frameworks import TRAINABLE_BACKBONE_ATTR, build_model  # noqa: E402
from pred_ssl.models.rel_head import RelHead  # noqa: E402
from pred_ssl.models.split import FeatSplit, build_split  # noqa: E402

FRAMEWORKS = ["simclr", "moco", "byol", "looc", "vicreg"]


def _cfg(fw, arch="resnet18", **over):
    cfg = {
        "arch": arch, "framework": fw, "rel_lambda": 0.5, "rel_head_hidden": 64,
        "feat_split": True, "split_ratios": [0.5, 0.25, 0.25],
        # framework knobs (small, CPU-friendly)
        "simclr_dim": 128, "temperature": 0.5,
        "moco_dim": 128, "moco_k": 64, "moco_m": 0.999, "moco_t": 0.2,
        "dim": 128, "K": 64, "m": 0.999, "T": 0.2, "n_aug": 0, "full_multiview": False,
        "proj_hidden_dim": 128, "proj_dim": 64, "tau_base": 0.996,
        "vicreg_expander_layers": 3, "vicreg_expander_dim": 64, "vicreg_proj_dim": 64,
        "vicreg_sim_coeff": 25.0, "vicreg_std_coeff": 25.0, "vicreg_cov_coeff": 1.0,
    }
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# (1) FeatSplit sizing, bounds, slices
# ---------------------------------------------------------------------------

def test_split_sizes_default_ratios():
    s = FeatSplit(2048, ratios=(0.5, 0.25, 0.25), enabled=True)
    assert (s.n_vanilla, s.n_common, s.n_rel) == (1024, 512, 512)
    assert s.ssl_dim == 1536 and s.rel_dim == 1024
    assert s.bounds("vanilla") == (0, 1024)
    assert s.bounds("common") == (1024, 1536)
    assert s.bounds("rel") == (1536, 2048)
    assert s.bounds("ssl") == (0, 1536)
    assert s.bounds("relhead") == (1024, 2048)
    assert s.bounds("full") == (0, 2048)


def test_split_disabled_is_identity():
    s = FeatSplit(512, enabled=False)
    h = torch.randn(4, 512)
    assert s.ssl_dim == 512 and s.rel_dim == 512
    assert torch.equal(s.ssl(h), h) and torch.equal(s.rel(h), h)


def test_split_slices_partition_h():
    s = FeatSplit(512, ratios=(0.5, 0.25, 0.25), enabled=True)
    h = torch.randn(4, 512)
    assert s.ssl(h).shape == (4, s.ssl_dim)
    assert s.rel(h).shape == (4, s.rel_dim)
    # exclusive blocks + common block tile h exactly
    recon = torch.cat([s.vanilla_excl(h), h[:, s.n_vanilla:s.n_vanilla + s.n_common],
                       s.rel_excl(h)], dim=1)
    assert torch.equal(recon, h)
    # overlap of the two heads == the common block
    assert torch.equal(s.ssl(h)[:, s.n_vanilla:], s.rel(h)[:, :s.n_common])


def test_split_invalid_ratios_raise():
    with pytest.raises(ValueError):
        FeatSplit(512, ratios=(0.5, 0.25), enabled=True)          # wrong length
    with pytest.raises(ValueError):
        FeatSplit(512, ratios=(0.7, 0.25, 0.25), enabled=True)    # sum != 1
    with pytest.raises(ValueError):
        FeatSplit(512, ratios=(0.5, -0.25, 0.75), enabled=True)   # negative
    with pytest.raises(ValueError):
        FeatSplit(512, ratios=(1.0, 0.0, 0.0), enabled=True)      # rel head gets 0 dims
    # ...but a degenerate layout is fine while the split is DISABLED
    FeatSplit(512, ratios=(1.0, 0.0, 0.0), enabled=False)


def test_build_split_reads_cfg():
    s = build_split({"feat_split": True, "split_ratios": [0.25, 0.5, 0.25]}, 512)
    assert s.enabled and (s.n_vanilla, s.n_common, s.n_rel) == (128, 256, 128)
    s = build_split({}, 512)
    assert not s.enabled


def test_matrix_ratio_variants_resolve():
    # The two extra matrix experiments (80/10/10 and 45/45/10) must produce
    # valid partitions on both supported feat_dims.
    import yaml
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment")
    for name, expected_2048 in [
        ("relpred_split_80_10_10", (1638, 205, 205)),
        ("relpred_split_45_45_10", (921, 922, 205)),
    ]:
        with open(os.path.join(cfg_dir, name + ".yaml")) as f:
            cfg = yaml.safe_load(f)
        assert cfg["feat_split"] is True and abs(sum(cfg["split_ratios"]) - 1.0) < 1e-9
        s50 = build_split(cfg, 2048)   # resnet50
        assert (s50.n_vanilla, s50.n_common, s50.n_rel) == expected_2048
        s18 = build_split(cfg, 512)    # resnet18
        assert s18.n_vanilla + s18.n_common + s18.n_rel == 512
        assert s18.ssl_dim > 0 and s18.rel_dim > 0


# ---------------------------------------------------------------------------
# (2) THE property: gradient-level separation at h
# ---------------------------------------------------------------------------

def test_ssl_grad_never_touches_rel_exclusive_block():
    model = build_model(_cfg("simclr")).train()
    out = model(torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64))
    out.h1.retain_grad()
    out.ssl_loss.backward()
    s = model.split
    assert out.h1.grad[:, s.ssl_dim:].abs().sum() == 0, \
        "SSL loss must not send gradient into the rel-exclusive block"
    assert out.h1.grad[:, :s.ssl_dim].abs().sum() > 0


def test_rel_grad_never_touches_vanilla_exclusive_block():
    model = build_model(_cfg("simclr")).train()
    s = model.split
    head = RelHead(s.rel_dim, num_factors=9, hidden=32)
    crit = RelPairLoss()
    out = model(torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64))
    out.h1.retain_grad()
    labels = (torch.rand(4, 9) > 0.5).float()
    rel_loss, _, _ = crit(head(s.rel(out.h1), s.rel(out.h2)), labels, torch.ones(4, 9))
    rel_loss.backward()
    assert out.h1.grad[:, :s.n_vanilla].abs().sum() == 0, \
        "relational loss must not send gradient into the vanilla-exclusive block"
    assert out.h1.grad[:, s.n_vanilla:].abs().sum() > 0


# ---------------------------------------------------------------------------
# (3) per-framework integration: full step with split + head + decov
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_split_full_step_all_frameworks(fw):
    model = build_model(_cfg(fw)).train()
    s = model.split
    assert s.enabled and s.ssl_dim < model.feat_dim
    head = RelHead(s.rel_dim, num_factors=9, hidden=32)
    crit, decov = RelPairLoss(), SplitDecovLoss()

    out = model(torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64))
    assert torch.isfinite(out.ssl_loss)
    assert out.h1.shape == (4, model.feat_dim)          # h stays FULL width
    labels = (torch.rand(4, 9) > 0.5).float()
    rel_loss, _, _ = crit(head(s.rel(out.h1), s.rel(out.h2)), labels, torch.ones(4, 9))
    d = 0.5 * (decov(s.vanilla_excl(out.h1), s.rel_excl(out.h1))
               + decov(s.vanilla_excl(out.h2), s.rel_excl(out.h2)))
    assert torch.isfinite(d)
    (out.ssl_loss + 0.5 * rel_loss + 1.0 * d).backward()
    trunk = getattr(model, TRAINABLE_BACKBONE_ATTR[fw])
    assert any(p.grad is not None for p in trunk.parameters())
    assert any(p.grad is not None for p in head.parameters())


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_split_projector_input_width(fw):
    model = build_model(_cfg(fw))
    s = model.split
    proj = {"simclr": "projector", "moco": "projector_q", "byol": "online_projector",
            "looc": "head_q", "vicreg": "expander"}[fw]
    first_linear = [m for m in getattr(model, proj).modules()
                    if isinstance(m, torch.nn.Linear)][0]
    assert first_linear.in_features == s.ssl_dim, \
        f"{fw} SSL head must consume the {s.ssl_dim}-dim [vanilla|common] slice"


def test_split_off_keeps_native_projector_width():
    model = build_model(_cfg("simclr", feat_split=False))
    assert model.projector[0].in_features == model.feat_dim


# ---------------------------------------------------------------------------
# (4) SplitDecovLoss sanity
# ---------------------------------------------------------------------------

def test_decov_loss_scale_and_gradient():
    torch.manual_seed(0)
    crit = SplitDecovLoss()
    # Fully redundant blocks (every dim an affine copy of one signal): |corr| == 1
    # for EVERY (i, j) pair -> penalty ~= 1. Independent blocks -> ~1/N.
    base = torch.randn(256, 1)
    a = base.repeat(1, 8).detach().requires_grad_(True)
    b = base.repeat(1, 4) * 3.0 - 1.0
    high = crit(a, b)
    low = crit(torch.randn(256, 16), torch.randn(256, 16))
    assert high.item() > 0.9
    assert low.item() < 0.1
    high.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
