"""
Phase 1 CPU checks: backbone hook + grad, RelHead swap-invariance, RelPairLoss
masking, the SimCLR forward + relational-loss backward, and the trunk-only
checkpoint normalization (loads into a plain torchvision resnet with
missing == {fc.weight, fc.bias}).

Run:  python -m pytest pred_ssl/tests/test_models.py -q
"""

import os
import sys

import torch
import torchvision.models as models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.losses import RelPairLoss  # noqa: E402
from pred_ssl.models.backbones import build_backbone  # noqa: E402
from pred_ssl.models.frameworks import backbone_state_dict, build_model  # noqa: E402
from pred_ssl.models.rel_head import RelHead  # noqa: E402


def _cfg(arch="resnet18", framework="simclr", rel_lambda=0.5):
    return {"arch": arch, "framework": framework, "simclr_dim": 128,
            "temperature": 0.5, "rel_lambda": rel_lambda, "rel_head_hidden": 64}


def test_backbone_hook_and_grad():
    enc, feat_dim = build_backbone("resnet18")
    assert feat_dim == 512
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    out = enc(x)
    assert enc._feat is not None
    assert torch.allclose(out, enc._feat)          # fc=Identity -> output == pooled feature
    assert enc._feat.requires_grad and enc._feat.grad_fn is not None
    enc._feat.sum().backward()
    grads = [p.grad is not None for p in enc.parameters() if p.requires_grad]
    assert any(grads), "backbone params should receive gradient"


def test_relhead_swap_invariance():
    head = RelHead(feat_dim=32, num_factors=8, hidden=16)
    h1 = torch.randn(4, 32)
    h2 = torch.randn(4, 32)
    a = head(h1, h2)
    b = head(h2, h1)
    assert a.shape == (4, 8)
    assert torch.allclose(a, b, atol=1e-6), "head must be invariant to view order"


def test_relpair_loss_masking():
    crit = RelPairLoss()
    torch.manual_seed(0)
    logits = torch.randn(8, 8, requires_grad=True)
    labels = (torch.rand(8, 8) > 0.5).float()
    mask = torch.ones(8, 8)
    # mask out saturation (idx 4) and hue (idx 5) for all samples -> fully-masked factors
    mask[:, 4] = 0.0
    mask[:, 5] = 0.0
    loss, acc, active = crit(logits, labels, mask)
    assert torch.isfinite(loss), "loss must be finite even with fully-masked factors"
    loss.backward()
    assert logits.grad is not None
    assert active[4] == 0 and active[5] == 0
    assert active[0] == 8
    assert acc.shape == (8,)


def test_simclr_forward_and_relloss_backward():
    model = build_model(_cfg())
    head = RelHead(model.feat_dim, num_factors=8, hidden=64)
    crit = RelPairLoss()
    v1 = torch.randn(4, 3, 64, 64)
    v2 = torch.randn(4, 3, 64, 64)
    out = model(v1, v2)
    assert out.h1.requires_grad and out.h2.requires_grad
    assert out.h1.shape == (4, model.feat_dim)
    assert torch.isfinite(out.ssl_loss)
    labels = (torch.rand(4, 8) > 0.5).float()
    mask = torch.ones(4, 8)
    rel_loss, _, _ = crit(head(out.h1, out.h2), labels, mask)
    total = out.ssl_loss + 0.5 * rel_loss
    total.backward()
    # both the backbone and the head must receive gradient
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert any(p.grad is not None for p in head.parameters())


def test_checkpoint_trunk_only_loads_into_resnet():
    for arch in ("resnet18", "resnet50"):
        model = build_model(_cfg(arch=arch))
        sd = backbone_state_dict(model, "simclr")
        # no projector / fc keys leaked
        assert all(not k.startswith("backbone.fc.") for k in sd)
        # strip the "backbone." prefix and load into a plain torchvision resnet
        stripped = {k[len("backbone."):]: v for k, v in sd.items()}
        plain = models.__dict__[arch]()
        msg = plain.load_state_dict(stripped, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, msg.missing_keys
        assert msg.unexpected_keys == [], msg.unexpected_keys


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
