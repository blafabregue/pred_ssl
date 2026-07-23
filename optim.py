"""
LARS optimizer (You et al., 2017) for the large-LR SSL frameworks.

SimCLR/BYOL train fine under plain SGD in this repo because their losses act on
L2-normalized features (bounded). VICReg's loss acts on the UN-normalized expander
outputs with large coefficients (sim/std = 25), and plain SGD at the same learning
rate diverges to NaN within a few steps. LARS rescales each layer's update by the
trust ratio ||w|| / ||grad||, which is exactly what stabilizes this regime and what
the official VICReg/SimCLR/BYOL code uses.

``lars_param_groups`` splits parameters so that biases and normalization parameters
(ndim < 2) skip BOTH weight decay and the trust-ratio adaptation, matching the
reference VICReg implementation.
"""

import torch


class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, eta=0.001, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        eta=eta, eps=eps, lars_exclude=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                dp = p.grad
                if not g["lars_exclude"]:
                    if g["weight_decay"] != 0:
                        dp = dp.add(p, alpha=g["weight_decay"])
                    p_norm = torch.norm(p)
                    g_norm = torch.norm(dp)
                    one = torch.ones_like(p_norm)
                    # trust ratio; falls back to 1 when a norm is 0 (fresh/zero param)
                    trust = torch.where(
                        p_norm > 0,
                        torch.where(g_norm > 0, g["eta"] * p_norm / (g_norm + g["eps"]), one),
                        one,
                    )
                    dp = dp.mul(trust)
                state = self.state[p]
                if "mu" not in state:
                    buf = state["mu"] = torch.clone(dp).detach()
                else:
                    buf = state["mu"]
                    buf.mul_(g["momentum"]).add_(dp)
                p.add_(buf, alpha=-g["lr"])
        return loss


def lars_param_groups(params, weight_decay):
    """Two groups: weights (adapted, weight-decayed) and bias/norm (excluded)."""
    params = [p for p in params if p.requires_grad]
    adapted = [p for p in params if p.ndim >= 2]
    excluded = [p for p in params if p.ndim < 2]
    return [
        {"params": adapted, "weight_decay": weight_decay, "lars_exclude": False},
        {"params": excluded, "weight_decay": 0.0, "lars_exclude": True},
    ]
