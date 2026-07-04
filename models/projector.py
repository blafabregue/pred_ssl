"""
Shared projection-head builder with a native/custom switch.

By default (``proj_preset='native'``) every framework keeps its ORIGINAL hand-written
projection head, byte-for-byte — so the reference numbers are reproduced exactly. When
``proj_preset='custom'`` the head is replaced, for ALL frameworks, by a configurable MLP
whose size is driven by knobs: ``proj_layers`` (number of Linear layers), ``proj_hidden``
(hidden width), ``proj_out`` (output dim) and ``proj_bn`` (BatchNorm between layers).

Frameworks call ``build_projector(cfg, in_dim, native_fn)`` passing a zero-arg lambda that
builds their native head; momentum/queue/predictor couplings (MoCo/LooC queue dim, BYOL
predictor) read the effective output dim via ``projector_out_dim``.
"""

import torch.nn as nn


def build_mlp(in_dim, hidden_dim, out_dim, num_layers, batch_norm=True):
    """A plain MLP: ``num_layers`` Linear layers, Linear->BN->ReLU between them.

    num_layers=1 -> single Linear(in,out); num_layers=3, batch_norm=True ->
    Linear->BN->ReLU -> Linear->BN->ReLU -> Linear (the canonical VICReg/BYOL shape).
    """
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def is_custom(cfg):
    return cfg.get("proj_preset", "native") == "custom"


def projector_out_dim(cfg, native_dim):
    """Output dim of the projection head: proj_out in custom mode, else the native dim.

    Used by frameworks whose downstream shapes depend on it (MoCo/LooC queue width,
    BYOL predictor in/out).
    """
    return cfg.get("proj_out", 256) if is_custom(cfg) else native_dim


def build_projector(cfg, in_dim, native_fn):
    """Return the native head (default) or the configurable custom MLP."""
    if not is_custom(cfg):
        return native_fn()
    return build_mlp(
        in_dim,
        hidden_dim=cfg.get("proj_hidden", 2048),
        out_dim=cfg.get("proj_out", 256),
        num_layers=cfg.get("proj_layers", 2),
        batch_norm=cfg.get("proj_bn", True),
    )
