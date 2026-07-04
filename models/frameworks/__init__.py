"""
Framework registry + checkpoint normalization helper.

Each framework module exposes a model whose forward returns a `ModelOutput`
(ssl_loss, ssl_acc, h1, h2) so the training loop is identical across frameworks.
`build_model(cfg)` dispatches on cfg["framework"].

`backbone_state_dict(model, framework)` extracts the TRAINABLE backbone's ResNet
trunk (dropping any `fc.*` projector keys) and re-keys it under the `backbone.`
prefix, so the existing eval scripts load it with missing == {fc.weight, fc.bias}
for every framework by construction.
"""

# Attribute name of the trainable backbone on each framework's model.
TRAINABLE_BACKBONE_ATTR = {
    "simclr": "backbone",
    "moco": "encoder_q",
    "byol": "online_backbone",
    "looc": "backbone_q",
    "vicreg": "backbone",
}


def build_model(cfg):
    fw = cfg["framework"]
    if fw == "simclr":
        from .simclr import SimCLRModel
        return SimCLRModel(cfg)
    if fw == "moco":
        from .moco import MoCoModel          # Phase 3
        return MoCoModel(cfg)
    if fw == "byol":
        from .byol import BYOLModel          # Phase 3
        return BYOLModel(cfg)
    if fw == "looc":
        from .looc import LooCModel          # Phase 3
        return LooCModel(cfg)
    if fw == "vicreg":
        from .vicreg import VICRegModel
        return VICRegModel(cfg)
    raise ValueError(f"unknown framework: {fw}")


def encode_features(model, framework, x):
    """Pooled (post-avgpool) features from the TRAINABLE backbone, WITH grad.

    Used by the decoupled relpred path (rel_decoupled=True): the framework forward
    already consumed the SSL pair, so the separate relational pair is embedded here.
    Every trainable backbone has fc=Identity, so calling it returns the (N, feat_dim)
    pooled feature directly — framework-agnostic via TRAINABLE_BACKBONE_ATTR.
    """
    backbone = getattr(model, TRAINABLE_BACKBONE_ATTR[framework])
    return backbone(x)


def backbone_state_dict(model, framework):
    """Trunk-only, `backbone.`-prefixed state dict for the eval scripts."""
    attr = TRAINABLE_BACKBONE_ATTR[framework]
    trunk = getattr(model, attr)
    out = {}
    for k, v in trunk.state_dict().items():
        if k.startswith("fc."):   # drop the projector that MoCo/LooC stash under fc
            continue
        out["backbone." + k] = v
    return out
