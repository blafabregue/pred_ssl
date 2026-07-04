"""
Shared eval helpers: load a pred_ssl checkpoint's trainable-backbone trunk into a
plain torchvision ResNet (framework-agnostic), and small metric utilities.
"""

import torch
import torchvision.models as models


def resolve_arch(ckpt, arch_arg):
    """Prefer the arch stored in the checkpoint; fall back to the CLI arg."""
    return ckpt.get("arch", arch_arg) if isinstance(ckpt, dict) else arch_arg


def load_backbone(model, pretrained_path, verbose=True):
    """Load the ResNet trunk from a pred_ssl checkpoint into `model` (a plain resnet).

    Reads checkpoint["backbone_state_dict"] (trunk-only, `backbone.`-prefixed). Falls
    back to filtering a flat state_dict by the `backbone.` prefix (dropping `fc.`),
    which also handles the existing repos' checkpoints. Asserts the only missing keys
    are the freshly-initialized classifier {fc.weight, fc.bias}.
    """
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
        sd = {k[len("backbone."):]: v for k, v in ckpt["backbone_state_dict"].items()
              if k.startswith("backbone.")}
    else:
        raw = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
        # state_dict may be the pred_ssl nested {"model":..,"rel_head":..} or a flat dict
        if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            raw = raw["model"]
        sd = {}
        for k, v in raw.items():
            if not k.startswith("backbone."):
                continue
            nk = k[len("backbone."):]
            if nk.startswith("fc."):
                continue
            sd[nk] = v
    msg = model.load_state_dict(sd, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, \
        f"unexpected missing keys: {msg.missing_keys}"
    if verbose:
        print(f"=> loaded {len(sd)} backbone tensors from {pretrained_path}")
        print(f"   missing (expected): {sorted(msg.missing_keys)}")
    return ckpt


def build_resnet(arch, num_classes):
    model = models.__dict__[arch]()
    feat_dim = model.fc.in_features
    import torch.nn as nn
    model.fc = nn.Linear(feat_dim, num_classes)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    return model


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / bs)
                for k in topk]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
