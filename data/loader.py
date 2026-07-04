"""
Pretraining DataLoader for pred_ssl.

Wraps ``torchvision.datasets.ImageFolder`` with either the per-factor sharing
transform (``RelPairTransform``) or the standard independent two-view transform
(``StandardTwoViewTransform``), selected by ``cfg["aug_sharing"]`` /
``cfg["rel_decoupled"]``. Each batch is ``[view1, view2, labels(N,9), mask(N,9)]`` —
or ``[view1, view2, rel1, rel2, labels(N,9), mask(N,9)]`` in the decoupled variant
(the class index from ImageFolder is ignored by the training loop, exactly as in the
existing main_*.py loops). The default collate handles either tuple width.
"""

import os

import torch
import torchvision.datasets as datasets

from .transforms import build_transform, worker_init_fn


def build_pretrain_loader(cfg):
    """Build the pretraining DataLoader from a config dict.

    Expected cfg keys: data, batch_size, workers (and the augmentation keys consumed
    by build_transform). Looks for a ``train/`` subdirectory under ``cfg["data"]``.
    """
    traindir = os.path.join(cfg["data"], "train")
    transform = build_transform(cfg)
    dataset = datasets.ImageFolder(traindir, transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("workers", 8),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.get("workers", 8) > 0,
    )
    return dataset, loader
