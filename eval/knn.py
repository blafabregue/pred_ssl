"""
kNN validation monitor for SSL pretraining.

Gives the "val accuracy per epoch" curve that SSL pretraining otherwise lacks
(no labels are used for TRAINING; the labels here only score frozen features).
Every ``knn_eval_freq`` epochs, train.py extracts pooled features (trainable
backbone, eval mode, no grad) for a fixed train "bank" (``knn_per_class`` images
per class) and for the full val split, then classifies each val image by the
temperature-weighted vote of its ``knn_k`` nearest bank neighbours (cosine
similarity) — the standard InstDisc/MoCo monitoring protocol.

The result is printed as ``KNN_Acc: 61.23%`` in the pretrain log (parsed by
scripts/extract_results.py and plotted by scripts/plot_curves.py). Cost: one
forward pass over bank+val on the eval transform; with the defaults on IN-100
that is ~15k images every 5 epochs.
"""

import os

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from ..models.frameworks import encode_features

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def knn_predict(query, bank, bank_labels, num_classes, k=20, temp=0.07):
    """Weighted kNN top-1 predictions. query/bank are L2-normalized (Q,D)/(B,D)."""
    k = min(k, bank.size(0))
    sim = query @ bank.t()                                   # (Q, B) cosine
    topsim, topidx = sim.topk(k, dim=1)
    weights = (topsim / temp).exp()                          # (Q, k)
    onehot = F.one_hot(bank_labels[topidx], num_classes).float()  # (Q, k, C)
    scores = (onehot * weights.unsqueeze(2)).sum(dim=1)      # (Q, C)
    return scores.argmax(dim=1)


def _bank_subset(dataset, per_class):
    """First ``per_class`` sample indices of each class (deterministic)."""
    counts = {}
    keep = []
    for i, (_, y) in enumerate(dataset.samples):
        if counts.get(y, 0) < per_class:
            counts[y] = counts.get(y, 0) + 1
            keep.append(i)
    return torch.utils.data.Subset(dataset, keep)


class KnnMonitor:
    def __init__(self, bank_loader, val_loader, num_classes, k, temp):
        self.bank_loader = bank_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.k = k
        self.temp = temp

    @torch.no_grad()
    def _extract(self, model, framework, loader, device):
        feats, labels = [], []
        for x, y in loader:
            f = encode_features(model, framework, x.to(device, non_blocking=True))
            feats.append(F.normalize(f, dim=1))
            labels.append(y.to(device))
        return torch.cat(feats), torch.cat(labels)

    @torch.no_grad()
    def evaluate(self, model, framework, device):
        """Top-1 kNN accuracy (%) of the current frozen features on the val split."""
        was_training = model.training
        model.eval()
        try:
            bank, bank_y = self._extract(model, framework, self.bank_loader, device)
            correct = total = 0
            for x, y in self.val_loader:
                f = encode_features(model, framework, x.to(device, non_blocking=True))
                pred = knn_predict(F.normalize(f, dim=1), bank, bank_y,
                                   self.num_classes, self.k, self.temp)
                correct += (pred == y.to(device)).sum().item()
                total += y.size(0)
        finally:
            if was_training:
                model.train()
        return 100.0 * correct / max(total, 1)


def build_knn_monitor(cfg):
    """KnnMonitor from the train config, or None (with a printed reason) if off/impossible."""
    freq = cfg.get("knn_eval_freq", 0)
    if not freq or freq <= 0:
        return None
    traindir = os.path.join(cfg["data"], "train")
    valdir = os.path.join(cfg["data"], "val")
    if not os.path.isdir(valdir):
        print(f"=> kNN monitor OFF: no val split at {valdir}")
        return None

    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD),
    ])
    bank_ds = _bank_subset(datasets.ImageFolder(traindir, tf),
                           cfg.get("knn_per_class", 100))
    val_ds = datasets.ImageFolder(valdir, tf)
    num_classes = len(val_ds.classes)
    kw = dict(batch_size=cfg.get("batch_size", 256), shuffle=False,
              num_workers=cfg.get("workers", 8), pin_memory=True)
    return KnnMonitor(torch.utils.data.DataLoader(bank_ds, **kw),
                      torch.utils.data.DataLoader(val_ds, **kw),
                      num_classes, cfg.get("knn_k", 20), cfg.get("knn_temp", 0.07))
