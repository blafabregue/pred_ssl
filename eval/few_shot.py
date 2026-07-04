"""
Few-shot evaluation on Flowers-102 for pred_ssl checkpoints. Adapts
SimCLR-Imagenet/main_fewshot.py. Extracts frozen features once, then for each
K-shot value runs N trials of a linear classifier (Adam). Only the checkpoint
loader changed (pred_ssl.eval.common.load_backbone). Output line format
"  {k}-shot: {mean:.1f}% (± {ci95:.1f}%)" is preserved for extract_results.

    python -m pred_ssl.eval.few_shot --data ./pred_ssl/datasets/flowers102_prepared \
        --pretrained ./pred_ssl/checkpoints/simclr_relpred/checkpoint_0500.pth.tar
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pred_ssl.eval.common import build_resnet, get_device, load_backbone, resolve_arch  # noqa: E402


def extract_features(model, loader, device):
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, y in loader:
            feats.append(model(images.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def few_shot_trial(train_f, train_y, test_f, test_y, k, feat_dim, num_classes,
                   lr, iters, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = []
    for c in range(num_classes):
        ci = (train_y == c).nonzero(as_tuple=False).squeeze(1)
        idx.append(ci[torch.randperm(len(ci))[:k]])
    idx = torch.cat(idx)
    shot_f, shot_y = train_f[idx], train_y[idx]

    clf = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    clf.train()
    for _ in range(iters):
        b = torch.randint(0, len(shot_f), (min(64, len(shot_f)),))
        loss = crit(clf(shot_f[b].to(device)), shot_y[b].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(test_f.to(device)).argmax(dim=1)
        return (pred == test_y.to(device)).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser(description="pred_ssl few-shot eval (Flowers-102)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--n-shots", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--iterations", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), t)
    test_ds = datasets.ImageFolder(os.path.join(args.data, "test"), t)
    num_classes = len(train_ds.classes)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    arch = resolve_arch(ckpt, args.arch)
    model = build_resnet(arch, num_classes)
    load_backbone(model, args.pretrained)
    model.fc = nn.Identity()
    model.to(device).eval()
    feat_dim = 2048 if arch == "resnet50" else 512

    print("=> extracting features...")
    train_f, train_y = extract_features(model, train_loader, device)
    test_f, test_y = extract_features(model, test_loader, device)

    print("\n" + "=" * 70)
    print("Few-Shot Classification Results — Flowers-102")
    print(f"  Checkpoint: {args.pretrained}")
    print(f"  Trials: {args.n_trials}")
    print("=" * 70)
    for k in args.n_shots:
        accs = [few_shot_trial(train_f, train_y, test_f, test_y, k, feat_dim,
                               num_classes, args.lr, args.iterations, args.seed + t_, device)
                for t_ in range(args.n_trials)]
        mean, std = float(np.mean(accs)), float(np.std(accs))
        ci95 = 1.96 * std / np.sqrt(len(accs))
        print(f"  {k}-shot: {mean:.1f}% (± {ci95:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
