"""
Few-shot evaluation on Flowers-102 for pred_ssl checkpoints. Adapts
SimCLR-Imagenet/main_fewshot.py. Extracts frozen features once, then for each
K-shot value runs N trials of a linear classifier. Only the checkpoint loader
changed (pred_ssl.eval.common.load_backbone).

Two protocols, selected with --protocol:

  ours (default)  the protocol every number in the paper was measured with: raw
                  features, Adam at lr 0.03 for 250 iterations, no weight decay,
                  fit on the k shots only. Prints "  {k}-shot: ...", which is the
                  line extract_results parses.

  ref             the protocol AugSelf (Lee et al., 2021) uses in their Table 3:
                  L2-normalized features, LBFGS with strong-Wolfe line search, and
                  an L2 penalty selected by sweeping a log grid. It is much more
                  strongly regularized, and on 102 classes x 2048 dims that gap is
                  worth several points at 5 shots. Prints "  {k}-shot/ref: ...",
                  deliberately NOT matching extract_results' regex, so turning it
                  on never perturbs results.csv.

  both            runs the two back to back on the same sampled shots.

The two are not interchangeable: only `ours` is comparable across our own runs, and
only `ref` is comparable to AugSelf's published numbers. Do not mix them in a table.

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


def _sample_shots(train_y, k, num_classes, seed):
    """The k-shot support set. Seeded identically for both protocols, so that
    running --protocol both compares objectives and not draws."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = []
    for c in range(num_classes):
        ci = (train_y == c).nonzero(as_tuple=False).squeeze(1)
        idx.append(ci[torch.randperm(len(ci))[:k]])
    return torch.cat(idx)


def few_shot_trial(train_f, train_y, test_f, test_y, k, feat_dim, num_classes,
                   lr, iters, seed, device):
    idx = _sample_shots(train_y, k, num_classes, seed)
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


# ---------------------------------------------------------------------------
# Reference protocol (AugSelf, Lee et al. 2021)
# ---------------------------------------------------------------------------

def _fit_lbfgs(X, y, num_classes, weight, device, max_iter=200):
    clf = nn.Linear(X.shape[1], num_classes).to(device)
    nn.init.zeros_(clf.weight)
    nn.init.zeros_(clf.bias)
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=max_iter,
                            line_search_fn="strong_wolfe")
    crit = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = crit(clf(X), y) + weight * clf.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return clf.eval()


def _accuracy(clf, X, y):
    with torch.no_grad():
        return (clf(X).argmax(dim=1) == y).float().mean().item() * 100


def few_shot_trial_ref(train_f, train_y, test_f, test_y, k, feat_dim, num_classes,
                       n_weights, seed, device, sel_f=None, sel_y=None):
    """LBFGS + an L2 penalty swept over a log grid.

    The penalty has to be chosen on held-out data. AugSelf selects it on the
    dataset's own validation split; our prepared Flowers tree has no val/, so when
    none is supplied we hold out one shot per class from the support set, pick the
    weight there, and refit on all k. That keeps the selection honest -- scoring the
    grid on test_f would leak the test set and inflate the result.
    """
    idx = _sample_shots(train_y, k, num_classes, seed)
    shot_f = torch.nn.functional.normalize(train_f[idx], dim=1).to(device)
    shot_y = train_y[idx].to(device)
    test_fn = torch.nn.functional.normalize(test_f, dim=1).to(device)
    test_yn = test_y.to(device)

    weights = torch.logspace(-6, 5, n_weights).tolist()

    if sel_f is not None:                       # a real validation split
        fit_f, fit_y = shot_f, shot_y
        val_f = torch.nn.functional.normalize(sel_f, dim=1).to(device)
        val_y = sel_y.to(device)
    elif k >= 2 and len(shot_y) == k * num_classes:
        # _sample_shots concatenates per-class chunks of k, so every k-th entry is
        # that class's first shot. (Guarded above: the stride is only meaningful if
        # no class came up short.)
        hold = torch.zeros(len(shot_y), dtype=torch.bool)
        hold[torch.arange(0, len(shot_y), k)] = True
        fit_f, fit_y = shot_f[~hold], shot_y[~hold]
        val_f, val_y = shot_f[hold], shot_y[hold]
    else:                                       # k=1, or a class with < k images
        return _accuracy(_fit_lbfgs(shot_f, shot_y, num_classes,
                                    weights[len(weights) // 2], device),
                         test_fn, test_yn)

    best_w, best_acc = weights[0], -1.0
    for w in weights:
        acc = _accuracy(_fit_lbfgs(fit_f, fit_y, num_classes, w, device), val_f, val_y)
        if acc > best_acc:
            best_w, best_acc = w, acc
    return _accuracy(_fit_lbfgs(shot_f, shot_y, num_classes, best_w, device),
                     test_fn, test_yn)


def main():
    ap = argparse.ArgumentParser(description="pred_ssl few-shot eval (Flowers-102)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--n-shots", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--protocol", default="ours", choices=["ours", "ref", "both"],
                    help="ours = Adam/250it, the paper's numbers (parsed by "
                         "extract_results); ref = AugSelf's LBFGS + L2 sweep")
    ap.add_argument("--lr", type=float, default=0.03, help="protocol 'ours' only")
    ap.add_argument("--iterations", type=int, default=250, help="protocol 'ours' only")
    ap.add_argument("--n-weights", type=int, default=45,
                    help="protocol 'ref' only: points in the logspace(-6,5) L2 grid")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
    def _loader(ds):
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.workers, pin_memory=True)

    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), t)
    test_ds = datasets.ImageFolder(os.path.join(args.data, "test"), t)
    num_classes = len(train_ds.classes)
    train_loader, test_loader = _loader(train_ds), _loader(test_ds)
    # AugSelf draws its shots from trn/ and val/ pooled. Our prepared tree has no
    # val/, so this stays inactive unless one is added; when it is, the reference
    # protocol uses it to select the L2 penalty (see few_shot_trial_ref).
    val_dir = os.path.join(args.data, "val")
    val_loader = _loader(datasets.ImageFolder(val_dir, t)) if os.path.isdir(val_dir) else None

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
    sel_f = sel_y = None
    if val_loader is not None:
        sel_f, sel_y = extract_features(model, val_loader, device)
        print(f"=> found val/ ({len(sel_y)} images): used to select the L2 penalty")

    def _report(label, fn):
        for k in args.n_shots:
            accs = [fn(k, args.seed + t_) for t_ in range(args.n_trials)]
            mean, std = float(np.mean(accs)), float(np.std(accs))
            ci95 = 1.96 * std / np.sqrt(len(accs))
            print(f"  {k}-shot{label}: {mean:.1f}% (± {ci95:.1f}%)")

    print("\n" + "=" * 70)
    print("Few-Shot Classification Results — Flowers-102")
    print(f"  Checkpoint: {args.pretrained}")
    print(f"  Trials: {args.n_trials}   Protocol: {args.protocol}")
    print("=" * 70)
    if args.protocol in ("ours", "both"):
        _report("", lambda k, s: few_shot_trial(
            train_f, train_y, test_f, test_y, k, feat_dim, num_classes,
            args.lr, args.iterations, s, device))
    if args.protocol in ("ref", "both"):
        # "/ref" keeps these out of extract_results' "(\d+)-shot:" regex on purpose:
        # results.csv must stay on one protocol whatever this flag is set to.
        _report("/ref", lambda k, s: few_shot_trial_ref(
            train_f, train_y, test_f, test_y, k, feat_dim, num_classes,
            args.n_weights, s, device, sel_f, sel_y))
    print("=" * 70)


if __name__ == "__main__":
    main()
