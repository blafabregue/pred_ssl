"""
Unified, config-driven pretraining entrypoint for pred_ssl.

    python pred_ssl/train.py --framework simclr --experiment relpred \
        --data ./pred_ssl/datasets/imagenet100 --arch resnet50 --save-dir ./checkpoints/simclr_relpred

Config resolution: configs/base.yaml  <-  configs/framework/<fw>.yaml  <-
configs/experiment/<exp>.yaml  <-  CLI overrides.

Logging matches the existing repo format so scripts/extract_results.py can parse it:
    Epoch [e/E]  Loss: x  Pred_Loss: y  Pred_Acc: z%  LR: lr
plus a diagnostic per-factor line when the relational head is active.
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import yaml

# Make `pred_ssl` importable whether run as `python -m pred_ssl.train` or `python pred_ssl/train.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pred_ssl.ckpt import AsyncCheckpointSaver, snapshot_to_cpu  # noqa: E402
from pred_ssl.data.loader import build_pretrain_loader  # noqa: E402
from pred_ssl.data.transforms import FACTORS  # noqa: E402
from pred_ssl.eval.knn import build_knn_monitor  # noqa: E402
from pred_ssl.losses import RelPairLoss, SplitDecovLoss  # noqa: E402
from pred_ssl.models.frameworks import backbone_state_dict, build_model, encode_features  # noqa: E402
from pred_ssl.models.rel_head import RelHead  # noqa: E402
from pred_ssl.models.split import build_split  # noqa: E402
from pred_ssl.optim import LARS, lars_param_groups  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def load_cfg(args):
    cfg = _load_yaml(os.path.join(CONFIG_DIR, "base.yaml"))
    _deep_merge(cfg, _load_yaml(os.path.join(CONFIG_DIR, "framework", f"{args.framework}.yaml")))
    if args.experiment:
        _deep_merge(cfg, _load_yaml(os.path.join(CONFIG_DIR, "experiment", f"{args.experiment}.yaml")))

    # Optional overlay for YAML-only knobs (merged after experiment, before CLI).
    # _deep_merge is recursive, so a partial `delta: {hue: 0.08}` overlay updates
    # just that sub-key and leaves the other delta entries intact.
    if getattr(args, "config_overlay", None):
        _deep_merge(cfg, _load_yaml(args.config_overlay))

    # CLI overrides (only when explicitly provided)
    overrides = {
        "data": args.data, "arch": args.arch, "epochs": args.epochs,
        "batch_size": args.batch_size, "workers": args.workers, "lr": args.lr,
        "save_dir": args.save_dir, "seed": args.seed, "print_freq": args.print_freq,
        "rel_lambda": args.rel_lambda, "color_strength": args.color_strength,
        "blur_mode": args.blur_mode, "save_freq": args.save_freq,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Utilities (copied from the existing main_*.py)
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        if n <= 0:
            return
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, cfg, base_lr):
    warmup = cfg.get("warmup_epochs", 0)
    if warmup and epoch < warmup:
        # linear warmup to base_lr over the first `warmup` epochs (epoch-granular);
        # essential for VICReg/LARS, which diverge if hit with the full LR at step 0.
        lr = base_lr * (epoch + 1) / warmup
    elif cfg["lr_schedule"] == "cosine":
        # cosine annealing over the post-warmup span (identical to before when warmup=0)
        t = (epoch - warmup) / max(1, cfg["epochs"] - warmup)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * t))
    else:  # step
        lr = base_lr
        for milestone in cfg["schedule"]:
            if epoch >= milestone:
                lr *= 0.1
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_one_epoch(loader, model, rel_head, rel_criterion, optimizer, device, cfg, epoch,
                    split=None, decov_criterion=None):
    use_rel = rel_head is not None and cfg["rel_lambda"] > 0
    decoupled = cfg.get("rel_decoupled", False)
    decov_lambda = cfg.get("split_decov_lambda", 0.0)
    use_decov = (decov_criterion is not None and split is not None and split.enabled
                 and decov_lambda > 0 and split.n_vanilla > 0 and split.n_rel > 0)
    losses = AverageMeter()
    ssl_losses = AverageMeter()
    pred_losses = AverageMeter()
    decov_losses = AverageMeter()
    factor_meters = [AverageMeter() for _ in FACTORS]

    model.train()
    if rel_head is not None:
        rel_head.train()
    end = time.time()

    for i, (data, _) in enumerate(loader):
        if decoupled:
            v1, v2, u1, u2, labels, mask = data
        else:
            v1, v2, labels, mask = data
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)

        out = model(v1, v2)
        loss = out.ssl_loss
        pred_loss_val = 0.0

        if use_rel:
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if decoupled:
                # The SSL pair (v1,v2) drove the contrastive loss above; embed the
                # SEPARATE relational pair (u1,u2) through the trainable backbone.
                u1 = u1.to(device, non_blocking=True)
                u2 = u2.to(device, non_blocking=True)
                f1 = encode_features(model, cfg["framework"], u1)
                f2 = encode_features(model, cfg["framework"], u2)
            else:
                f1, f2 = out.h1, out.h2
            # With the latent split the head only sees the [common | rel] slice.
            rel_logits = (rel_head(split.rel(f1), split.rel(f2)) if split is not None
                          else rel_head(f1, f2))
            rel_loss, acc_pct, active = rel_criterion(rel_logits, labels, mask)
            loss = loss + cfg["rel_lambda"] * rel_loss
            pred_loss_val = rel_loss.item()
            for f in range(len(FACTORS)):
                a = int(active[f].item())
                if a > 0:
                    factor_meters[f].update(acc_pct[f].item(), a)

            if use_decov:
                decov = 0.5 * (decov_criterion(split.vanilla_excl(f1), split.rel_excl(f1))
                               + decov_criterion(split.vanilla_excl(f2), split.rel_excl(f2)))
                loss = loss + decov_lambda * decov
                decov_losses.update(decov.item(), v1.size(0))

        if not torch.isfinite(loss):
            raise SystemExit(
                f"\n!! non-finite loss ({loss.item()}) at epoch {epoch + 1}, "
                f"iter {i}/{len(loader)} — training stopped before it wastes the run. "
                f"For VICReg/large-LR setups use optimizer=lars with warmup_epochs>0 "
                f"and lower lr if it persists.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = v1.size(0)
        losses.update(loss.item(), bs)
        ssl_losses.update(out.ssl_loss.item(), bs)
        pred_losses.update(pred_loss_val, bs)

        if i % cfg["print_freq"] == 0:
            dt = time.time() - end
            end = time.time()
            decov_part = f"  Decov {decov_losses.avg:.4f}" if use_decov else ""
            print(f"  Epoch [{epoch + 1}][{i}/{len(loader)}]  "
                  f"Loss {losses.avg:.4f}  SSL {ssl_losses.avg:.4f}  "
                  f"Pred {pred_losses.avg:.4f}{decov_part}  ({dt:.1f}s)", flush=True)

    mean_acc = 0.0
    active_meters = [m for m in factor_meters if m.count > 0]
    if active_meters:
        mean_acc = sum(m.avg for m in active_meters) / len(active_meters)
    return losses.avg, ssl_losses.avg, pred_losses.avg, mean_acc, factor_meters


def main():
    parser = argparse.ArgumentParser(description="pred_ssl unified pretraining")
    parser.add_argument("--framework", required=True,
                        choices=["simclr", "moco", "byol", "looc", "vicreg"])
    parser.add_argument("--experiment", default="relpred",
                        help="config in configs/experiment/ (baseline|relpred|"
                             "relpred_lambda0|relpred_decoupled|relpred_proj3|relpred_split|"
                             "relpred_split_80_10_10|relpred_split_45_45_10)")
    # overrides
    parser.add_argument("--data", default=None)
    parser.add_argument("--arch", default=None, choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--save-latest", action="store_true",
                        help="suppress intermediate milestone checkpoints (disk-efficient "
                             "SLURM mode); checkpoint_last.pth.tar (every epoch), "
                             "checkpoint_best.pth.tar and the final "
                             "checkpoint_<epochs>.pth.tar are always written regardless")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-freq", type=int, default=None)
    parser.add_argument("--rel-lambda", type=float, default=None)
    parser.add_argument("--color-strength", type=float, default=None)
    parser.add_argument("--blur-mode", default=None, choices=["sigma", "binary"])
    parser.add_argument("--resume", default="")
    # Extra YAML merged after the experiment config, before the CLI overrides
    # below. This is the hook for YAML-only knobs (momentum, lr_schedule, the
    # per-factor `delta` dict, framework-specific temperature/tau_base/moco_k, ...)
    # that have no dedicated CLI flag. Default-off: no overlay == identical behaviour.
    parser.add_argument("--config-overlay", default=None,
                        help="path to an extra YAML merged after configs/experiment/<exp>.yaml "
                             "and before CLI overrides (for YAML-only knobs)")
    parser.add_argument("--print-config", action="store_true",
                        help="print the fully-resolved merged config (incl. computed base_lr) "
                             "as YAML and exit, without training")
    args = parser.parse_args()

    cfg = load_cfg(args)

    # Scale LR per framework convention.
    base_lr = cfg["lr"] * (cfg["batch_size"] / 256 if cfg["lr_scale_by_batch"] else 1.0)

    if args.print_config:
        # Resolved config (base <- framework <- experiment <- overlay <- CLI) plus
        # the derived base_lr, as YAML — consumed by the relctl control panel to
        # show exactly what a run will use. No torch/data work happens here.
        shown = dict(cfg)
        shown["base_lr"] = round(base_lr, 6)
        print(yaml.safe_dump(shown, default_flow_style=False, sort_keys=False), end="")
        return

    print("=" * 70)
    print("pred_ssl pretraining")
    print("=" * 70)
    print(f"  framework:    {cfg['framework']}")
    print(f"  experiment:   {args.experiment}  (rel_lambda={cfg['rel_lambda']}, "
          f"aug_sharing={cfg['aug_sharing']}, rel_decoupled={cfg.get('rel_decoupled', False)})")
    print(f"  arch:         {cfg['arch']}")
    print(f"  epochs:       {cfg['epochs']}   batch_size: {cfg['batch_size']}")
    print(f"  lr:           {base_lr:.5f} (base {cfg['lr']}, scale_by_batch={cfg['lr_scale_by_batch']}, {cfg['lr_schedule']})")
    print(f"  blur_mode:    {cfg['blur_mode']}   crop_scale: {cfg['crop_scale']}")
    if cfg.get("feat_split", False):
        print(f"  feat_split:   ON  ratios={cfg.get('split_ratios')} "
              f"decov_lambda={cfg.get('split_decov_lambda', 0.0)}")
    print(f"  data:         {cfg['data']}")
    print(f"  save_dir:     {cfg['save_dir']}")
    print("=" * 70, flush=True)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> device: {device}")

    model = build_model(cfg).to(device)
    feat_dim = getattr(model, "feat_dim", 2048)
    # Same partition object the framework built for its SSL head (identity when off).
    split = getattr(model, "split", None) or build_split(cfg, feat_dim)
    if split.enabled:
        print(f"=> latent split [vanilla|common|rel] = "
              f"[{split.n_vanilla}|{split.n_common}|{split.n_rel}] of {feat_dim} "
              f"(SSL head sees {split.ssl_dim}, rel head sees {split.rel_dim})")

    rel_head = None
    rel_criterion = None
    decov_criterion = None
    if cfg["rel_lambda"] > 0:
        rel_head = RelHead(split.rel_dim, num_factors=len(FACTORS),
                           hidden=cfg["rel_head_hidden"]).to(device)
        rel_criterion = RelPairLoss().to(device)
    if cfg.get("split_decov_lambda", 0.0) > 0:
        decov_criterion = SplitDecovLoss().to(device)

    params = list(model.parameters())
    if rel_head is not None:
        params += list(rel_head.parameters())
    opt_name = cfg.get("optimizer", "sgd").lower()
    if opt_name == "lars":
        optimizer = LARS(lars_param_groups(params, cfg["weight_decay"]),
                         lr=base_lr, momentum=cfg["momentum"])
    else:
        optimizer = torch.optim.SGD(params, base_lr, momentum=cfg["momentum"],
                                    weight_decay=cfg["weight_decay"])
    print(f"=> optimizer: {opt_name}  (lr {base_lr:.5f}, wd {cfg['weight_decay']}, "
          f"warmup {cfg.get('warmup_epochs', 0)} ep)", flush=True)

    start_epoch = 0
    best_metric = None
    best_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["state_dict"]["model"])
        if rel_head is not None and ckpt["state_dict"].get("rel_head") is not None:
            rel_head.load_state_dict(ckpt["state_dict"]["rel_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_metric = ckpt.get("best_metric")
        best_epoch = ckpt.get("best_epoch", 0)

    _, loader = build_pretrain_loader(cfg)
    print(f"=> train batches/epoch: {len(loader)}", flush=True)

    # kNN val monitor: the "val accuracy per epoch" curve for SSL pretraining.
    knn = build_knn_monitor(cfg)
    knn_freq = cfg.get("knn_eval_freq", 0)
    if knn is not None:
        print(f"=> kNN monitor ON: every {knn_freq} epoch(s), "
              f"k={cfg.get('knn_k', 20)}, bank {cfg.get('knn_per_class', 100)}/class "
              f"({knn.num_classes} classes); 'best' = highest KNN_Acc", flush=True)
    else:
        print("=> kNN monitor off; 'best' = lowest train loss", flush=True)

    # Optional framework hook (BYOL uses it for the cosine tau schedule).
    if hasattr(model, "set_total_steps"):
        model.set_total_steps(cfg["epochs"] * len(loader))

    # Background checkpoint writer (async disk I/O; snapshots decouple from live tensors).
    saver = AsyncCheckpointSaver(enabled=cfg.get("async_checkpoint", True))
    try:
        _train_loop(cfg, model, rel_head, rel_criterion, optimizer, device, base_lr,
                    loader, knn, knn_freq, split, decov_criterion, saver,
                    start_epoch, best_metric, best_epoch, args)
    finally:
        saver.close()   # flush every queued write before we exit (final ckpt guaranteed)

    print("\n=> Training complete!")
    print(f"   Checkpoints in: {cfg['save_dir']}")


def _train_loop(cfg, model, rel_head, rel_criterion, optimizer, device, base_lr,
                loader, knn, knn_freq, split, decov_criterion, saver,
                start_epoch, best_metric, best_epoch, args):
    for epoch in range(start_epoch, cfg["epochs"]):
        lr = adjust_learning_rate(optimizer, epoch, cfg, base_lr)
        loss_avg, ssl_loss_avg, pred_loss_avg, pred_acc_avg, factor_meters = train_one_epoch(
            loader, model, rel_head, rel_criterion, optimizer, device, cfg, epoch,
            split=split, decov_criterion=decov_criterion)

        print(f"Epoch [{epoch + 1}/{cfg['epochs']}]  "
              f"Loss: {loss_avg:.4f}  SSL_Loss: {ssl_loss_avg:.4f}  "
              f"Pred_Loss: {pred_loss_avg:.4f}  "
              f"Pred_Acc: {pred_acc_avg:.2f}%  LR: {lr:.6f}")
        if rel_head is not None and cfg["rel_lambda"] > 0:
            parts = " ".join(f"{FACTORS[i]}={factor_meters[i].avg:.1f}"
                             for i in range(len(FACTORS)) if factor_meters[i].count > 0)
            print(f"  PerFactor: {parts}", flush=True)

        is_final = (epoch + 1) == cfg["epochs"]
        on_save_freq = is_final or (epoch + 1) % cfg["save_freq"] == 0

        # kNN val probe (the pretraining "validation accuracy" curve).
        knn_acc = None
        if knn is not None and ((epoch + 1) % knn_freq == 0 or is_final):
            knn_acc = knn.evaluate(model, cfg["framework"], device)
            print(f"  KNN_Acc: {knn_acc:.2f}%  (epoch {epoch + 1})", flush=True)

        # 'best' = highest kNN val accuracy when the monitor runs; without the monitor,
        # lowest train loss sampled on the save_freq cadence (avoids a write per epoch).
        metric = knn_acc if knn is not None else (-loss_avg if on_save_freq else None)
        improved = metric is not None and (best_metric is None or metric > best_metric)
        if improved:
            best_metric, best_epoch = metric, epoch + 1

        need_milestone = is_final or (on_save_freq and not args.save_latest)
        if on_save_freq or improved or need_milestone:
            state = {
                "epoch": epoch + 1,
                "arch": cfg["arch"],
                "framework": cfg["framework"],
                "state_dict": {
                    "model": model.state_dict(),
                    "rel_head": rel_head.state_dict() if rel_head is not None else None,
                },
                "backbone_state_dict": backbone_state_dict(model, cfg["framework"]),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "knn_acc": knn_acc,
            }
            # One CPU snapshot per epoch, reused across the (up to three) filenames.
            snap = snapshot_to_cpu(state)
            # LAST: rolling checkpoint_last.pth.tar every save_freq epochs (<= save_freq
            # epochs lost on a kill; sbatch_pretrain.slurm resumes from it).
            if on_save_freq:
                saver.save(snap, cfg["save_dir"], "checkpoint_last.pth.tar",
                           verbose=False, snapshot=False)
            # BEST: whenever the monitored metric improves.
            if improved:
                label = (f"KNN_Acc {best_metric:.2f}%" if knn is not None
                         else f"loss {loss_avg:.4f}")
                saver.save(snap, cfg["save_dir"], "checkpoint_best.pth.tar",
                           verbose=False, snapshot=False)
                print(f"  => new best ({label}) -> checkpoint_best.pth.tar", flush=True)
            # MILESTONES + the canonical final checkpoint_<epochs>.pth.tar the evals load.
            # --save-latest suppresses intermediate milestones (disk-friendly SLURM mode).
            if need_milestone:
                saver.save(snap, cfg["save_dir"], f"checkpoint_{epoch + 1:04d}.pth.tar",
                           snapshot=False)


if __name__ == "__main__":
    main()
