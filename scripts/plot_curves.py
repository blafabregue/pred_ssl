"""
Per-epoch training/eval curves from pred_ssl logs -> CSV (+ PNG if matplotlib).

Parses, from any pretrain log (logs/<tag>.log) and/or eval log (logs/<tag>.eval.log
or a full pipeline log):
  - pretrain epochs:  Loss / SSL_Loss / Pred_Loss / Pred_Acc  (+ KNN_Acc when the
    kNN monitor is on) -> task "pretrain"
  - linear probes:    Train/Val Loss + Val Acc@1/@5 per epoch, one task per
    "Linear Evaluation: ..." header (object, rotation, CUB...)

Appended/resumed logs are handled: for a repeated (task, epoch) the LAST occurrence
wins. Output: <stem>.curves.csv next to the log (override with --out-dir), plus
<stem>.curves.png when matplotlib is importable (skip with --csv-only).

    python -m pred_ssl.scripts.plot_curves pred_ssl/logs/simclr_relpred_resnet50_s1.log
    python -m pred_ssl.scripts.plot_curves pred_ssl/logs/*.log --out-dir pred_ssl/curves
"""

import argparse
import csv
import os
import re

_PRE = re.compile(r"Epoch \[(\d+)/\d+\]\s+Loss:\s+([\d.]+)(?:\s+SSL_Loss:\s+([\d.]+))?"
                  r"\s+Pred_Loss:\s+([\d.]+)\s+Pred_Acc:\s+([\d.]+)%")
_KNN = re.compile(r"KNN_Acc:\s+([\d.]+)%\s+\(epoch (\d+)\)")
_EVAL = re.compile(r"Epoch \[(\d+)/\d+\]\s+Train Loss:\s+([\d.]+)\s+Train Acc@1:\s+([\d.]+)%"
                   r"\s+Val Loss:\s+([\d.]+)\s+Val Acc@1:\s+([\d.]+)%\s+Val Acc@5:\s+([\d.]+)%")
_TASK = re.compile(r"Linear Evaluation:\s+(.+)")

FIELDS = ["task", "epoch", "loss", "ssl_loss", "pred_loss", "pred_acc", "knn_acc",
          "train_loss", "train_acc1", "val_loss", "val_acc1", "val_acc5"]


def parse_curves(path):
    """{task -> {epoch -> row}}; row keys are a subset of FIELDS."""
    tasks = {}

    def row(task, epoch):
        return tasks.setdefault(task, {}).setdefault(epoch, {"task": task, "epoch": epoch})

    eval_task = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            t = _TASK.search(line)
            if t:
                eval_task = t.group(1).strip()
                continue
            m = _PRE.search(line)
            if m:
                r = row("pretrain", int(m.group(1)))
                r["loss"], r["ssl_loss"] = m.group(2), m.group(3) or ""
                r["pred_loss"], r["pred_acc"] = m.group(4), m.group(5)
                continue
            m = _KNN.search(line)
            if m:
                row("pretrain", int(m.group(2)))["knn_acc"] = m.group(1)
                continue
            m = _EVAL.search(line)
            if m and eval_task:
                r = row(eval_task, int(m.group(1)))
                (r["train_loss"], r["train_acc1"], r["val_loss"],
                 r["val_acc1"], r["val_acc5"]) = m.groups()[1:]
    return tasks


def write_csv(tasks, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        for task in tasks:
            for epoch in sorted(tasks[task]):
                w.writerow(tasks[task][epoch])
    return out_csv


def plot_png(tasks, out_png, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not installed -> no PNG; CSV written)")
        return None

    def series(task, key):
        pts = [(e, float(r[key])) for e, r in sorted(tasks.get(task, {}).items())
               if r.get(key) not in (None, "")]
        return ([p[0] for p in pts], [p[1] for p in pts]) if pts else (None, None)

    eval_tasks = [t for t in tasks if t != "pretrain"]
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(title)

    for key, label in (("loss", "total loss"), ("ssl_loss", "SSL loss"),
                       ("pred_loss", "rel-head loss")):
        x, y = series("pretrain", key)
        if x:
            ax_l.plot(x, y, label=label)
    for t in eval_tasks:
        x, y = series(t, "val_loss")
        if x:
            ax_l.plot(x, y, "--", label=f"val loss [{t}]")
    ax_l.set_xlabel("epoch"), ax_l.set_ylabel("loss"), ax_l.set_title("losses")
    ax_l.legend(fontsize=8), ax_l.grid(alpha=0.3)

    for key, label in (("knn_acc", "kNN val acc (pretrain)"),
                       ("pred_acc", "rel-head train acc")):
        x, y = series("pretrain", key)
        if x:
            ax_a.plot(x, y, marker="o" if key == "knn_acc" else None,
                      markersize=3, label=label)
    for t in eval_tasks:
        x, y = series(t, "val_acc1")
        if x:
            ax_a.plot(x, y, "--", label=f"val acc@1 [{t}]")
    ax_a.set_xlabel("epoch"), ax_a.set_ylabel("accuracy (%)"), ax_a.set_title("accuracies")
    ax_a.legend(fontsize=8), ax_a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description="pred_ssl per-epoch curves -> CSV/PNG")
    ap.add_argument("logs", nargs="+", help="pretrain and/or eval log file(s)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: next to each log)")
    ap.add_argument("--csv-only", action="store_true", help="skip the PNG")
    args = ap.parse_args()

    for path in args.logs:
        tasks = parse_curves(path)
        n = sum(len(v) for v in tasks.values())
        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = args.out_dir or os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        if not n:
            print(f"{path}: no epoch lines found — skipped")
            continue
        out_csv = write_csv(tasks, os.path.join(out_dir, stem + ".curves.csv"))
        print(f"{path}: {n} epoch points ({', '.join(tasks)}) -> {out_csv}")
        if not args.csv_only:
            out_png = plot_png(tasks, os.path.join(out_dir, stem + ".curves.png"), stem)
            if out_png:
                print(f"  plot -> {out_png}")


if __name__ == "__main__":
    main()
