"""
Per-model pretraining progression plots with a mean +/- std band over seeds.

One figure per model (framework x variant), aggregating all its seeds: each panel
plots the per-epoch mean across seeds as a line and a shaded +/-1 standard-deviation
band --- the matplotlib equivalent of seaborn's ``errorbar=('sd', 1)`` --- so a
single plot shows both the trajectory and the seed-to-seed variance. By default it
plots kNN validation accuracy and the training loss.

    python -m pred_ssl.scripts.plot_progression --logs-dir ./pred_ssl/logs \
        --out-dir ./pred_ssl/curves
    python -m pred_ssl.scripts.plot_progression --metrics knn_acc,ssl_loss --show-seeds

Reads the same pretrain logs as extract_results (``<tag>.log``), groups them by
(framework, variant) via the tag, and aligns the per-epoch series (from
plot_curves.parse_curves) across seeds. A per-model aggregated CSV is always
written; the PNG requires matplotlib (``pip install matplotlib``).
"""

import argparse
import csv
import glob
import os
import statistics

from pred_ssl.scripts.extract_results import parse_tag
from pred_ssl.scripts.plot_curves import parse_curves

# metric key -> (axis label, "lower is better" flag used only for annotation)
METRIC_LABELS = {
    "knn_acc":   "kNN val accuracy (%)",
    "loss":      "Training loss (total)",
    "ssl_loss":  "SSL loss",
    "pred_loss": "Relational loss",
    "pred_acc":  "Rel. head accuracy (%)",
}


def _seed_series(path, metric):
    """{epoch: float} for one metric of one pretrain log (empty if none present)."""
    pre = parse_curves(path).get("pretrain", {})
    out = {}
    for epoch, rowdict in pre.items():
        v = rowdict.get(metric, "")
        if v not in ("", None):
            try:
                out[int(epoch)] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def aggregate_progression(seed_series, min_seeds=1):
    """List of per-seed {epoch: value} -> sorted (epochs, means, stds, counts).

    At each epoch, the mean and sample std are taken over the seeds that reached it
    (std = 0 for a single seed). Epochs covered by fewer than ``min_seeds`` seeds are
    dropped. This is exactly what ``errorbar=('sd', 1)`` shades, computed explicitly.
    """
    epochs = sorted({e for s in seed_series for e in s})
    xs, means, stds, counts = [], [], [], []
    for e in epochs:
        vals = [s[e] for s in seed_series if e in s]
        if len(vals) < min_seeds:
            continue
        xs.append(e)
        means.append(statistics.fmean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        counts.append(len(vals))
    return xs, means, stds, counts


def _write_csv(path, metrics, agg):
    cols = ["epoch"]
    for m in metrics:
        cols += [f"{m}_mean", f"{m}_std", f"{m}_n"]
    # union of epochs across metrics, sorted
    all_epochs = sorted({e for m in metrics for e in agg[m][0]})
    idx = {m: {e: i for i, e in enumerate(agg[m][0])} for m in metrics}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in all_epochs:
            row = [e]
            for m in metrics:
                xs, means, stds, counts = agg[m]
                if e in idx[m]:
                    i = idx[m][e]
                    row += [f"{means[i]:.4f}", f"{stds[i]:.4f}", counts[i]]
                else:
                    row += ["", "", ""]
            w.writerow(row)


def _plot(plt, path, title, metrics, agg, seed_series_by_metric, show_seeds):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        xs, means, stds, counts = agg[m]
        if show_seeds:
            for s in seed_series_by_metric[m]:
                sx = sorted(s)
                ax.plot(sx, [s[e] for e in sx], color="0.7", lw=0.7, alpha=0.6, zorder=1)
        if xs:
            lo = [me - sd for me, sd in zip(means, stds)]
            hi = [me + sd for me, sd in zip(means, stds)]
            ax.fill_between(xs, lo, hi, alpha=0.25, zorder=2, label="$\\pm$1 sd")
            ax.plot(xs, means, lw=2.0, zorder=3, label="mean")
        ax.set_xlabel("epoch")
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    nseeds = max((c for m in metrics for c in agg[m][3]), default=0)
    fig.suptitle(f"{title}  (over {nseeds} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="per-model pretraining progression plots")
    ap.add_argument("--logs-dir", default="./pred_ssl/logs")
    ap.add_argument("--out-dir", default="./pred_ssl/curves")
    ap.add_argument("--glob", default="*.log")
    ap.add_argument("--metrics", default="knn_acc,loss",
                    help="comma-separated: knn_acc, loss, ssl_loss, pred_loss, pred_acc")
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="drop epochs covered by fewer than this many seeds")
    ap.add_argument("--show-seeds", action="store_true",
                    help="overlay faint individual-seed lines under the band")
    ap.add_argument("--csv-only", action="store_true", help="skip PNGs")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    # group logs by (framework, variant), excluding eval logs
    all_logs = sorted(glob.glob(os.path.join(args.logs_dir, args.glob)))
    groups = {}
    for path in all_logs:
        if path.endswith(".eval.log"):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        fw, variant, _arch, _seed = parse_tag(stem)
        groups.setdefault((fw, variant), []).append(path)

    if not groups:
        print(f"no pretrain logs matched {args.glob} in {args.logs_dir}")
        return

    plt = None
    if not args.csv_only:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F811
        except ImportError:
            print("  (matplotlib not installed -> CSV only; pip install matplotlib)")

    for (fw, variant), paths in sorted(groups.items()):
        series = {m: [_seed_series(p, m) for p in paths] for m in metrics}
        series = {m: [s for s in lst if s] for m, lst in series.items()}  # drop empty
        agg = {m: aggregate_progression(series[m], args.min_seeds) for m in metrics}
        if not any(agg[m][0] for m in metrics):
            print(f"{fw}/{variant}: no per-epoch data for {metrics} — skipped")
            continue
        tag = f"{fw}_{variant}"
        _write_csv(os.path.join(args.out_dir, f"{tag}.progression.csv"), metrics, agg)
        nmax = max((c for m in metrics for c in agg[m][3]), default=0)
        print(f"{fw}/{variant}: {len(paths)} log(s), over {nmax} seed(s) -> "
              f"{tag}.progression.csv")
        if plt is not None:
            out_png = os.path.join(args.out_dir, f"{tag}.progression.png")
            _plot(plt, out_png, f"{fw} / {variant}", metrics, agg, series, args.show_seeds)
            print(f"    plot -> {out_png}")


if __name__ == "__main__":
    main()
