"""
Aggregate the per-run results.csv (one row per seed, from extract_results.py) into
per-(framework, variant) statistics: mean +/- std over seeds for every metric.

    # on the server, after the matrix finishes:
    python -m pred_ssl.scripts.extract_results --logs-dir ./pred_ssl/logs \
        --out ./pred_ssl/results.csv
    python -m pred_ssl.scripts.aggregate_results --in ./pred_ssl/results.csv \
        --out ./pred_ssl/results_agg.csv
    #   -> scp results.csv (and/or results_agg.csv) to your laptop

Pure standard library (no pandas/numpy), so it runs in the cluster venv unchanged.
"""

import argparse
import csv
import os
import statistics

# Fixed display order (matches the experiment matrix / paper).
FRAMEWORK_ORDER = ["simclr", "moco", "byol", "looc", "vicreg"]
VARIANT_ORDER = ["baseline", "relpred", "relpred_proj3", "relpred_split",
                 "relpred_split_80_10_10", "relpred_split_45_45_10",
                 "relpred_lambda0", "relpred_decoupled"]

# Metrics we aggregate + their nice labels.
METRICS = [
    ("in100_acc1", "IN-100 top-1"),
    ("rotation_acc1", "Rotation"),
    ("cub200_acc1", "CUB-200 top-1"),
    ("flowers_10shot", "Flowers 10-shot"),
    ("flowers_5shot", "Flowers 5-shot"),
    ("in100_acc5", "IN-100 top-5"),
    ("cub200_acc5", "CUB-200 top-5"),
    ("knn_acc", "kNN (pretrain)"),
]


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_key(row):
    return (row.get("framework", ""), row.get("variant", ""))


def aggregate(rows):
    """{(framework, variant): {metric: (mean, std, n)}}, plus seed count."""
    groups = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)

    out = {}
    for key, grp in groups.items():
        stats = {"_n_runs": len(grp)}
        for col, _ in METRICS:
            vals = [v for v in (_to_float(r.get(col)) for r in grp) if v is not None]
            if not vals:
                stats[col] = None
                continue
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            stats[col] = (mean, std, len(vals))
        out[key] = stats
    return out


def _order_index(order, value):
    return order.index(value) if value in order else len(order)


def sorted_keys(agg):
    return sorted(agg, key=lambda k: (_order_index(FRAMEWORK_ORDER, k[0]),
                                      _order_index(VARIANT_ORDER, k[1])))


def write_agg_csv(agg, path):
    cols = ["framework", "variant", "n_runs"]
    for col, _ in METRICS:
        cols += [f"{col}_mean", f"{col}_std"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key in sorted_keys(agg):
            fw, variant = key
            stats = agg[key]
            row = [fw, variant, stats["_n_runs"]]
            for col, _ in METRICS:
                cell = stats.get(col)
                if cell is None:
                    row += ["", ""]
                else:
                    mean, std, _n = cell
                    row += [f"{mean:.2f}", f"{std:.2f}"]
            w.writerow(row)
    return path


def main():
    ap = argparse.ArgumentParser(description="aggregate pred_ssl results over seeds")
    ap.add_argument("--in", dest="inp", default="./pred_ssl/results.csv")
    ap.add_argument("--out", default="./pred_ssl/results_agg.csv")
    args = ap.parse_args()

    rows = load(args.inp)
    agg = aggregate(rows)

    write_agg_csv(agg, args.out)
    # Console summary.
    print(f"{'framework':<8} {'variant':<24} {'n':>2}  "
          + "  ".join(f"{lab:>14}" for _, lab in METRICS[:4]))
    for key in sorted_keys(agg):
        fw, variant = key
        stats = agg[key]
        cells = []
        for col, _ in METRICS[:4]:
            v = stats.get(col)
            cells.append(f"{v[0]:6.2f}+-{v[1]:4.2f}" if v else f"{'--':>11}")
        print(f"{fw:<8} {variant:<24} {stats['_n_runs']:>2}  " + "  ".join(cells))
    print(f"\n=> wrote per-(framework,variant) stats to {args.out}")


if __name__ == "__main__":
    main()
