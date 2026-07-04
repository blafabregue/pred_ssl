"""
Parse pred_ssl pipeline logs into results.csv (the existing "Best Val" convention).

Each log is one (framework, experiment) run with STEP markers, named
<framework>_<experiment>.log (e.g. simclr_relpred.log, moco_baseline.log). Adapts
ESSL-Figure1-Imagenet/extract_results.py: section detection by STEP marker, then
regex for the pretrain final epoch (+ per-factor accuracies), the *BEST* Val Acc@1/@5
lines per eval section, and the few-shot lines.

    python pred_ssl/scripts/extract_results.py --logs-dir ./pred_ssl/logs --out ./pred_ssl/results.csv
"""

import argparse
import csv
import glob
import os
import re

FACTORS = ["rotation", "hflip", "brightness", "contrast",
           "saturation", "hue", "grayscale", "blur", "crop"]

_EPOCH = re.compile(r"Epoch \[\d+/\d+\]\s+Loss:\s+([\d.]+)(?:\s+SSL_Loss:\s+([\d.]+))?"
                    r"\s+Pred_Loss:\s+([\d.]+)\s+Pred_Acc:\s+([\d.]+)%")
_PERFACTOR = re.compile(r"PerFactor:\s+(.*)")
_BEST_ACC1 = re.compile(r"Val Acc@1:\s+([\d.]+)%")
_BEST_ACC5 = re.compile(r"Val Acc@5:\s+([\d.]+)%")
_SHOT = re.compile(r"(\d+)-shot:\s+([\d.]+)%\s+\(.*?([\d.]+)%\)")
_STEP = re.compile(r"\bSTEP\s+([1-5])\b")
_STEP_SECTION = {"1": "pretrain", "2": "in100", "3": "rotation", "4": "cub200", "5": "flowers"}


def _section(line):
    """Section transitions are driven only by the STEP markers run_pipeline.sh emits.

    This is unambiguous; content keywords are NOT used because several eval steps
    print the same phrases (e.g. both IN-100 and CUB-200 print "Object Classification").
    """
    m = _STEP.search(line)
    return _STEP_SECTION[m.group(1)] if m else None


def parse_log(path):
    r = {c: "" for c in (
        "pretrain_loss", "pretrain_ssl_loss", "pretrain_pred_loss", "pretrain_pred_acc",
        "in100_acc1", "in100_acc5", "rotation_acc1", "cub200_acc1", "cub200_acc5",
        "flowers_5shot", "flowers_5shot_ci", "flowers_10shot", "flowers_10shot_ci")}
    for f in FACTORS:
        r[f"pf_{f}"] = ""
    sec = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            s = _section(line)
            if s:
                sec = s
            if sec == "pretrain":
                m = _EPOCH.search(line)
                if m:
                    r["pretrain_loss"] = m.group(1)
                    r["pretrain_ssl_loss"] = m.group(2) or ""
                    r["pretrain_pred_loss"] = m.group(3)
                    r["pretrain_pred_acc"] = m.group(4)
                pm = _PERFACTOR.search(line)
                if pm:
                    for tok in pm.group(1).split():
                        if "=" in tok:
                            k, v = tok.split("=", 1)
                            if k in FACTORS:
                                r[f"pf_{k}"] = v
            if "*BEST*" in line:
                a1, a5 = _BEST_ACC1.search(line), _BEST_ACC5.search(line)
                if a1 and sec == "in100":
                    r["in100_acc1"] = a1.group(1)
                    if a5:
                        r["in100_acc5"] = a5.group(1)
                elif a1 and sec == "rotation":
                    r["rotation_acc1"] = a1.group(1)
                elif a1 and sec == "cub200":
                    r["cub200_acc1"] = a1.group(1)
                    if a5:
                        r["cub200_acc5"] = a5.group(1)
            if sec == "flowers":
                sm = _SHOT.search(line)
                if sm:
                    k, mean, ci = sm.group(1), sm.group(2), sm.group(3)
                    if k == "5":
                        r["flowers_5shot"], r["flowers_5shot_ci"] = mean, ci
                    elif k == "10":
                        r["flowers_10shot"], r["flowers_10shot_ci"] = mean, ci
    return r


def split_name(stem, known=("simclr", "moco", "byol", "looc", "vicreg")):
    for fw in known:
        if stem == fw or stem.startswith(fw + "_"):
            return fw, stem[len(fw) + 1:] or ""
    parts = stem.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def main():
    ap = argparse.ArgumentParser(description="pred_ssl results extractor")
    ap.add_argument("--logs-dir", default="./pred_ssl/logs")
    ap.add_argument("--out", default="./pred_ssl/results.csv")
    ap.add_argument("--glob", default="*.log")
    args = ap.parse_args()

    fieldnames = (["framework", "experiment", "pretrain_loss", "pretrain_ssl_loss",
                   "pretrain_pred_loss", "pretrain_pred_acc"]
                  + [f"pf_{f}" for f in FACTORS]
                  + ["in100_acc1", "in100_acc5", "rotation_acc1", "cub200_acc1",
                     "cub200_acc5", "flowers_5shot", "flowers_5shot_ci",
                     "flowers_10shot", "flowers_10shot_ci"])

    rows = []
    for path in sorted(glob.glob(os.path.join(args.logs_dir, args.glob))):
        stem = os.path.splitext(os.path.basename(path))[0]
        fw, exp = split_name(stem)
        row = {"framework": fw, "experiment": exp}
        row.update(parse_log(path))
        rows.append(row)
        print(f"parsed {os.path.basename(path)}: in100={row['in100_acc1']} "
              f"rot={row['rotation_acc1']} cub={row['cub200_acc1']} "
              f"5shot={row['flowers_5shot']} 10shot={row['flowers_10shot']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n=> wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
