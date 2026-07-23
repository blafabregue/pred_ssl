"""
Delete the checkpoints and logs of pretraining runs that ended with a non-finite loss.

A diverged run still writes ``checkpoint_<epochs>.pth.tar``, so slurm_submit.sh would
otherwise skip it as "already complete" forever, and its eval would keep reporting
chance-level numbers. This clears those runs so they are retrained from scratch.

    # inspect only (default — nothing is deleted):
    python -m pred_ssl.scripts.clean_nan_runs --frameworks vicreg
    # actually delete:
    python -m pred_ssl.scripts.clean_nan_runs --frameworks vicreg --yes

IMPORTANT — pretrain logs are APPENDED across resubmits, so a log may contain NaN from
an old attempt followed by a healthy re-run. The verdict is therefore based on the LAST
training-progress line in the log, never on "the file contains nan somewhere". A run is
flagged only when the most recent evidence is non-finite.

Deleting is destructive, so the default is a dry run and ``--yes`` is required to remove
anything; restrict the blast radius with ``--frameworks``.
"""

import argparse
import glob
import math
import os
import re
import shutil

from pred_ssl.scripts.extract_results import parse_tag

# per-iteration line:  "  Epoch [1][20/494]  Loss nan  SSL nan  Pred nan  (399.2s)"
_ITER = re.compile(r"Epoch \[\d+\]\[\d+/\d+\]\s+Loss\s+(\S+)")
# per-epoch summary:   "Epoch [1/500]  Loss: nan  SSL_Loss: nan  ..."
_EPOCH = re.compile(r"Epoch \[\d+/\d+\]\s+Loss:\s+(\S+)")
# the train.py guard:  "!! non-finite loss (nan) at epoch 1, iter 20/494 — ..."
_GUARD = re.compile(r"non-finite loss")


def last_loss(path):
    """The last training loss reported in the log, as a string, or None if there is none.

    Lines are scanned in order and the value is overwritten, so a healthy re-run appended
    after a diverged attempt correctly wins.
    """
    last = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = _EPOCH.search(line) or _ITER.search(line)
            if m:
                last = m.group(1)
            elif _GUARD.search(line):
                last = "nan"        # the guard fired here; later lines may still override
    return last


def is_non_finite(value):
    """True only when the value parses AND is nan/inf (unparseable == leave it alone)."""
    if value is None:
        return False
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def find_bad_runs(logs_dir, frameworks=None):
    """[(tag, last_loss)] for pretrain logs whose last reported loss is non-finite."""
    bad = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
        if path.endswith(".eval.log"):
            continue
        tag = os.path.splitext(os.path.basename(path))[0]
        fw, _variant, _arch, seed = parse_tag(tag)
        if seed is None:                     # not a matrix run (pilot/pipeline log)
            continue
        if frameworks and fw not in frameworks:
            continue
        value = last_loss(path)
        if is_non_finite(value):
            bad.append((tag, value))
    return bad


def targets_for(tag, logs_dir, checkpoints_dir):
    """Existing paths to remove for one run."""
    candidates = [
        os.path.join(checkpoints_dir, tag),
        os.path.join(logs_dir, f"{tag}.log"),
        os.path.join(logs_dir, f"{tag}.eval.log"),
        os.path.join(logs_dir, f"{tag}.curves.csv"),
        os.path.join(logs_dir, f"{tag}.curves.png"),
    ]
    return [p for p in candidates if os.path.exists(p)]


def main():
    ap = argparse.ArgumentParser(
        description="delete checkpoints/logs of runs that ended with a non-finite loss")
    ap.add_argument("--logs-dir", default="./pred_ssl/logs")
    ap.add_argument("--checkpoints-dir", default="./pred_ssl/checkpoints")
    ap.add_argument("--frameworks", default="",
                    help="restrict to these frameworks (space/comma separated); "
                         "empty = all. Use this to keep the blast radius small.")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete (without it, this is a dry run)")
    args = ap.parse_args()

    frameworks = {f for f in args.frameworks.replace(",", " ").split() if f}
    scope = ", ".join(sorted(frameworks)) if frameworks else "all frameworks"
    bad = find_bad_runs(args.logs_dir, frameworks or None)

    if not bad:
        print(f"nan-clean: no diverged runs found ({scope}).")
        return

    print(f"nan-clean: {len(bad)} run(s) ended with a non-finite loss ({scope}):")
    removed = 0
    for tag, value in bad:
        paths = targets_for(tag, args.logs_dir, args.checkpoints_dir)
        print(f"  {tag}  (last loss: {value})")
        for p in paths:
            if args.yes:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
                print(f"      removed {p}")
                removed += 1
            else:
                print(f"      would remove {p}")
    if args.yes:
        print(f"nan-clean: removed {removed} path(s); these runs will retrain from scratch.")
    else:
        print("nan-clean: dry run — nothing deleted. Re-run with --yes to remove.")


if __name__ == "__main__":
    main()
