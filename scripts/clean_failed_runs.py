"""
Delete the checkpoints and logs of pretraining runs that cannot make progress, so
slurm_submit.sh retrains them instead of skipping or resuming them forever.

Two failure modes are handled:

  diverged  the run's last reported loss is non-finite. It still wrote
            checkpoint_<epochs>, so slurm_submit would otherwise treat it as
            "already complete" and its eval would keep reporting chance-level
            numbers. Checkpoints AND logs are removed.

  crashed   the run died before producing any usable state (CUDA OOM, ECC fault,
            config error). Removed ONLY when there is no checkpoint to resume
            from -- if a checkpoint exists the run is resumable and is left alone,
            so nothing of value is ever discarded.

    # inspect only (default -- nothing is deleted):
    python -m pred_ssl.scripts.clean_failed_runs --frameworks vicreg
    # actually delete:
    python -m pred_ssl.scripts.clean_failed_runs --frameworks vicreg --yes

SAFETY. Three independent guards, because deleting a healthy 500-epoch run is far
worse than leaving a dead one in place:

  1. Runs currently queued or running (per squeue) are never touched. A job in its
     first epochs legitimately has no checkpoint and would otherwise look "crashed".
     If squeue cannot be consulted, crash cleanup is skipped entirely unless
     --force-no-squeue is given; divergence cleanup still applies, since a
     non-finite loss is a definitive verdict.
  2. Pretrain logs are APPENDED across resubmits, so the verdict always comes from
     the LAST training-progress line -- a healthy re-run after a diverged attempt
     is correctly left alone.
  3. Deletion requires --yes; the default is a dry run.
"""

import argparse
import glob
import math
import os
import re
import shutil
import subprocess

from pred_ssl.scripts.extract_results import parse_tag

# per-iteration line:  "  Epoch [1][20/494]  Loss nan  SSL nan  ..."
_ITER = re.compile(r"Epoch \[\d+\]\[\d+/\d+\]\s+Loss\s+(\S+)")
# per-epoch summary:   "Epoch [1/500]  Loss: nan  SSL_Loss: nan  ..."
_EPOCH = re.compile(r"Epoch \[\d+/\d+\]\s+Loss:\s+(\S+)")
# the train.py guard:  "!! non-finite loss (nan) at epoch 1, iter 20/494 -- ..."
_GUARD = re.compile(r"non-finite loss")
# crash signatures
_CRASH = re.compile(r"Traceback \(most recent call last\)|OutOfMemoryError|"
                    r"CUDA error|AcceleratorError|ERROR: CUDA not available")


def last_loss(path):
    """The last training loss reported in the log, as a string, or None."""
    last = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = _EPOCH.search(line) or _ITER.search(line)
            if m:
                last = m.group(1)
            elif _GUARD.search(line):
                last = "nan"
    return last


def crashed_after_last_progress(path):
    """True when a crash signature appears AFTER the last training-progress line.

    Scanning in order and resetting on every progress line means a crash from an
    earlier attempt, followed by a healthy re-run appended to the same log, does
    not count.
    """
    crashed = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            if _EPOCH.search(line) or _ITER.search(line):
                crashed = False
            elif _CRASH.search(line):
                crashed = True
    return crashed


def is_non_finite(value):
    """True only when the value parses AND is nan/inf (unparseable == leave alone)."""
    if value is None:
        return False
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def has_checkpoint(tag, checkpoints_dir):
    """Any checkpoint the run could resume from."""
    d = os.path.join(checkpoints_dir, tag)
    return bool(glob.glob(os.path.join(d, "checkpoint_*.pth.tar")))


def queued_tags():
    """Tags with a queued/running SLURM job, or None if squeue is unavailable."""
    try:
        out = subprocess.run(["squeue", "--me", "--noheader", "--format=%j"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    tags = set()
    for name in out.stdout.split():
        for prefix in ("pre_", "eval_"):
            if name.startswith(prefix):
                tags.add(name[len(prefix):])
    return tags


def find_failed_runs(logs_dir, checkpoints_dir, frameworks=None, queued=None,
                     include_crashed=True):
    """[(tag, reason, detail)] for runs that should be cleared."""
    bad = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
        if path.endswith(".eval.log"):
            continue
        tag = os.path.splitext(os.path.basename(path))[0]
        fw, _variant, _arch, seed = parse_tag(tag)
        if seed is None:                          # not a matrix run (pilot/pipeline)
            continue
        if frameworks and fw not in frameworks:
            continue
        if queued and tag in queued:              # never touch a live job
            continue

        value = last_loss(path)
        if is_non_finite(value):
            bad.append((tag, "diverged", f"last loss {value}"))
        elif include_crashed and crashed_after_last_progress(path):
            # only when there is nothing to resume from: otherwise the job simply
            # continues from its checkpoint on the next submission.
            if not has_checkpoint(tag, checkpoints_dir):
                bad.append((tag, "crashed", "no checkpoint to resume"))
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
        description="delete checkpoints/logs of diverged or unrecoverably crashed runs")
    ap.add_argument("--logs-dir", default="./pred_ssl/logs")
    ap.add_argument("--checkpoints-dir", default="./pred_ssl/checkpoints")
    ap.add_argument("--frameworks", default="",
                    help="restrict to these frameworks (space/comma separated); "
                         "empty = all. Use this to keep the blast radius small.")
    ap.add_argument("--no-crashed", action="store_true",
                    help="only clear diverged (NaN) runs, not crashed ones")
    ap.add_argument("--force-no-squeue", action="store_true",
                    help="allow crash cleanup even when squeue cannot be consulted "
                         "(unsafe while jobs are running)")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete (without it, this is a dry run)")
    args = ap.parse_args()

    frameworks = {f for f in args.frameworks.replace(",", " ").split() if f}
    scope = ", ".join(sorted(frameworks)) if frameworks else "all frameworks"

    queued = queued_tags()
    include_crashed = not args.no_crashed
    if queued is None and include_crashed and not args.force_no_squeue:
        print("clean-failed: squeue unavailable -> skipping CRASHED runs (a running job "
              "has no checkpoint yet and would look crashed). Use --force-no-squeue to "
              "override; diverged runs are still handled.")
        include_crashed = False
    elif queued:
        print(f"clean-failed: {len(queued)} run(s) queued/running are protected.")

    bad = find_failed_runs(args.logs_dir, args.checkpoints_dir, frameworks or None,
                           queued, include_crashed)
    if not bad:
        print(f"clean-failed: nothing to clear ({scope}).")
        return

    print(f"clean-failed: {len(bad)} run(s) to clear ({scope}):")
    removed = 0
    for tag, reason, detail in bad:
        print(f"  {tag}  [{reason}: {detail}]")
        for p in targets_for(tag, args.logs_dir, args.checkpoints_dir):
            if args.yes:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
                print(f"      removed {p}")
                removed += 1
            else:
                print(f"      would remove {p}")
    if args.yes:
        print(f"clean-failed: removed {removed} path(s); these runs will retrain.")
    else:
        print("clean-failed: dry run -- nothing deleted. Re-run with --yes to remove.")


if __name__ == "__main__":
    main()
