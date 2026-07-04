"""
Experiment status report — what is done, running, partial, or still to do.

For every experiment in the matrix (scripts/experiments.py) it inspects, on disk:
  - the final checkpoint  checkpoint_<epochs>.pth.tar   -> pretraining DONE
  - the pretrain log      logs/<tag>.log                 -> last epoch reached (X/E)
  - the eval log          logs/<tag>.eval.log            -> evaluation DONE marker
and, if `squeue` is available, whether a job for that tag is currently queued/running.

Run it any time (e.g. after a 24h wall-clock batch expires) to see the remaining work;
then `bash pred_ssl/scripts/slurm_submit.sh` resubmits exactly the unfinished ones,
resuming pretraining from the last checkpoint.

    python -m pred_ssl.scripts.slurm_status
    FRAMEWORKS="simclr moco" SEEDS="1 2" python -m pred_ssl.scripts.slurm_status
"""

import os
import re
import shutil
import subprocess

from pred_ssl.scripts.experiments import matrix

_EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")


def _last_epoch(log_path):
    """Highest 'Epoch [X/E]' seen in the pretrain log, or (0, 0) if none/absent."""
    if not os.path.isfile(log_path):
        return 0, 0
    cur = tot = 0
    with open(log_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = _EPOCH_RE.search(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
    return cur, tot


def _eval_done(eval_log):
    if not os.path.isfile(eval_log):
        return False
    with open(eval_log, encoding="utf-8", errors="ignore") as fh:
        return "EVAL_DONE" in fh.read()


def _queued_jobs():
    """Set of SLURM job names for the current user (empty if squeue is unavailable)."""
    if not shutil.which("squeue"):
        return None
    try:
        out = subprocess.run(["squeue", "--me", "--noheader", "--format=%j"],
                             capture_output=True, text=True, timeout=30)
        return {j.strip() for j in out.stdout.splitlines() if j.strip()}
    except Exception:  # noqa: BLE001
        return None


def main():
    exps = matrix()
    queued = _queued_jobs()

    rows = []
    n_done = n_partial = n_running = n_todo = n_eval_todo = 0
    for e in exps:
        final_ckpt = os.path.join(e["save_dir"], f"checkpoint_{e['epochs']:04d}.pth.tar")
        pre_done = os.path.isfile(final_ckpt)
        cur, _ = _last_epoch(e["log"])
        running = queued is not None and (f"pre_{e['tag']}" in queued or f"eval_{e['tag']}" in queued)
        eval_done = _eval_done(os.path.join("./pred_ssl/logs", f"{e['tag']}.eval.log"))

        if pre_done:
            if eval_done:
                state = "DONE"
                n_done += 1
            else:
                state = "EVAL" if running else "EVAL-TODO"
                n_eval_todo += 1
        elif running:
            state = f"RUN ({cur}/{e['epochs']})"
            n_running += 1
        elif cur > 0:
            state = f"PARTIAL ({cur}/{e['epochs']})"
            n_partial += 1
        else:
            state = "TODO"
            n_todo += 1
        rows.append((e["tag"], state))

    width = max((len(t) for t, _ in rows), default=10)
    print(f"{'EXPERIMENT':<{width}}  STATE")
    print(f"{'-' * width}  {'-' * 18}")
    for tag, state in rows:
        print(f"{tag:<{width}}  {state}")

    print()
    print(f"total {len(exps)} | done {n_done} | eval-to-run {n_eval_todo} | "
          f"running {n_running} | partial {n_partial} | not-started {n_todo}")
    if queued is None:
        print("(squeue not available -> 'running' not detected; run on the cluster for live state)")
    remaining = n_eval_todo + n_running + n_partial + n_todo
    if remaining:
        print(f"\n{remaining} experiment(s) not finished. Resubmit the unfinished ones with:")
        print("  bash pred_ssl/scripts/slurm_submit.sh")
    else:
        print("\nAll experiments finished. Collect results with:")
        print("  python -m pred_ssl.scripts.extract_results --logs-dir ./pred_ssl/logs --out ./pred_ssl/results.csv")


if __name__ == "__main__":
    main()
