"""
Tests for the failed-run cleaner (scripts/clean_failed_runs.py).

The dangerous cases are the ones that could destroy healthy work, and they are what
this file is built around:
  - a log holding NaN from an old attempt followed by a healthy re-run (append-mode
    logs) must NOT be flagged;
  - a crashed run that HAS a checkpoint must NOT be flagged -- it is resumable;
  - a currently queued/running job must NOT be flagged -- it legitimately has no
    checkpoint during its first epochs.

Run:  python -m pytest pred_ssl/tests/test_clean_failed_runs.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.scripts.clean_failed_runs import (  # noqa: E402
    crashed_after_last_progress, find_failed_runs, has_checkpoint, is_non_finite,
    last_loss, targets_for)

HEALTHY = ("Epoch [1/500]  Loss: 6.2366  SSL_Loss: 5.9  Pred_Loss: 0.7  "
           "Pred_Acc: 50.00%  LR: 0.3\n")
NAN_EPOCH = ("Epoch [1/500]  Loss: nan  SSL_Loss: nan  Pred_Loss: nan  "
             "Pred_Acc: 50.06%  LR: 0.300000\n")
NAN_ITER = "  Epoch [1][20/494]  Loss nan  SSL nan  Pred nan  (399.2s)\n"
GUARD = "\n!! non-finite loss (nan) at epoch 1, iter 20/494 -- training stopped.\n"
OOM = ("Traceback (most recent call last):\n  File \"train.py\", line 1\n"
       "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 196.00 MiB.\n")
ECC = ("Traceback (most recent call last):\n"
       "torch.AcceleratorError: CUDA error: uncorrectable ECC error encountered\n")


def _setup(tmp_path):
    logs, ckpts = tmp_path / "logs", tmp_path / "checkpoints"
    logs.mkdir(); ckpts.mkdir()
    return str(logs), str(ckpts)


def _log(logs, tag, text):
    (open(os.path.join(logs, f"{tag}.log"), "w")).write(text)


def _ckpt(ckpts, tag, name="checkpoint_0010.pth.tar"):
    d = os.path.join(ckpts, tag)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, name), "w").write("x")


# ---------------------------------------------------------------------------
# Divergence detection (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_diverged_run_is_flagged(tmp_path):
    logs, ckpts = _setup(tmp_path)
    _log(logs, "vicreg_relpred_resnet50_s1", HEALTHY + NAN_ITER + NAN_EPOCH)
    bad = find_failed_runs(logs, ckpts, queued=set())
    assert [(t, r) for t, r, _ in bad] == [("vicreg_relpred_resnet50_s1", "diverged")]


def test_healthy_rerun_after_nan_is_not_flagged(tmp_path):
    logs, ckpts = _setup(tmp_path)
    _log(logs, "vicreg_relpred_resnet50_s1",
         NAN_ITER + NAN_EPOCH + GUARD + "=== resubmit ===\n" + HEALTHY)
    assert find_failed_runs(logs, ckpts, queued=set()) == []


def test_is_non_finite_is_conservative():
    assert is_non_finite("nan") and is_non_finite("inf") and is_non_finite("-inf")
    assert not is_non_finite("6.24") and not is_non_finite(None)
    assert not is_non_finite("<fresh start>")


# ---------------------------------------------------------------------------
# Crash detection -- and the guards that make it safe
# ---------------------------------------------------------------------------

def test_crashed_run_without_checkpoint_is_flagged(tmp_path):
    logs, ckpts = _setup(tmp_path)
    _log(logs, "byol_augself_resnet50_s4", "pretrain byol/augself\n" + OOM)
    bad = find_failed_runs(logs, ckpts, queued=set())
    assert [(t, r) for t, r, _ in bad] == [("byol_augself_resnet50_s4", "crashed")]


def test_crashed_run_WITH_checkpoint_is_left_alone(tmp_path):
    # resumable: the next submission continues from the checkpoint, nothing to clear
    logs, ckpts = _setup(tmp_path)
    _log(logs, "byol_augself_resnet50_s4", HEALTHY + OOM)
    _ckpt(ckpts, "byol_augself_resnet50_s4")
    assert find_failed_runs(logs, ckpts, queued=set()) == []


def test_queued_or_running_job_is_never_flagged(tmp_path):
    # a job in its first epochs has no checkpoint yet and must not be mistaken
    # for a crashed one
    logs, ckpts = _setup(tmp_path)
    _log(logs, "byol_augself_resnet50_s4", "pretrain byol/augself\n" + OOM)
    queued = {"byol_augself_resnet50_s4"}
    assert find_failed_runs(logs, ckpts, queued=queued) == []


def test_crash_before_a_healthy_rerun_is_not_flagged(tmp_path):
    logs, ckpts = _setup(tmp_path)
    _log(logs, "byol_augself_resnet50_s4", OOM + "=== resubmit ===\n" + HEALTHY)
    assert find_failed_runs(logs, ckpts, queued=set()) == []


def test_ecc_and_oom_are_both_recognised(tmp_path):
    logs, ckpts = _setup(tmp_path)
    _log(logs, "simclr_relpred_resnet50_s1", ECC)
    _log(logs, "moco_relpred_resnet50_s1", OOM)
    assert len(find_failed_runs(logs, ckpts, queued=set())) == 2


def test_include_crashed_can_be_disabled(tmp_path):
    logs, ckpts = _setup(tmp_path)
    _log(logs, "byol_augself_resnet50_s4", OOM)
    _log(logs, "vicreg_relpred_resnet50_s1", NAN_EPOCH)
    only_nan = find_failed_runs(logs, ckpts, queued=set(), include_crashed=False)
    assert [t for t, _, _ in only_nan] == ["vicreg_relpred_resnet50_s1"]


def test_crashed_after_last_progress_helper(tmp_path):
    logs, _ = _setup(tmp_path)
    p = os.path.join(logs, "a.log")
    open(p, "w").write(HEALTHY + OOM)
    assert crashed_after_last_progress(p)
    open(p, "w").write(OOM + HEALTHY)
    assert not crashed_after_last_progress(p)


# ---------------------------------------------------------------------------
# Scoping and targets
# ---------------------------------------------------------------------------

def test_framework_filter(tmp_path):
    logs, ckpts = _setup(tmp_path)
    _log(logs, "vicreg_relpred_resnet50_s1", NAN_EPOCH)
    _log(logs, "simclr_relpred_resnet50_s1", NAN_EPOCH)
    got = find_failed_runs(logs, ckpts, frameworks={"vicreg"}, queued=set())
    assert [t for t, _, _ in got] == ["vicreg_relpred_resnet50_s1"]


def test_non_matrix_and_eval_logs_are_ignored(tmp_path):
    logs, ckpts = _setup(tmp_path)
    _log(logs, "pilot_simclr", NAN_EPOCH)
    open(os.path.join(logs, "vicreg_relpred_resnet50_s1.eval.log"), "w").write(NAN_EPOCH)
    assert find_failed_runs(logs, ckpts, queued=set()) == []


def test_has_checkpoint_and_targets(tmp_path):
    logs, ckpts = _setup(tmp_path)
    tag = "vicreg_relpred_resnet50_s1"
    assert not has_checkpoint(tag, ckpts)
    _ckpt(ckpts, tag, "checkpoint_last.pth.tar")
    assert has_checkpoint(tag, ckpts)
    _log(logs, tag, NAN_EPOCH)
    found = targets_for(tag, logs, ckpts)
    assert len(found) == 2 and all(os.path.exists(p) for p in found)


def test_last_loss_reads_the_final_value(tmp_path):
    logs, _ = _setup(tmp_path)
    p = os.path.join(logs, "a.log")
    open(p, "w").write(HEALTHY + NAN_ITER)
    assert last_loss(p) == "nan"
    open(p, "w").write(NAN_ITER + HEALTHY)
    assert last_loss(p) == "6.2366"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
