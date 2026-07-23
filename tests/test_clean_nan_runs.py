"""
Tests for the NaN-run cleaner (scripts/clean_nan_runs.py).

The dangerous case is the append-across-resubmits one: a log holding NaN from an old
attempt followed by a healthy re-run must NOT be flagged, or the cleaner would delete a
good 500-epoch run. That is the first thing tested here.

Run:  python -m pytest pred_ssl/tests/test_clean_nan_runs.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.scripts.clean_nan_runs import (  # noqa: E402
    find_bad_runs, is_non_finite, last_loss, targets_for)

HEALTHY = ("Epoch [1/500]  Loss: 6.2366  SSL_Loss: 5.9  Pred_Loss: 0.7  "
           "Pred_Acc: 50.00%  LR: 0.3\n")
NAN_EPOCH = ("Epoch [1/500]  Loss: nan  SSL_Loss: nan  Pred_Loss: nan  "
             "Pred_Acc: 50.06%  LR: 0.300000\n")
NAN_ITER = "  Epoch [1][20/494]  Loss nan  SSL nan  Pred nan  (399.2s)\n"
GUARD = "\n!! non-finite loss (nan) at epoch 1, iter 20/494 — training stopped.\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------------------
# last_loss / is_non_finite
# ---------------------------------------------------------------------------

def test_last_loss_reads_the_final_value(tmp_path):
    p = _write(tmp_path, "a.log", HEALTHY + NAN_ITER)
    assert last_loss(p) == "nan"
    p = _write(tmp_path, "b.log", NAN_ITER + HEALTHY)
    assert last_loss(p) == "6.2366"


def test_guard_message_counts_as_nan(tmp_path):
    p = _write(tmp_path, "c.log", HEALTHY + GUARD)
    assert is_non_finite(last_loss(p))


def test_is_non_finite_is_conservative():
    assert is_non_finite("nan") and is_non_finite("inf") and is_non_finite("-inf")
    assert not is_non_finite("6.24") and not is_non_finite(None)
    assert not is_non_finite("<fresh start>")      # unparseable -> leave it alone


# ---------------------------------------------------------------------------
# THE critical case: appended logs across resubmits
# ---------------------------------------------------------------------------

def test_healthy_rerun_after_nan_is_not_flagged(tmp_path):
    # old diverged attempt, then a successful re-run appended to the same log
    _write(tmp_path, "vicreg_relpred_resnet50_s1.log",
           NAN_ITER + NAN_EPOCH + GUARD + "=== resubmit ===\n" + HEALTHY)
    assert find_bad_runs(str(tmp_path)) == []


def test_diverged_run_is_flagged(tmp_path):
    _write(tmp_path, "vicreg_relpred_resnet50_s1.log", HEALTHY + NAN_ITER + NAN_EPOCH)
    bad = find_bad_runs(str(tmp_path))
    assert [t for t, _ in bad] == ["vicreg_relpred_resnet50_s1"]


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------

def test_framework_filter_limits_blast_radius(tmp_path):
    _write(tmp_path, "vicreg_relpred_resnet50_s1.log", NAN_EPOCH)
    _write(tmp_path, "simclr_relpred_resnet50_s1.log", NAN_EPOCH)
    only_vicreg = [t for t, _ in find_bad_runs(str(tmp_path), frameworks={"vicreg"})]
    assert only_vicreg == ["vicreg_relpred_resnet50_s1"]
    assert len(find_bad_runs(str(tmp_path))) == 2      # unfiltered sees both


def test_non_matrix_and_eval_logs_are_ignored(tmp_path):
    _write(tmp_path, "pilot_simclr.log", NAN_EPOCH)                 # no _archs<seed> tag
    _write(tmp_path, "vicreg_relpred_resnet50_s1.eval.log", NAN_EPOCH)
    assert find_bad_runs(str(tmp_path)) == []


def test_targets_only_lists_existing_paths(tmp_path):
    logs = tmp_path / "logs"
    ckpts = tmp_path / "checkpoints"
    logs.mkdir()
    (ckpts / "vicreg_relpred_resnet50_s1").mkdir(parents=True)
    (logs / "vicreg_relpred_resnet50_s1.log").write_text(NAN_EPOCH)
    found = targets_for("vicreg_relpred_resnet50_s1", str(logs), str(ckpts))
    assert len(found) == 2                       # the ckpt dir + the pretrain log only
    assert all(os.path.exists(p) for p in found)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
