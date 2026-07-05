"""
Tests for the async checkpoint saver (pred_ssl/ckpt.py): correctness of the written
file, the anti-race decoupling (mutating the source after save must NOT change the
file), atomicity (no leftover .tmp), synchronous fallback, and error resilience
(a failing write warns + counts but never crashes training).

Run:  python -m pytest pred_ssl/tests/test_ckpt.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pred_ssl.ckpt as ckptmod  # noqa: E402
from pred_ssl.ckpt import AsyncCheckpointSaver, snapshot_to_cpu  # noqa: E402


def _state():
    return {"epoch": 3, "state_dict": {"model": {"w": torch.arange(4.0)}},
            "cfg": {"arch": "resnet18"}, "best_metric": 1.0}


def test_snapshot_decouples_from_source():
    src = _state()
    snap = snapshot_to_cpu(src)
    src["state_dict"]["model"]["w"].add_(100.0)      # mutate AFTER snapshot
    assert torch.equal(snap["state_dict"]["model"]["w"], torch.arange(4.0))
    assert snap["cfg"]["arch"] == "resnet18"


def test_async_save_writes_loadable_file(tmp_path):
    saver = AsyncCheckpointSaver(enabled=True)
    saver.save(_state(), str(tmp_path), "checkpoint_last.pth.tar", verbose=False)
    saver.close()                                     # flush the queue
    path = tmp_path / "checkpoint_last.pth.tar"
    assert path.is_file()
    assert not (tmp_path / "checkpoint_last.pth.tar.tmp").exists()  # atomic: no leftover
    loaded = torch.load(str(path), weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["state_dict"]["model"]["w"], torch.arange(4.0))
    assert saver.errors == 0


def test_async_save_is_race_safe(tmp_path):
    # Mutating the live tensor immediately after enqueuing must not affect the file,
    # because save() snapshots before handing off to the worker thread.
    saver = AsyncCheckpointSaver(enabled=True)
    st = _state()
    saver.save(st, str(tmp_path), "c.pth.tar", verbose=False)
    st["state_dict"]["model"]["w"].add_(999.0)        # race the background writer
    saver.close()
    loaded = torch.load(str(tmp_path / "c.pth.tar"), weights_only=False)
    assert torch.equal(loaded["state_dict"]["model"]["w"], torch.arange(4.0))


def test_shared_snapshot_across_filenames(tmp_path):
    saver = AsyncCheckpointSaver(enabled=True)
    snap = snapshot_to_cpu(_state())
    for name in ("checkpoint_last.pth.tar", "checkpoint_best.pth.tar", "checkpoint_0003.pth.tar"):
        saver.save(snap, str(tmp_path), name, verbose=False, snapshot=False)
    saver.close()
    for name in ("checkpoint_last.pth.tar", "checkpoint_best.pth.tar", "checkpoint_0003.pth.tar"):
        assert (tmp_path / name).is_file()
    assert saver.errors == 0


def test_sync_fallback_writes_immediately(tmp_path):
    saver = AsyncCheckpointSaver(enabled=False)
    saver.save(_state(), str(tmp_path), "c.pth.tar", verbose=False)
    assert (tmp_path / "c.pth.tar").is_file()          # written before close()
    saver.close()


def test_worker_error_does_not_crash(tmp_path, monkeypatch, capsys):
    # A failing write must warn, bump the error count, and leave close() well-behaved
    # (training keeps running) — never propagate the exception.
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(ckptmod.torch, "save", boom)
    saver = AsyncCheckpointSaver(enabled=True)
    saver.save(_state(), str(tmp_path), "c.pth.tar", verbose=False)
    saver.close()
    assert saver.errors == 1
    assert "checkpoint save FAILED" in capsys.readouterr().out
    assert not (tmp_path / "c.pth.tar").exists()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
