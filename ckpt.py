"""
Asynchronous checkpoint saver.

torch.save of a ResNet-50 checkpoint (~140 MB) is disk I/O — several seconds on a
cluster's network filesystem — during which training would otherwise sit idle. This
moves the slow serialize+write onto a background thread so pretraining resumes
immediately, WITHOUT the two race hazards that make naive async saving unsafe:

  1. Torn reads: `model.state_dict()` returns live references to the parameter
     tensors, which the very next optimizer.step() mutates. So the main thread first
     takes a DECOUPLED CPU snapshot (snapshot_to_cpu); the background thread only ever
     touches that immutable copy. The GPU->host copy is a fast memcpy; the slow disk
     write is what runs in the background.
  2. Partial files on a kill: the writer writes to `<name>.tmp` then os.replace()s it
     into place (atomic on the same filesystem), so a job killed at the SLURM wall-clock
     limit never leaves a truncated checkpoint — resume always finds a complete file.

A single worker thread drains a FIFO queue, so saves never overlap or reorder. Any
exception in the worker is caught and reported (errors counter) — a failed write warns
but never crashes training. close() flushes the queue (join), so the final checkpoint
is guaranteed on disk before the process exits. Set async_checkpoint=false (or
enabled=False) to fall back to fully-synchronous saving.
"""

import os
import queue
import threading

import torch


def snapshot_to_cpu(obj):
    """Deep copy of a checkpoint state with every tensor moved to a fresh CPU tensor.

    Decouples the snapshot from the live (training-mutated) GPU tensors. `copy=True`
    forces a real copy even for tensors already on CPU (CPU-only training), so the
    result is always safe to serialize from another thread.
    """
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return type(obj)((k, snapshot_to_cpu(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(snapshot_to_cpu(v) for v in obj)
    return obj


def _write(state, save_dir, filename, verbose):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)              # atomic: never a half-written checkpoint on disk
    if verbose:
        print(f"  => Saved checkpoint: {path}", flush=True)


class AsyncCheckpointSaver:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.errors = 0
        if enabled:
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self._worker, name="ckpt-saver",
                                            daemon=True)
            self._thread.start()

    def save(self, state, save_dir, filename, verbose=True, snapshot=True):
        """Queue (or, if disabled, synchronously write) a checkpoint.

        snapshot=True takes the CPU snapshot here; pass an already-snapshotted state
        with snapshot=False to reuse one snapshot across several filenames.
        """
        snap = snapshot_to_cpu(state) if snapshot else state
        if not self.enabled:
            _write(snap, save_dir, filename, verbose)
            return
        self._q.put((snap, save_dir, filename, verbose))

    def _worker(self):
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                _write(*item)
            except BaseException as e:  # noqa: BLE001 — the saver thread must never die silently
                self.errors += 1
                name = item[2] if item else "?"
                print(f"  !! checkpoint save FAILED ({name}): {type(e).__name__}: {e}",
                      flush=True)
            finally:
                self._q.task_done()

    def close(self):
        """Flush every queued write and stop the worker (blocks until the disk is caught up)."""
        if not self.enabled:
            return
        self._q.put(None)
        self._thread.join()
        if self.errors:
            print(f"  !! {self.errors} checkpoint save(s) failed during this run", flush=True)
