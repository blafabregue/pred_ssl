"""
Preflight — reuse run.sh's prerequisite checklist instead of re-implementing it.

We shell out to `bash pred_ssl/run.sh --dry-run` with the right MODE + env and parse its
[ OK ] / [warn] / [FAIL] lines. run.sh exits 0 on --dry-run regardless, so this is
always safe to call; it never launches anything.
"""

import re
import subprocess

_MARKER = re.compile(r"\[\s*(OK|warn|FAIL)\s*\]\s+(.*)")
_LEVEL = {"OK": "ok", "warn": "warn", "FAIL": "fail"}


def run(repo_root, mode, env_overrides, timeout=90):
    """Return (checks, raw) where checks is a list of (level, message)."""
    import os
    env = dict(os.environ)
    env["MODE"] = mode
    for k, v in env_overrides.items():
        env[k] = str(v)
    try:
        p = subprocess.run(["bash", "pred_ssl/run.sh", "--dry-run"], cwd=repo_root,
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        raw = p.stdout.decode("utf-8", "ignore")
    except subprocess.TimeoutExpired:
        return [("warn", "preflight timed out (run.sh --dry-run took too long)")], ""
    except OSError as e:
        return [("warn", "could not run run.sh --dry-run: %s" % e)], ""

    checks = []
    for line in raw.splitlines():
        m = _MARKER.search(line)
        if m:
            checks.append((_LEVEL[m.group(1)], m.group(2).strip()))
    if not checks:
        checks = [("warn", "no preflight markers parsed from run.sh output")]
    return checks, raw


def env_for(model):
    """The env run.sh's preflight needs to check the right things."""
    return {
        "GPU": model.value("GPU"),
        "CONDA_ENV": model.value("conda_env"),
        "IN100": model.value("IN100"),
        "CUB": model.value("CUB"),
        "FLOWERS": model.value("FLOWERS"),
        "ARCH": model.value("arch"),
        "EPOCHS": model.value("epochs"),
        "FRAMEWORKS": " ".join(model.matrix_frameworks) if model.action == "matrix" else model.framework,
        "EXPERIMENTS": " ".join(model.matrix_experiments) if model.action == "matrix" else model.experiment,
    }
