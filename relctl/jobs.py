"""
JobManager — launch long runs detached, track them across relctl restarts/SSH drops,
and scrape their logs for live progress.

Background launch uses a detached session (start_new_session=True, the nohup
equivalent) so the run survives quitting relctl. The child is its own process-group
leader, so stop() can signal the whole tree via killpg. tmux is used instead when
available and requested. A registry at pred_ssl/.relctl/jobs.json persists jobs.
"""

import json
import os
import re
import shlex
import signal
import subprocess
import time
from datetime import datetime


_EPOCH = re.compile(r"Epoch \[(\d+)/(\d+)\]")
_PERFACTOR = re.compile(r"PerFactor:\s+(.*)")
_BEST = re.compile(r"\*BEST\*")
_VALACC = re.compile(r"Val Acc@1:\s+([\d.]+)%")
_SHOT = re.compile(r"(\d+)-shot:\s+([\d.]+)%")


class JobManager:
    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.state_dir = os.path.join(repo_root, "pred_ssl", ".relctl")
        self.registry = os.path.join(self.state_dir, "jobs.json")
        self.overlay_dir = os.path.join(self.state_dir, "overlays")
        self.jobs = self._load()
        self._procs = {}    # id -> Popen for jobs launched in THIS session (lets us reap)

    # ------------------------------------------------------------ persistence
    def _load(self):
        if os.path.isfile(self.registry):
            try:
                with open(self.registry) as f:
                    return json.load(f)
            except (ValueError, OSError):
                return []
        return []

    def _save(self):
        os.makedirs(self.state_dir, exist_ok=True)
        tmp = self.registry + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.jobs, f, indent=1)
        os.replace(tmp, self.registry)

    def _next_id(self):
        return (max((j["id"] for j in self.jobs), default=0) + 1)

    # ------------------------------------------------------------------ launch
    def overlay_path(self, ts):
        return os.path.join(self.overlay_dir, "run_%s.yaml" % ts)

    def _wrapper(self, commands, conda_env, banner=True):
        body = " && ".join("( %s )" % c for c in commands)
        head = (
            'cd %s || exit 1\n'
            'if command -v conda >/dev/null 2>&1; then '
            'source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null; '
            'conda activate %s 2>/dev/null || true; fi\n'
            % (shlex.quote(self.repo_root), shlex.quote(conda_env))
        )
        if banner:
            head += 'echo "[relctl] $(date) starting"\n'
            # capture rc BEFORE $(date), which would otherwise clobber $? with date's
            # own (zero) exit status during command substitution.
            return head + body + '\nrc=$?\necho "[relctl] $(date) exit $rc"\n'
        return head + body + "\n"

    def run_inline(self, plan, conda_env):
        """Run a quick/foreground plan with live output to the terminal. Returns rc."""
        wrapper = self._wrapper(plan.commands, conda_env, banner=False)
        return subprocess.run(["bash", "-c", wrapper], cwd=self.repo_root).returncode

    def launch(self, plan, conda_env, backend="nohup"):
        """Launch a background plan; returns the job dict. Caller has already written
        the overlay (plan.overlay_path points at it)."""
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_abs = os.path.join(self.repo_root, plan.log.lstrip("./")) if plan.log else \
            os.path.join(self.state_dir, "runs", "job_%s.log" % ts)
        os.makedirs(os.path.dirname(log_abs), exist_ok=True)
        wrapper = self._wrapper(plan.commands, conda_env, banner=True)

        job = {
            "id": self._next_id(), "action": plan.action, "tag": plan.tag,
            "backend": backend, "log": log_abs, "save_dir": plan.save_dir,
            "ckpt": plan.ckpt, "overlay": plan.overlay_path,
            "cmd": " ; ".join(plan.commands), "started": ts, "pid": None, "tmux": None,
        }

        if backend == "tmux" and self._have_tmux():
            sess = "relctl_%s_%d" % (plan.action, job["id"])
            script = os.path.join(self.state_dir, "runs", "wrap_%s.sh" % ts)
            os.makedirs(os.path.dirname(script), exist_ok=True)
            with open(script, "w") as f:
                f.write("#!/bin/bash\n" + wrapper)
            subprocess.run(["tmux", "new-session", "-d", "-s", sess, "bash %s" % shlex.quote(script)],
                           check=True)
            job["tmux"] = sess
        else:
            job["backend"] = "nohup"
            mode = "ab" if plan.append_log else "wb"
            logf = open(log_abs, mode)
            proc = subprocess.Popen(["bash", "-c", wrapper], stdout=logf,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    cwd=self.repo_root)
            job["pid"] = proc.pid
            self._procs[job["id"]] = proc

        self.jobs.append(job)
        self._save()
        return job

    @staticmethod
    def _have_tmux():
        return subprocess.run(["bash", "-c", "command -v tmux"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    # ------------------------------------------------------------------- state
    def get(self, jid):
        for j in self.jobs:
            if j["id"] == jid:
                return j
        return None

    def _alive(self, job):
        if job.get("tmux"):
            r = subprocess.run(["tmux", "has-session", "-t", job["tmux"]],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return r.returncode == 0
        proc = self._procs.get(job["id"])
        if proc is not None:
            return proc.poll() is None     # poll() also reaps the zombie when finished
        pid = job.get("pid")
        if not pid:
            return False
        try:
            os.kill(pid, 0)                 # job from a previous relctl session
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _exit_code(self, job):
        """The wrapper prints '[relctl] <date> exit <code>' as its last line. Reading it
        is session-independent and survives relctl restarts / PID reuse."""
        for line in reversed(self.tail(job, 10)):
            m = re.search(r"\[relctl\].*\bexit (\d+)", line)
            if m:
                return int(m.group(1))
        return None

    def state(self, job):
        code = self._exit_code(job)
        if code is not None:
            return "DONE" if code == 0 else "FAILED"
        if self._alive(job):
            return "RUNNING"
        # gone but never printed an exit banner -> killed or crashed before finishing
        return "FAILED?"

    def progress(self, job):
        """A short live-progress string scraped from the log tail."""
        last_epoch = last_pf = last_val = last_shot = ""
        try:
            with open(job["log"], errors="ignore") as f:
                for line in f:
                    m = _EPOCH.search(line)
                    if m:
                        last_epoch = "ep %s/%s" % (m.group(1), m.group(2))
                    pf = _PERFACTOR.search(line)
                    if pf:
                        last_pf = pf.group(1).strip()
                    if _BEST.search(line):
                        v = _VALACC.search(line)
                        if v:
                            last_val = "best@1 %s%%" % v.group(1)
                    s = _SHOT.search(line)
                    if s:
                        last_shot = "%s-shot %s%%" % (s.group(1), s.group(2))
        except OSError:
            return "(no log yet)"
        bits = [b for b in (last_epoch, last_val, last_shot) if b]
        out = "  ".join(bits) if bits else "(starting...)"
        return out

    def perfactor(self, job):
        last = ""
        try:
            with open(job["log"], errors="ignore") as f:
                for line in f:
                    pf = _PERFACTOR.search(line)
                    if pf:
                        last = pf.group(1).strip()
        except OSError:
            pass
        return last

    def tail(self, job, n=40):
        try:
            with open(job["log"], errors="ignore") as f:
                return f.readlines()[-n:]
        except OSError:
            return []

    # -------------------------------------------------------------------- stop
    def stop(self, job, hard=False):
        sig = signal.SIGKILL if hard else signal.SIGTERM
        if job.get("tmux"):
            subprocess.run(["tmux", "kill-session", "-t", job["tmux"]],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        pid = job.get("pid")
        if not pid:
            return False
        try:
            os.killpg(pid, sig)          # whole process group (bash + python child)
        except ProcessLookupError:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return False
        return True

    def clear_finished(self):
        self.jobs = [j for j in self.jobs if self.state(j) == "RUNNING"]
        self._save()
