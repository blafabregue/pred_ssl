"""
relctl controller — the interactive menu loop. Talks only to ConfigModel (state),
the plan builder (actions.build_plan), JobManager (launch/monitor), preflight, and
the UI abstraction, so the exact same flow drives the plain and Rich tiers.
"""

import csv
import glob
import os
import subprocess
from datetime import datetime

from . import preflight
from .actions import build_plan
from .config import ConfigModel, ValidationError
from .jobs import JobManager
from .knobs import (GROUPS, KNOBS_BY_KEY, ACTIONS, ACTIONS_BY_KEY, FRAMEWORKS,
                    EXPERIMENTS, FRAMEWORK_KNOBS, DELTA_KEYS, knobs_in_group)


def fmtval(v):
    if isinstance(v, bool):
        return "ON" if v else "OFF"
    if isinstance(v, float):
        return ("%.6g" % v)
    if isinstance(v, list):
        return "[%s]" % ", ".join(fmtval(x) for x in v)
    if isinstance(v, dict):
        return "{%s}" % ", ".join("%s=%s" % (k, fmtval(x)) for k, x in v.items())
    return str(v)


def short(v, n=34):
    """Compact value for table/summary cells (long lists/dicts are summarised)."""
    if isinstance(v, list) and len(v) > 4:
        return "[%d items]" % len(v)
    if isinstance(v, dict):
        return "{%d keys} ▸" % len(v)
    s = fmtval(v)
    return s if len(s) <= n else s[:n - 1] + "…"


def trunc(s, n=48):
    return s if len(s) <= n else s[:n - 1] + "…"


# actions that need a checkpoint before they can run
_NEEDS_CKPT = {"eval", "eval_in100", "eval_rotation", "eval_cub", "eval_flowers"}


class App:
    def __init__(self, repo_root, ui):
        self.repo_root = repo_root
        self.ui = ui
        self.model = ConfigModel(repo_root)
        self.jobs = JobManager(repo_root)

    # ===================================================================== run
    def run(self):
        while True:
            try:
                if not self.main_menu():
                    break
            except KeyboardInterrupt:
                self.ui.print()
                self.ui.note("info", "interrupted — back to menu (q to quit)")
            except EOFError:
                self.ui.print()
                self.ui.note("info", "EOF — bye")
                break

    # =============================================================== main menu
    def _running(self):
        return sum(1 for j in self.jobs.jobs if self.jobs.state(j) == "RUNNING")

    def main_menu(self):
        ui, m = self.ui, self.model
        ui.clear()
        act = ACTIONS_BY_KEY[m.action]
        scale = "x%g" % (m.value("batch_size") / 256.0) if m.value("lr_scale_by_batch") else "x1"
        ui.header("relctl · pred_ssl control panel", [
            "action=%s  fw=%s  exp=%s  arch=%s  epochs=%s  GPU=%s"
            % (m.action, m.framework, m.experiment, m.value("arch"), m.value("epochs"), m.value("GPU")),
            "base_lr = %g %s = %.4f   [lr_scale_by_batch=%s]"
            % (m.value("lr"), scale, m.base_lr(), "ON" if m.value("lr_scale_by_batch") else "OFF"),
            "overrides: %d edits   profile: %s   jobs running: %d"
            % (len(m.dirty_keys()), m.profile_name or "<unsaved>", self._running()),
        ])
        ui.print()
        ui.print(ui.bold("  CONFIGURE"))
        for i, (gk, gl) in enumerate(GROUPS, 1):
            ui.print("   %d) %-22s %s" % (i, gl, ui.dim(self._group_summary(gk))))
        ui.print()
        ui.print(ui.bold("  SELECT"))
        ui.print("   f) framework [%s]   e) experiment [%s]   a) action [%s]"
                 % (m.framework, m.experiment, act.label))
        ui.print(ui.bold("  RUN"))
        ui.print("   r) preview & launch ▸   j) jobs (%d) ▸   o) results ▸" % len(self.jobs.jobs))
        ui.print(ui.bold("  PROFILES"))
        ui.print("   s) save   l) load   z) reset edits   v) verify resolved config   q) quit")
        ui.rule()
        for level, msg in m.warnings():
            ui.note(level, msg)

        c = ui.ask("choice").strip().lower()
        if c in ("q", "quit", "exit"):
            return False
        if c.isdigit() and 1 <= int(c) <= len(GROUPS):
            self.edit_group(GROUPS[int(c) - 1][0])
        elif c == "f":
            self.select_from("framework", FRAMEWORKS, m.set_framework, m.framework)
        elif c == "e":
            self.select_from("experiment", EXPERIMENTS, m.set_experiment, m.experiment)
        elif c == "a":
            self.select_action()
        elif c == "r":
            self.preview_launch()
        elif c == "j":
            self.jobs_screen()
        elif c == "o":
            self.results_screen()
        elif c == "s":
            self.save_profile()
        elif c == "l":
            self.load_profile()
        elif c == "z":
            if ui.confirm("reset all %d edits to config defaults?" % len(m.dirty_keys())):
                m.reset()
                ui.note("ok", "edits cleared")
                ui.pause()
        elif c == "v":
            self.verify_config()
        elif c == "":
            pass
        else:
            ui.note("warn", "unknown choice: %s" % c)
            ui.pause()
        return True

    def _group_summary(self, gk):
        keys = [k.key for k in knobs_in_group(gk, self.model.framework)][:3]
        return "  ".join("%s=%s" % (k, short(self.model.value(k), 16)) for k in keys)

    # ============================================================ config editor
    def edit_group(self, gk):
        ui, m = self.ui, self.model
        label = dict(GROUPS)[gk]
        while True:
            ui.clear()
            ui.header("Configure · %s" % label)
            if gk == "model":
                ui.print(" framework=%s  (use 'f' at main menu to change)" % m.framework)
            knobs = knobs_in_group(gk, m.framework)
            rows = []
            for i, kn in enumerate(knobs, 1):
                mark = ui.bold("*") if m.is_dirty(kn.key) else " "
                valid = self._valid_str(kn)
                note = kn.note or kn.coupling
                rows.append([i, mark, kn.key, short(m.value(kn.key)),
                             trunc((valid + ("  " + note if note else "")).strip())])
            ui.table(["#", "", "key", "value", "valid / note"], rows)
            ui.rule()
            ui.print(ui.dim("  <#> edit · d=diff vs defaults · h <#>=help · b=back"))
            c = ui.ask("edit").strip().lower()
            if c in ("b", "back", "", "q"):
                return
            if c == "d":
                self._show_diff()
                continue
            if c.startswith("h"):
                self._knob_help(knobs, c)
                continue
            if c.isdigit() and 1 <= int(c) <= len(knobs):
                self.edit_knob(knobs[int(c) - 1])
            else:
                ui.note("warn", "pick a number, d, h <#>, or b")
                ui.pause()

    def _valid_str(self, kn):
        if kn.type == "enum":
            return "|".join(kn.valid)
        if kn.type == "bool":
            return "true|false"
        if isinstance(kn.valid, tuple):
            lo, hi = kn.valid
            if lo is not None and hi is not None:
                return "%g..%g" % (lo, hi)
            if lo is not None:
                return ">= %g" % lo
            if hi is not None:
                return "<= %g" % hi
        return {"list_int": "ints", "list_float": "floats", "list_str": "names",
                "dict_float": "dict", "path": "path"}.get(kn.type, "")

    def edit_knob(self, kn):
        ui, m = self.ui, self.model
        if kn.key == "delta":
            return self.delta_editor()
        if kn.note and ("DEAD" in kn.note or "GUARD" in kn.note):
            ui.note("warn", kn.note)
        ui.print(" %s — %s" % (ui.bold(kn.key), kn.doc))
        cur = m.value(kn.key)
        if kn.type == "bool":
            raw = ui.ask("set %s (true/false), Enter toggles" % kn.key)
            raw = (not cur) if raw == "" else raw
        elif kn.type == "enum":
            raw = ui.ask("%s (%s)" % (kn.key, "|".join(kn.valid)), default=cur)
        elif kn.type in ("list_int", "list_float", "list_str"):
            raw = ui.ask("%s (space/comma separated)" % kn.key, default=fmtval(cur).strip("[]"))
        else:
            raw = ui.ask(kn.key, default=cur)
        try:
            v = m.set(kn.key, raw)
            ui.note("ok", "%s = %s%s" % (kn.key, fmtval(v),
                    "" if m.is_dirty(kn.key) else "  (== default)"))
        except ValidationError as e:
            ui.note("err", str(e))
        ui.pause()

    def delta_editor(self):
        ui, m = self.ui, self.model
        while True:
            ui.clear()
            ui.header("Configure · Relational head · delta (per-factor min 'different' gap)")
            cur = m.value("delta")
            rows = [[chr(ord('a') + i), k, fmtval(cur.get(k, "?"))] for i, k in enumerate(DELTA_KEYS)]
            ui.table(["key", "factor", "value"], rows)
            ui.note("info", "all 5 keys are required at runtime; relctl always writes the full dict")
            ui.rule()
            c = ui.ask("letter to edit (b=back)").strip().lower()
            if c in ("b", "back", "", "q"):
                return
            idx = ord(c) - ord('a')
            if 0 <= idx < len(DELTA_KEYS):
                sub = DELTA_KEYS[idx]
                raw = ui.ask("delta.%s (> 0)" % sub, default=cur.get(sub))
                try:
                    m.set_delta_key(sub, raw)
                    ui.note("ok", "delta.%s = %s" % (sub, raw))
                except ValidationError as e:
                    ui.note("err", str(e))
                ui.pause()
            else:
                ui.note("warn", "pick a..%s or b" % chr(ord('a') + len(DELTA_KEYS) - 1))
                ui.pause()

    def _show_diff(self):
        ui, m = self.ui, self.model
        dirty = m.dirty_keys()
        if not dirty:
            ui.note("info", "no edits — all values are the committed config defaults")
        else:
            rows = [[k, fmtval(m.baseline(k)), fmtval(m.value(k))] for k in dirty]
            ui.table(["key", "default", "edited"], rows)
        ui.pause()

    def _knob_help(self, knobs, c):
        ui = self.ui
        parts = c.split()
        if len(parts) == 2 and parts[1].isdigit() and 1 <= int(parts[1]) <= len(knobs):
            kn = knobs[int(parts[1]) - 1]
            ui.print(" %s" % ui.bold(kn.key))
            ui.print("   %s" % kn.doc)
            if kn.coupling:
                ui.note("info", "coupling: " + kn.coupling)
            if kn.note:
                ui.note("warn", kn.note)
            ui.note("info", "source: %s%s" % (kn.domain, "  (YAML overlay)" if kn.yaml_only else
                    ("  (flag %s)" % kn.cli_flag if kn.cli_flag else "")))
        else:
            ui.note("warn", "usage: h <#>")
        ui.pause()

    # ================================================================ selectors
    def select_from(self, what, choices, setter, current):
        ui = self.ui
        ui.clear()
        ui.header("Select %s" % what)
        for i, ch in enumerate(choices, 1):
            ui.print("   %d) %s %s" % (i, ch, "  <- current" if ch == current else ""))
        c = ui.ask("pick (b=back)").strip().lower()
        if c in ("b", "back", ""):
            return
        if c.isdigit() and 1 <= int(c) <= len(choices):
            try:
                setter(choices[int(c) - 1])
                ui.note("ok", "%s = %s" % (what, choices[int(c) - 1]))
            except ValidationError as e:
                ui.note("err", str(e))
            ui.pause()

    def select_action(self):
        ui, m = self.ui, self.model
        ui.clear()
        ui.header("Select action to run")
        rows = [[i, a.label, ("bg" if a.background else "-"), a.needs]
                for i, a in enumerate(ACTIONS, 1)]
        ui.table(["#", "action", "mode", "prerequisite"], rows)
        c = ui.ask("pick (b=back)").strip().lower()
        if c in ("b", "back", ""):
            return
        if c.isdigit() and 1 <= int(c) <= len(ACTIONS):
            a = ACTIONS[int(c) - 1]
            m.action = a.key
            ui.note("ok", "action = %s — %s" % (a.label, a.doc))
            if a.key in _NEEDS_CKPT:
                m.eval_ckpt = self._pick_checkpoint(m.eval_ckpt)
            elif a.key == "resume":
                m.resume_ckpt = self._pick_checkpoint(m.resume_ckpt)
            elif a.key == "gate_check":
                m.gate_log = self._pick_log(m.gate_log)
            elif a.key == "matrix":
                self._configure_matrix()
            ui.pause()

    def _glob(self, pattern):
        return sorted(glob.glob(os.path.join(self.repo_root, pattern)))

    def _pick_checkpoint(self, current):
        ui, m = self.ui, self.model
        found = self._glob("pred_ssl/checkpoints/*/checkpoint_*.pth.tar")
        ui.print(ui.dim(" checkpoints found:"))
        for i, p in enumerate(found[-12:], 1):
            ui.print("   %d) %s" % (i, os.path.relpath(p, self.repo_root)))
        raw = ui.ask("checkpoint: number, path, or Enter to keep", default=current or "")
        if raw.isdigit() and found and 1 <= int(raw) <= len(found[-12:]):
            return os.path.relpath(found[-12:][int(raw) - 1], self.repo_root)
        return raw

    def _pick_log(self, current):
        ui = self.ui
        found = self._glob("pred_ssl/logs/*.log")
        for i, p in enumerate(found[-12:], 1):
            ui.print("   %d) %s" % (i, os.path.relpath(p, self.repo_root)))
        raw = self.ui.ask("log: number, path, or Enter", default=current or "")
        if raw.isdigit() and found and 1 <= int(raw) <= len(found[-12:]):
            return os.path.relpath(found[-12:][int(raw) - 1], self.repo_root)
        return raw

    def _configure_matrix(self):
        ui, m = self.ui, self.model
        fws = ui.ask("matrix frameworks (space sep)", default=" ".join(m.matrix_frameworks))
        m.matrix_frameworks = [x for x in fws.split() if x in FRAMEWORKS] or list(FRAMEWORKS)
        exps = ui.ask("matrix experiments (space sep)", default=" ".join(m.matrix_experiments))
        m.matrix_experiments = [x for x in exps.split() if x in EXPERIMENTS] or ["baseline", "relpred"]
        m.include_ablation = ui.confirm("include relpred_lambda0 ablation?", m.include_ablation)

    # ============================================================ preview/launch
    def preview_launch(self):
        ui, m = self.ui, self.model
        if m.action in _NEEDS_CKPT and not m.eval_ckpt:
            m.eval_ckpt = self._pick_checkpoint("")
        if m.action == "resume" and not m.resume_ckpt:
            m.resume_ckpt = self._pick_checkpoint("")

        plan = build_plan(m, overlay_path="pred_ssl/.relctl/overlays/run_<ts>.yaml")
        ui.clear()
        a = ACTIONS_BY_KEY[m.action]
        ui.header("Run · %s" % a.label, [a.doc])

        cfg = m.resolved_train_cfg()
        ui.rule("effective config (base <- %s <- %s <- your edits)" % (m.framework, m.experiment))
        ui.print("   arch=%s  epochs=%s  bs=%s  base_lr=%.4f  lr_schedule=%s"
                 % (cfg.get("arch"), cfg.get("epochs"), cfg.get("batch_size"), m.base_lr(),
                    cfg.get("lr_schedule")))
        ui.print("   rel_lambda=%s  aug_sharing=%s  blur_mode=%s  p_same=%s"
                 % (cfg.get("rel_lambda"), "ON" if cfg.get("aug_sharing") else "OFF",
                    cfg.get("blur_mode"), cfg.get("p_same")))
        if plan.ckpt:
            ui.print("   checkpoint: %s" % plan.ckpt)

        if plan.overlay_dict:
            ui.rule("YAML overlay (edited knobs train.py gets via --config-overlay)")
            for k, v in sorted(plan.overlay_dict.items()):
                ui.print("   %s: %s" % (k, fmtval(v)))

        ui.rule("command(s)")
        for cmd in plan.commands:
            ui.print(ui.dim("   $ ") + cmd)

        for level, msg in m.warnings():
            ui.note(level, msg)
        for n in plan.notes:
            ui.note("info", n)

        if plan.preflight_mode:
            ui.rule("preflight (run.sh --dry-run)")
            ui.note("info", "checking prerequisites… (the first run imports torch — "
                            "can take ~10-30s; Ctrl-C to skip)")
            try:
                checks, _ = preflight.run(self.repo_root, plan.preflight_mode,
                                          preflight.env_for(m))
                for level, msg in checks:
                    ui.note(level, msg)
                if any(lv == "fail" for lv, _ in checks):
                    ui.note("err", "a real run.sh run would ABORT on the [FAIL]s above")
            except KeyboardInterrupt:
                ui.note("warn", "preflight skipped (Ctrl-C) — launch options below")

        ui.rule()
        if plan.background:
            ui.print(ui.dim("   n) launch background (nohup)   t) launch in tmux   "
                            "f) foreground   d) bash dry-run   b) back"))
        else:
            ui.print(ui.dim("   x) run now (foreground)   d) bash dry-run   b) back"))
        c = ui.ask("choose").strip().lower()

        if c in ("b", "back", ""):
            return
        if c == "d":
            self._bash_dry_run(plan)
            return
        if plan.background and c in ("n", "t", "f"):
            self._launch(plan, background=(c != "f"), backend="tmux" if c == "t" else "nohup")
        elif (not plan.background) and c == "x":
            self._launch(plan, background=False, backend="nohup")
        else:
            ui.note("warn", "nothing launched")
            ui.pause()

    def _bash_dry_run(self, plan):
        ui = self.ui
        mode = plan.preflight_mode or "pilot"
        ui.note("info", "running: MODE=%s bash pred_ssl/run.sh --dry-run" % mode)
        _, raw = preflight.run(self.repo_root, mode, preflight.env_for(self.model))
        ui.print(raw)
        ui.pause()

    def _launch(self, plan, background, backend):
        ui, m = self.ui, self.model
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        if m.needs_overlay():
            rel = "pred_ssl/.relctl/overlays/run_%s.yaml" % ts
            m.write_overlay(os.path.join(self.repo_root, rel))
            plan = build_plan(m, overlay_path=rel)
        else:
            plan = build_plan(m, overlay_path=None)

        if not background:
            ui.note("info", "running in foreground — Ctrl-C to abort")
            ui.rule()
            rc = self.jobs.run_inline(plan, m.value("conda_env"))
            ui.rule()
            ui.note("ok" if rc == 0 else "err", "finished with exit code %d" % rc)
            ui.pause()
            return

        job = self.jobs.launch(plan, m.value("conda_env"), backend=backend)
        if job.get("tmux"):
            ui.note("ok", "launched job #%d in tmux session '%s'  log: %s"
                    % (job["id"], job["tmux"], os.path.relpath(job["log"], self.repo_root)))
        else:
            ui.note("ok", "launched job #%d  pid %s  log: %s"
                    % (job["id"], job["pid"], os.path.relpath(job["log"], self.repo_root)))
        ui.note("info", "returns to menu immediately — the run keeps going if you quit relctl")
        ui.pause()

    # =================================================================== jobs
    def jobs_screen(self):
        ui = self.ui
        while True:
            ui.clear()
            ui.header("Jobs")
            if not self.jobs.jobs:
                ui.note("info", "no jobs launched yet")
                ui.pause()
                return
            rows = []
            for j in self.jobs.jobs:
                st = self.jobs.state(j)
                rows.append([j["id"], j["action"], j["tag"] or "-",
                             j.get("pid") or j.get("tmux") or "-", st, self.jobs.progress(j),
                             os.path.basename(j["log"])])
            ui.table(["#", "action", "tag", "pid", "state", "progress", "log"], rows)
            ui.rule()
            ui.print(ui.dim("  t <#> tail -f · l <#> last40 · p <#> perfactor · s <#> stop · "
                            "k <#> kill · c clear-finished · r refresh · b back"))
            c = ui.ask("job").strip().lower()
            if c in ("b", "back", "", "q"):
                return
            if c == "r":
                continue
            if c == "c":
                self.jobs.clear_finished()
                continue
            parts = c.split()
            if len(parts) == 2 and parts[1].isdigit():
                job = self.jobs.get(int(parts[1]))
                if not job:
                    ui.note("warn", "no job #%s" % parts[1])
                    ui.pause()
                    continue
                self._job_action(parts[0], job)
            else:
                ui.note("warn", "usage: <cmd> <#>")
                ui.pause()

    def _job_action(self, cmd, job):
        ui = self.ui
        if cmd == "t":
            ui.note("info", "tail -f %s  (Ctrl-C to return; the job keeps running)" % job["log"])
            ui.rule()
            try:
                subprocess.run(["tail", "-f", job["log"]])
            except KeyboardInterrupt:
                pass
            except OSError as e:
                ui.note("warn", "tail failed: %s" % e)
                ui.pause()
        elif cmd == "l":
            ui.rule(os.path.basename(job["log"]))
            for line in self.jobs.tail(job, 40):
                ui.print("  " + line.rstrip())
            ui.pause()
        elif cmd == "p":
            ui.note("info", "PerFactor: " + (self.jobs.perfactor(job) or "(none yet)"))
            ui.pause()
        elif cmd in ("s", "k"):
            hard = cmd == "k"
            if ui.confirm("%s job #%d (%s)?" % ("SIGKILL" if hard else "stop", job["id"], job["tag"])):
                ok = self.jobs.stop(job, hard=hard)
                ui.note("ok" if ok else "warn", "signal sent" if ok else "could not signal (already gone?)")
                ui.pause()
        else:
            ui.note("warn", "unknown: %s" % cmd)
            ui.pause()

    # ================================================================ results
    def results_screen(self):
        ui = self.ui
        csv_path = os.path.join(self.repo_root, "pred_ssl", "results.csv")
        while True:
            ui.clear()
            ui.header("Results")
            if os.path.isfile(csv_path):
                self._show_results(csv_path)
            else:
                ui.note("info", "no results.csv yet — run 'x' to extract from logs")
            ui.rule()
            ui.print(ui.dim("  x) extract logs -> results.csv   o) show path   b) back"))
            c = ui.ask("results").strip().lower()
            if c in ("b", "back", "", "q"):
                return
            if c == "x":
                ui.rule()
                self.jobs.run_inline(build_plan_extract(self.model), self.model.value("conda_env"))
                ui.pause()
            elif c == "o":
                ui.note("info", csv_path)
                ui.pause()

    def _show_results(self, csv_path):
        ui = self.ui
        cols = ["framework", "experiment", "pretrain_pred_acc", "in100_acc1",
                "rotation_acc1", "cub200_acc1", "flowers_5shot", "flowers_10shot"]
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
        except OSError as e:
            ui.note("warn", str(e))
            return
        table = [[r.get(c, "") or "-" for c in cols] for r in rows]
        ui.table(["fw", "exp", "pred_acc", "in100", "rot", "cub", "5shot", "10shot"], table)

    # ================================================================ profiles
    def save_profile(self):
        ui, m = self.ui, self.model
        name = ui.ask("profile name", default=m.profile_name or "exp1").strip()
        if not name:
            return
        path = m.save_profile(name)
        ui.note("ok", "saved profile '%s' -> %s" % (name, os.path.relpath(path, self.repo_root)))
        ui.pause()

    def load_profile(self):
        ui, m = self.ui, self.model
        names = m.list_profiles()
        if not names:
            ui.note("info", "no saved profiles in pred_ssl/relctl/profiles/")
            ui.pause()
            return
        ui.clear()
        ui.header("Load profile")
        for i, n in enumerate(names, 1):
            ui.print("   %d) %s" % (i, n))
        c = ui.ask("pick (b=back)").strip().lower()
        if c.isdigit() and 1 <= int(c) <= len(names):
            try:
                m.load_profile(names[int(c) - 1])
                ui.note("ok", "loaded '%s'" % names[int(c) - 1])
            except ValidationError as e:
                ui.note("err", str(e))
            ui.pause()

    # ========================================================= verify resolved
    def verify_config(self):
        """Cross-check relctl's computed config against train.py --print-config."""
        ui, m = self.ui, self.model
        ui.clear()
        ui.header("Verify resolved config (relctl vs train.py --print-config)")
        ov_rel = ""
        if m.needs_overlay():
            ov_rel = "pred_ssl/.relctl/overlays/_verify.yaml"
            m.write_overlay(os.path.join(self.repo_root, ov_rel))
        cmd = ("python -m pred_ssl.train --framework %s --experiment %s --print-config %s"
               % (m.framework, m.experiment,
                  ("--config-overlay %s" % ov_rel) if ov_rel else ""))
        ui.note("info", cmd)
        wrapper = self.jobs._wrapper([cmd], m.value("conda_env"), banner=False)
        try:
            p = subprocess.run(["bash", "-c", wrapper], cwd=self.repo_root,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            out = p.stdout.decode("utf-8", "ignore")
            rc = p.returncode
        except (OSError, subprocess.TimeoutExpired) as e:
            ui.note("warn", "could not run train.py --print-config (%s)" % e)
            ui.note("info", "relctl computes the same merge itself; this is only a cross-check.")
            ui.pause()
            return
        ui.rule("train.py says")
        ui.print(out)
        if rc != 0:
            ui.note("warn", "train.py --print-config exited %d (env/torch not ready here?)" % rc)
            ui.note("info", "relctl computes the same base<-fw<-exp<-overlay merge itself; "
                            "this is only a cross-check.")
        ui.pause()


def build_plan_extract(model):
    model = model
    from .actions import build_plan as _bp
    saved = model.action
    model.action = "extract"
    plan = _bp(model)
    model.action = saved
    return plan
