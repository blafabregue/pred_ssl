"""Entry point + the --validate anti-drift check."""

import argparse
import os
import re
import sys

from .ui import make_ui


def _repo_root():
    # pred_ssl/relctl/__main__.py -> repo root is three dirs up
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate(repo_root):
    """Cross-check the KNOBS catalog against configs/*.yaml and the scripts' argparse.

    Catches the drift hazard of a hand-written catalog: a knob whose default no longer
    matches the YAML, a YAML key with no knob, or a cli_flag that no longer exists.
    Exit 0 == clean, 1 == drift found.
    """
    import yaml
    from .knobs import KNOBS, KNOBS_BY_KEY, FRAMEWORK_KNOBS, FRAMEWORKS, EXPERIMENTS
    from .config import _deep_merge, _load_yaml

    cfgdir = os.path.join(repo_root, "pred_ssl", "configs")
    problems = []
    info = []

    base = _load_yaml(os.path.join(cfgdir, "base.yaml"))

    # 1) every base.yaml key (except the framework selector) has a knob
    for k in base:
        if k == "framework":
            continue
        if k not in KNOBS_BY_KEY:
            problems.append("base.yaml key '%s' has no KNOBS entry" % k)

    # 2) value drift: a knob default that disagrees with the committed YAML
    def cmp_default(key, yaml_val):
        kn = KNOBS_BY_KEY.get(key)
        if kn is None:
            return
        dv = kn.default
        # normalise list/tuple + numeric
        if isinstance(dv, list) and isinstance(yaml_val, list):
            same = [float(x) if isinstance(x, (int, float)) else x for x in dv] == \
                   [float(x) if isinstance(x, (int, float)) else x for x in yaml_val]
        elif isinstance(dv, (int, float)) and isinstance(yaml_val, (int, float)):
            same = float(dv) == float(yaml_val)
        else:
            same = dv == yaml_val
        if not same:
            problems.append("default drift: %s catalog=%r but base.yaml=%r" % (key, dv, yaml_val))

    for k, v in base.items():
        if k != "framework":
            cmp_default(k, v)

    # 3) framework files: knobs present + default matches
    for fw in FRAMEWORKS:
        fwcfg = _load_yaml(os.path.join(cfgdir, "framework", fw + ".yaml"))
        for k, v in fwcfg.items():
            if k == "framework":
                continue
            if k not in KNOBS_BY_KEY:
                problems.append("framework/%s.yaml key '%s' has no KNOBS entry" % (fw, k))
                continue
            kn = KNOBS_BY_KEY[k]
            if kn.fw_scope not in (None, fw):
                # global knobs (lr, lr_schedule, schedule...) legitimately appear in fw files
                pass
            if kn.fw_scope == fw:
                if isinstance(kn.default, (int, float)) and isinstance(v, (int, float)):
                    if float(kn.default) != float(v):
                        problems.append("default drift: %s catalog=%r but %s.yaml=%r"
                                        % (k, kn.default, fw, v))
                elif kn.default != v:
                    problems.append("default drift: %s catalog=%r but %s.yaml=%r"
                                    % (k, kn.default, fw, v))
        # every declared framework knob exists in the file (or has a documented default)
        for k in FRAMEWORK_KNOBS[fw]:
            if k not in fwcfg and KNOBS_BY_KEY[k].note == "":
                info.append("framework knob '%s' not in %s.yaml (relies on code default)" % (k, fw))

    # 4) cli_flag existence: every flag a knob claims must exist in the right source
    def flags_in(path):
        try:
            with open(path) as f:
                src = f.read()
        except OSError:
            return set()
        return set(re.findall(r'add_argument\(\s*"(--[a-z0-9\-]+)"', src))

    train_flags = flags_in(os.path.join(repo_root, "pred_ssl", "train.py"))
    lincls_flags = flags_in(os.path.join(repo_root, "pred_ssl", "eval", "linear_probe.py"))
    fewshot_flags = flags_in(os.path.join(repo_root, "pred_ssl", "eval", "few_shot.py"))
    gate_flags = flags_in(os.path.join(repo_root, "pred_ssl", "scripts", "check_pilot_gate.py"))
    flagset = {"train": train_flags, "eval_lincls": lincls_flags,
               "eval_fewshot": fewshot_flags, "gate": gate_flags}

    for kn in KNOBS:
        if kn.cli_flag and kn.domain in flagset:
            if kn.cli_flag not in flagset[kn.domain]:
                problems.append("cli_flag drift: %s claims %s but it's not in %s argparse"
                                % (kn.key, kn.cli_flag, kn.domain))

    # 5) experiment files only set known coupled keys
    for exp in EXPERIMENTS:
        ecfg = _load_yaml(os.path.join(cfgdir, "experiment", exp + ".yaml"))
        for k in ecfg:
            if k not in KNOBS_BY_KEY:
                problems.append("experiment/%s.yaml key '%s' has no KNOBS entry" % (exp, k))

    # report
    print("relctl --validate: %d knobs checked against configs/ + argparse" % len(KNOBS))
    for m in info:
        print("  info: %s" % m)
    if problems:
        for p in problems:
            print("  DRIFT: %s" % p)
        print("FAIL (%d drift issue(s))" % len(problems))
        return 1
    print("OK — catalog is in sync with the committed configs and argparse.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="relctl", description="pred_ssl interactive control panel")
    ap.add_argument("--plain", action="store_true", help="force the zero-dependency plain tier")
    ap.add_argument("--rich", action="store_true", help="prefer the Rich tier (default if installed)")
    ap.add_argument("--tui", action="store_true",
                    help="(Phase 3) full-screen Textual dashboard; falls back to Rich for now")
    ap.add_argument("--validate", action="store_true",
                    help="check the knob catalog against configs/ + argparse, then exit")
    args = ap.parse_args(argv)

    repo_root = _repo_root()

    if args.validate:
        return validate(repo_root)

    ui = make_ui(prefer_rich=not args.plain, force_plain=args.plain)
    if args.tui and ui.rich:
        ui.note("info", "full Textual dashboard is Phase 3 — using the Rich tier for now")
    from .app import App
    try:
        App(repo_root, ui).run()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
