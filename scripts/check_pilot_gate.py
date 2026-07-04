"""
Automated Phase-2 pilot gate.

Parses a pred_ssl pretraining log and enforces the checks that must pass before
scaling to the full ResNet-50 / 500-epoch matrix:

  1. The SSL (contrastive) loss decreased over the pilot (it is learning at all).
  2. No per-factor same/different accuracy is pinned at ~100% (a LEAK -> usually
     crop accidentally shared, or a trivial shortcut).
  3. Enough per-factor accuracies moved OFF ~50% (the relational signal is real
     and the head is learning it).

Usage:
    python pred_ssl/scripts/check_pilot_gate.py path/to/pilot.log
    python pred_ssl/scripts/check_pilot_gate.py pilot.log --leak 98 --stuck 52 --min-learning 3

Exit code 0 == gate PASS, 1 == gate FAIL (so it can gate a launch script).
"""

import argparse
import re
import sys

# Epoch line (with or without the SSL_Loss field).
_EPOCH_FULL = re.compile(
    r"Epoch \[(\d+)/(\d+)\]\s+Loss:\s+([\d.]+)\s+SSL_Loss:\s+([\d.]+)\s+"
    r"Pred_Loss:\s+([\d.]+)\s+Pred_Acc:\s+([\d.]+)%")
_EPOCH_MIN = re.compile(
    r"Epoch \[(\d+)/(\d+)\]\s+Loss:\s+([\d.]+)\s+Pred_Loss:\s+([\d.]+)\s+Pred_Acc:\s+([\d.]+)%")
_PERFACTOR = re.compile(r"PerFactor:\s+(.*)")


def parse_log(text):
    """Return {'epochs': [...], 'perfactor': [{factor: acc}, ...]}."""
    epochs, perfactor = [], []
    for line in text.splitlines():
        m = _EPOCH_FULL.search(line)
        if m:
            epochs.append({"epoch": int(m.group(1)), "total": float(m.group(3)),
                           "ssl": float(m.group(4)), "pred_loss": float(m.group(5)),
                           "pred_acc": float(m.group(6))})
            continue
        m = _EPOCH_MIN.search(line)
        if m:
            # No explicit SSL_Loss -> use total Loss as the SSL proxy.
            epochs.append({"epoch": int(m.group(1)), "total": float(m.group(3)),
                           "ssl": float(m.group(3)), "pred_loss": float(m.group(4)),
                           "pred_acc": float(m.group(5))})
            continue
        m = _PERFACTOR.search(line)
        if m:
            d = {}
            for tok in m.group(1).split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
            if d:
                perfactor.append(d)
    return {"epochs": epochs, "perfactor": perfactor}


def evaluate_gate(parsed, leak=98.0, stuck=52.0, min_learning=3, min_ssl_drop=0.02):
    """Return (passed: bool, report: dict)."""
    epochs = parsed["epochs"]
    perfactor = parsed["perfactor"]
    reasons = []

    # --- (1) SSL loss decreased ---
    ssl_ok = False
    ssl_first = ssl_last = None
    if len(epochs) >= 2:
        ssl_first, ssl_last = epochs[0]["ssl"], epochs[-1]["ssl"]
        ssl_ok = ssl_last < ssl_first * (1.0 - min_ssl_drop)
        if not ssl_ok:
            reasons.append(f"SSL loss did not decrease enough: {ssl_first:.4f} -> {ssl_last:.4f}")
    else:
        reasons.append("need >= 2 epochs to judge the SSL-loss trend")

    # --- (2)/(3) per-factor verdicts (from the final PerFactor line) ---
    verdicts = {}
    n_learning = 0
    has_head = len(perfactor) > 0
    if has_head:
        final = perfactor[-1]
        for factor, acc in final.items():
            if acc >= leak:
                v = "LEAK"
            elif acc <= stuck:
                v = "STUCK"
            else:
                v = "LEARNING"
                n_learning += 1
            verdicts[factor] = (acc, v)
        leaks = [f for f, (_, v) in verdicts.items() if v == "LEAK"]
        if leaks:
            reasons.append(f"factor(s) pinned at ~100% (likely leak): {', '.join(leaks)}")
        if n_learning < min_learning:
            reasons.append(f"only {n_learning} factor(s) learning (need >= {min_learning})")
        head_ok = (not leaks) and (n_learning >= min_learning)
    else:
        head_ok = True  # baseline / no relational head -> only the SSL trend matters

    passed = ssl_ok and head_ok
    return passed, {
        "ssl_ok": ssl_ok, "ssl_first": ssl_first, "ssl_last": ssl_last,
        "verdicts": verdicts, "n_learning": n_learning, "has_head": has_head,
        "reasons": reasons,
    }


def _print_report(passed, rep):
    print("=" * 60)
    print("Phase-2 pilot gate")
    print("=" * 60)
    if rep["ssl_first"] is not None:
        arrow = "OK" if rep["ssl_ok"] else "FAIL"
        print(f"  SSL loss: {rep['ssl_first']:.4f} -> {rep['ssl_last']:.4f}  [{arrow}]")
    if rep["has_head"]:
        print("  Per-factor (final):")
        for factor, (acc, v) in rep["verdicts"].items():
            print(f"    {factor:11s} {acc:5.1f}%  {v}")
        print(f"  learning factors: {rep['n_learning']}")
    else:
        print("  (no relational head in this run)")
    if rep["reasons"]:
        print("  Issues:")
        for r in rep["reasons"]:
            print(f"    - {r}")
    print("-" * 60)
    print(f"  GATE: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="pred_ssl Phase-2 pilot gate checker")
    ap.add_argument("logfile")
    ap.add_argument("--leak", type=float, default=98.0, help="per-factor LEAK threshold (%)")
    ap.add_argument("--stuck", type=float, default=52.0, help="per-factor STUCK threshold (%)")
    ap.add_argument("--min-learning", type=int, default=3, help="min factors that must be learning")
    ap.add_argument("--min-ssl-drop", type=float, default=0.02, help="min fractional SSL-loss drop")
    args = ap.parse_args()

    with open(args.logfile) as f:
        parsed = parse_log(f.read())
    passed, rep = evaluate_gate(parsed, leak=args.leak, stuck=args.stuck,
                                min_learning=args.min_learning, min_ssl_drop=args.min_ssl_drop)
    _print_report(passed, rep)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
