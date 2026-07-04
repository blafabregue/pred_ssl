"""
Unit tests for the Phase-2 pilot gate parser + evaluator (CPU, no training).

Run:  python -m pytest pred_ssl/tests/test_pilot_gate.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.scripts.check_pilot_gate import evaluate_gate, parse_log  # noqa: E402

FACTORS = "rotation hflip brightness contrast saturation hue grayscale blur".split()


def _epoch_line(e, E, total, ssl, pred_loss, pred_acc, with_ssl=True):
    if with_ssl:
        return (f"Epoch [{e}/{E}]  Loss: {total:.4f}  SSL_Loss: {ssl:.4f}  "
                f"Pred_Loss: {pred_loss:.4f}  Pred_Acc: {pred_acc:.2f}%  LR: 0.300000")
    return (f"Epoch [{e}/{E}]  Loss: {total:.4f}  "
            f"Pred_Loss: {pred_loss:.4f}  Pred_Acc: {pred_acc:.2f}%  LR: 0.300000")


def _perfactor_line(accs):
    return "  PerFactor: " + " ".join(f"{f}={a:.1f}" for f, a in zip(FACTORS, accs))


def _log(epoch_accs, ssl_vals, with_ssl=True):
    """epoch_accs: list of per-factor acc lists (one per epoch). ssl_vals: per-epoch ssl."""
    lines = []
    E = len(ssl_vals)
    for i, (accs, ssl) in enumerate(zip(epoch_accs, ssl_vals), start=1):
        pred_acc = sum(accs) / len(accs) if accs is not None else 0.0
        lines.append(_epoch_line(i, E, ssl + 0.05, ssl, 0.69, pred_acc, with_ssl))
        if accs is not None:
            lines.append(_perfactor_line(accs))
    return "\n".join(lines)


def test_parse_log_full():
    text = _log([[55] * 8, [60] * 8], [7.0, 5.0])
    p = parse_log(text)
    assert len(p["epochs"]) == 2
    assert p["epochs"][0]["ssl"] == 7.0 and p["epochs"][-1]["ssl"] == 5.0
    assert len(p["perfactor"]) == 2
    assert p["perfactor"][-1]["rotation"] == 60.0


def test_parse_log_min_uses_total_as_ssl():
    text = _log([None, None], [7.0, 5.0], with_ssl=False)
    p = parse_log(text)
    # no SSL_Loss field -> ssl proxy = total Loss (ssl+0.05 here)
    assert p["epochs"][0]["ssl"] == 7.05 and p["epochs"][-1]["ssl"] == 5.05
    assert p["perfactor"] == []


def test_gate_pass():
    accs_final = [55, 60, 58, 57, 56, 53, 75, 62]   # all in (52, 98) -> LEARNING
    text = _log([[50] * 8, accs_final], [7.0, 5.0])
    passed, rep = evaluate_gate(parse_log(text))
    assert passed, rep["reasons"]
    assert rep["n_learning"] == 8 and rep["ssl_ok"]


def test_gate_fail_stuck():
    text = _log([[50] * 8, [50.5] * 8], [7.0, 5.0])
    passed, rep = evaluate_gate(parse_log(text))
    assert not passed
    assert rep["n_learning"] == 0
    assert any("learning" in r for r in rep["reasons"])


def test_gate_fail_leak():
    accs = [99.5, 60, 58, 57, 56, 53, 75, 62]       # rotation pinned -> LEAK
    text = _log([[50] * 8, accs], [7.0, 5.0])
    passed, rep = evaluate_gate(parse_log(text))
    assert not passed
    assert rep["verdicts"]["rotation"][1] == "LEAK"
    assert any("leak" in r.lower() for r in rep["reasons"])


def test_gate_fail_ssl_flat():
    accs = [55, 60, 58, 57, 56, 53, 75, 62]
    text = _log([[50] * 8, accs], [7.0, 6.99])       # essentially no drop
    passed, rep = evaluate_gate(parse_log(text))
    assert not passed and not rep["ssl_ok"]


def test_gate_baseline_no_head_passes_on_ssl():
    text = _log([None, None], [7.0, 4.0], with_ssl=True)
    passed, rep = evaluate_gate(parse_log(text))
    assert passed and not rep["has_head"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
