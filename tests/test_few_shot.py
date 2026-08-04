"""
Tests for the two Flowers-102 few-shot protocols.

The reason both exist: our numbers were all measured with a lightly regularized
Adam fit, while AugSelf's Table 3 uses LBFGS with a swept L2 penalty on normalized
features. On 102 classes over 2048 dimensions that difference is worth several
points, so the two are not interchangeable and must not land in the same table.

What these tests pin is exactly that separation:
  - both protocols see the SAME sampled shots, so --protocol both compares
    objectives rather than draws;
  - the reference protocol never touches the test set when selecting its penalty;
  - its output line does NOT match extract_results' regex, so enabling it cannot
    silently change results.csv.

Run:  python -m pytest pred_ssl/tests/test_few_shot.py -q
"""

import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.eval.few_shot import (  # noqa: E402
    _fit_lbfgs, _sample_shots, few_shot_trial, few_shot_trial_ref)

DEV = torch.device("cpu")
C, D = 20, 256


def _features(n_train=10, n_test=25, sep=0.7, noise=2.0, seed=0):
    """Deliberately hard: heavy overlap, so an unregularized fit can overfit."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(C, D, generator=g) * sep

    def make(n):
        y = torch.arange(C).repeat_interleave(n)
        return centers[y] + torch.randn(len(y), D, generator=g) * noise, y

    return make(n_train), make(n_test)


# ---------------------------------------------------------------------------
# The support set is shared, so the protocols differ only in the classifier
# ---------------------------------------------------------------------------

def test_sampling_is_reproducible_and_balanced():
    (_, train_y), _ = _features()
    a, b = _sample_shots(train_y, 5, C, 7), _sample_shots(train_y, 5, C, 7)
    assert torch.equal(a, b)
    assert len(a) == 5 * C
    assert all((train_y[a] == c).sum().item() == 5 for c in range(C))


def test_different_seeds_draw_different_shots():
    (_, train_y), _ = _features()
    assert not torch.equal(_sample_shots(train_y, 5, C, 7),
                           _sample_shots(train_y, 5, C, 8))


# ---------------------------------------------------------------------------
# Both protocols learn
# ---------------------------------------------------------------------------

def test_both_protocols_beat_chance():
    (trf, trY), (tef, teY) = _features()
    chance = 100.0 / C
    for k in (5, 10):
        ours = few_shot_trial(trf, trY, tef, teY, k, D, C, 0.03, 250, 7, DEV)
        ref = few_shot_trial_ref(trf, trY, tef, teY, k, D, C, 12, 7, DEV)
        assert ours > chance and ref > chance


def test_reference_protocol_handles_one_shot():
    # k=1 leaves nothing to hold out; it must fall back rather than crash
    (trf, trY), (tef, teY) = _features()
    assert few_shot_trial_ref(trf, trY, tef, teY, 1, D, C, 8, 7, DEV) > 100.0 / C


def test_reference_protocol_accepts_an_explicit_selection_split():
    (trf, trY), (tef, teY) = _features()
    (self_f, self_y), _ = _features(seed=1)
    acc = few_shot_trial_ref(trf, trY, tef, teY, 5, D, C, 8, 7, DEV,
                             sel_f=self_f, sel_y=self_y)
    assert 0.0 <= acc <= 100.0


def test_penalty_selection_never_sees_the_test_set():
    # if the sweep were scored on test_f, corrupting test labels would change the
    # chosen weight and hence the reported accuracy on a clean test set
    (trf, trY), (tef, teY) = _features()
    clean = few_shot_trial_ref(trf, trY, tef, teY, 5, D, C, 8, 7, DEV)
    shuffled = teY[torch.randperm(len(teY), generator=torch.Generator().manual_seed(3))]
    # fit with garbage test labels, then re-score: selection must be unaffected
    few_shot_trial_ref(trf, trY, tef, shuffled, 5, D, C, 8, 7, DEV)
    assert few_shot_trial_ref(trf, trY, tef, teY, 5, D, C, 8, 7, DEV) == clean


def test_stronger_penalty_shrinks_the_weights():
    (trf, trY), _ = _features()
    idx = _sample_shots(trY, 5, C, 7)
    X = torch.nn.functional.normalize(trf[idx], dim=1)
    y = trY[idx]
    weak = _fit_lbfgs(X, y, C, 1e-6, DEV).weight.norm().item()
    strong = _fit_lbfgs(X, y, C, 1e2, DEV).weight.norm().item()
    assert strong < weak


# ---------------------------------------------------------------------------
# results.csv must stay on one protocol
# ---------------------------------------------------------------------------

def test_reference_lines_are_invisible_to_extract_results():
    shot = re.compile(r"(\d+)-shot:\s+([\d.]+)%\s+\(.*?([\d.]+)%\)")
    assert shot.search("  5-shot: 60.4% (± 1.2%)")
    assert not shot.search("  5-shot/ref: 78.5% (± 1.2%)")
    assert not shot.search("  10-shot/ref: 81.2% (± 0.9%)")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
