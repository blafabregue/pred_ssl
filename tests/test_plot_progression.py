"""
Tests for the per-model progression aggregation (scripts/plot_progression.py):
the per-epoch mean/std band over seeds and its handling of ragged seed coverage.

Run:  python -m pytest pred_ssl/tests/test_plot_progression.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.scripts.plot_progression import aggregate_progression  # noqa: E402


def test_band_mean_and_std_over_seeds():
    # three seeds, same epochs
    seeds = [
        {10: 20.0, 20: 30.0},
        {10: 22.0, 20: 34.0},
        {10: 24.0, 20: 38.0},
    ]
    xs, means, stds, counts = aggregate_progression(seeds)
    assert xs == [10, 20]
    assert means == [22.0, 34.0]
    assert counts == [3, 3]
    assert abs(stds[0] - 2.0) < 1e-9                 # stdev(20,22,24) = 2.0
    assert abs(stds[1] - 4.0) < 1e-9                 # stdev(30,34,38) = 4.0


def test_single_seed_gives_zero_band():
    xs, means, stds, counts = aggregate_progression([{5: 12.0, 10: 15.0}])
    assert xs == [5, 10] and stds == [0.0, 0.0] and counts == [1, 1]


def test_ragged_epochs_use_available_seeds():
    # seed B stopped early at epoch 10; epoch 20 has only seed A
    seeds = [{10: 10.0, 20: 20.0}, {10: 12.0}]
    xs, means, stds, counts = aggregate_progression(seeds)
    assert xs == [10, 20]
    assert means[0] == 11.0 and counts[0] == 2 and abs(stds[0] - math.sqrt(2)) < 1e-9
    assert means[1] == 20.0 and counts[1] == 1 and stds[1] == 0.0


def test_min_seeds_filter_drops_thin_epochs():
    seeds = [{10: 10.0, 20: 20.0}, {10: 12.0}]
    xs, means, stds, counts = aggregate_progression(seeds, min_seeds=2)
    assert xs == [10] and counts == [2]              # epoch 20 (1 seed) dropped


def test_empty_input():
    assert aggregate_progression([]) == ([], [], [], [])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
