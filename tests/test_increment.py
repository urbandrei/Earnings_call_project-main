"""Result Table 8 helpers: the company-resampled relative MSE reduction and the reading rule."""

import numpy as np

from ecvol.data.targets import compute_call_targets
from ecvol.eval.increment import THRESHOLD, rel_reduction_ci, verdict


def test_rel_reduction_point_and_interval():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(20), 5)
    y = rng.normal(size=100)
    ref = y + rng.normal(scale=1.0, size=100)
    pred = y + 0.5 * (ref - y)  # halves every error → MSE falls by 75%
    pt, lo, hi = rel_reduction_ci(y, pred, ref, clusters)
    assert abs(pt - 0.75) < 1e-12
    assert abs(lo - 0.75) < 1e-9 and abs(hi - 0.75) < 1e-9  # constant ratio in every resample


def test_rel_reduction_is_deterministic_and_zero_for_identical_models():
    y = np.arange(10.0)
    ref = y + 1.0
    clusters = np.arange(10) % 3
    assert rel_reduction_ci(y, ref, ref, clusters) == (0.0, 0.0, 0.0)
    a = rel_reduction_ci(y, y + 0.5, ref, clusters)
    assert a == rel_reduction_ci(y, y + 0.5, ref, clusters)


def test_verdict_reading_rule():
    assert verdict(THRESHOLD + 0.01, 0.2) == "improvement"
    assert verdict(-0.2, THRESHOLD - 0.001) == "no meaningful benefit"
    assert verdict(-0.05, 0.10) == "inconclusive"


def test_post_skip_shifts_only_the_post_window():
    closes, d, p = {}, np.datetime64("2021-01-04"), 100.0
    rng = np.random.default_rng(1)
    for _ in range(140):
        if np.is_busday(d):
            p *= 1 + rng.normal(scale=0.02)
            closes[str(d)] = p
        d += 1
    call = {"call_id": 1, "ticker": "X", "date": "2021-03-10", "call_type": "earnings"}
    base = compute_call_targets(call, closes, horizons=(3, 7))
    skip = compute_call_targets(call, closes, horizons=(3, 7), post_skip=1)
    for a, b in zip(base, skip, strict=True):
        assert a.as_of == b.as_of and a.v_pre == b.v_pre and a.rv_monthly == b.rv_monthly
        assert a.v_post != b.v_post and b.n_post == a.n_post
