"""T2.2 Stage-0 baselines: returns/EWMA/GARCH/HAR (DESIGN §6 Stage 0)."""

import math

import numpy as np

from ecvol.models import baselines as B


def test_log_rv_from_variance():
    assert math.isclose(B.log_rv_from_variance(0.04), 0.5 * math.log(0.04))
    assert math.isnan(B.log_rv_from_variance(0.0))
    assert math.isnan(B.log_rv_from_variance(-1.0))


def test_returns_through_respects_information_rule():
    # closes on 5 dates; as_of cuts off the future
    close = {
        "2020-01-02": 100.0,
        "2020-01-03": 101.0,
        "2020-01-06": 102.0,
        "2020-01-07": 103.0,
        "2020-01-08": 200.0,
    }
    r = B.returns_through(close, "2020-01-07")
    assert len(r) == 3  # returns between the 4 sessions ≤ as_of; the 01-08 jump excluded
    assert math.isclose(r[0], 1.0 / 100.0)
    assert B.returns_through(close, "2019-12-31").size == 0  # nothing on/before


def test_ewma_fixed_point_for_constant_abs_return():
    r = np.array([0.02, -0.02] * 100)  # |r| constant ⇒ var → r²
    assert math.isclose(B.ewma_variance(r), 0.02**2, rel_tol=1e-6)


def test_har_fit_recovers_known_linear_map():
    rng = np.random.default_rng(0)
    rv_d = rng.uniform(1e-4, 1e-2, 500)
    rv_w = rng.uniform(1e-4, 1e-2, 500)
    rv_m = rng.uniform(1e-4, 1e-2, 500)
    x = B.har_design(rv_d, rv_w, rv_m)
    true = np.array([0.5, 0.3, 0.1, 0.05])
    y = x @ true  # exact linear target
    coef = B.har_fit(x, y)
    assert np.allclose(coef, true, atol=1e-8)


def test_garch_none_on_short_series_finite_on_long():
    rng = np.random.default_rng(1)
    short = {f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}": 100.0 + i for i in range(50)}
    assert B.garch_log_rv_multi(short, "2099-01-01", (3, 30)) is None  # <100 obs

    # 400 sessions of GARCH-ish returns → a converging fit, finite per-τ forecasts
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
    dates = [
        d.date().isoformat() for d in __import__("pandas").bdate_range("2018-01-01", periods=400)
    ]
    close = dict(zip(dates, prices, strict=True))
    out = B.garch_log_rv_multi(close, dates[-1], (3, 7, 30))
    assert out is not None
    assert set(out) == {3, 7, 30}
    assert all(math.isfinite(v) for v in out.values())
