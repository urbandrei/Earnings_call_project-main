"""T2.1 metrics: MSE/MAE/R²_OOS/Spearman + frame helpers (DESIGN §7.1).

Checks use analytically known values, not a re-run of the implementation.
"""

import math

import numpy as np
import pandas as pd

from ecvol.eval import metrics as m


def test_mse_mae_known():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.0, 4.0, 3.0])  # errors 0, -2, 0
    assert math.isclose(m.mse(yt, yp), 4.0 / 3.0)
    assert math.isclose(m.mae(yt, yp), 2.0 / 3.0)


def test_r2_oos_anchors():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    base = np.array([0.0, 0.0, 0.0, 0.0])  # persistence
    # perfect prediction → 1; equals baseline → 0
    assert math.isclose(m.r2_oos(yt, yt, base), 1.0)
    assert math.isclose(m.r2_oos(yt, base, base), 0.0)
    # worse than baseline → negative
    worse = base - 5.0
    assert m.r2_oos(yt, worse, base) < 0.0


def test_r2_oos_zero_baseline_ss_is_nan():
    yt = np.array([2.0, 2.0])
    base = np.array([2.0, 2.0])  # SS_baseline == 0
    assert math.isnan(m.r2_oos(yt, np.array([1.0, 3.0]), base))


def test_spearman_monotone():
    yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(m.spearman(yt, 2 * yt + 1), 1.0)  # monotone increasing
    assert math.isclose(m.spearman(yt, -yt), -1.0)  # monotone decreasing
    assert math.isnan(m.spearman(yt, np.full(5, 7.0)))  # constant → undefined


def test_nan_rows_dropped():
    yt = np.array([1.0, 2.0, np.nan, 4.0])
    yp = np.array([1.0, np.nan, 3.0, 4.0])
    # only rows 0 and 3 survive → zero error
    assert m.mse(yt, yp) == 0.0
    assert m.mae(yt, yp) == 0.0


def test_quarter_of():
    assert m.quarter_of("2019-01-01") == "2019Q1"
    assert m.quarter_of("2020-05-14") == "2020Q2"
    assert m.quarter_of("2020-09-30") == "2020Q3"
    assert m.quarter_of("2021-12-31") == "2021Q4"


def _frame() -> pd.DataFrame:
    # two quarters, two horizons; y_pred imperfect, persistence baseline = 0
    rows = []
    for as_of in ("2020-01-10", "2020-02-10", "2020-04-10", "2020-05-10"):
        for h in (3, 30):
            rows.append({"as_of": as_of, "horizon": h})
    df = pd.DataFrame(rows)
    df["y_true"] = np.arange(1.0, len(df) + 1.0)
    df["y_pred"] = df["y_true"] + 0.5
    df["y_persistence"] = 0.0
    return df


def test_spearman_by_quarter_means_over_quarters():
    df = _frame()
    mean, per_q = m.spearman_by_quarter(df)
    assert set(per_q) == {"2020Q1", "2020Q2"}
    # y_pred is a monotone transform of y_true within each quarter → rho = 1
    assert all(math.isclose(v, 1.0) for v in per_q.values())
    assert math.isclose(mean, 1.0)


def test_metrics_by_horizon_shape_and_values():
    df = _frame()
    out = m.metrics_by_horizon(df)
    assert set(out) == {3, 30}
    for h in (3, 30):
        assert math.isclose(out[h]["mse"], 0.25)  # constant +0.5 error
        assert math.isclose(out[h]["mae"], 0.5)
        assert out[h]["r2_oos"] < 1.0  # imperfect vs zero baseline
        assert out[h]["n"] == 4
