"""Forecast-evaluation metrics over the (call, τ) prediction frame — DESIGN §7.1.

The **prediction frame** is the common currency of evaluation: one row per
(call, horizon) with at least

    call_id, ticker, as_of (ISO date), horizon, y_true, y_pred

and, for the headline metric, a persistence baseline column `y_persistence`
(the naive forecast a model must beat — for level-v that's `v_pre`; for Δv it's
0). Metrics:

- **MSE / MAE** on the target (log RV by default) — literature comparability.
- **R²_OOS vs. persistence** (Gu–Kelly–Xiu [R26]): `1 − Σ(y−ŷ)² / Σ(y−ŷ_base)²`
  — the *headline* metric, not gameable by ticker-identity memorization the way
  raw MSE is.
- **Cross-sectional Spearman per calendar quarter** — what a practitioner
  ranking risk actually uses; reported as the equal-weight mean over quarters.

Array functions are pure (numpy in, float out); frame helpers add the grouping
(`by horizon`, `by quarter`) that `ecvol report` (T2.3) consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Canonical prediction-frame column names.
COL_TRUE = "y_true"
COL_PRED = "y_pred"
COL_BASELINE = "y_persistence"


def _clean_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either side is NaN (excluded targets carry NaN)."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _clean_pair(y_true, y_pred)
    return float(np.mean((a - b) ** 2)) if a.size else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _clean_pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> float:
    """Out-of-sample R² vs. a baseline forecast (DESIGN §7.1; Gu–Kelly–Xiu [R26]).

    `1 − Σ(y−ŷ)² / Σ(y−ŷ_baseline)²`. >0 means the model beats the baseline;
    0 ties it; <0 is worse. NaN if the baseline sum of squares is 0.
    """
    a = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    base = np.asarray(y_baseline, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(p) | np.isnan(base))
    a, p, base = a[keep], p[keep], base[keep]
    if a.size == 0:
        return float("nan")
    ss_res = float(np.sum((a - p) ** 2))
    ss_base = float(np.sum((a - base) ** 2))
    if ss_base == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_base


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation; NaN if <2 points or either side is constant."""
    a, b = _clean_pair(y_true, y_pred)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    rho = spearmanr(a, b).statistic
    return float(rho)


def quarter_of(as_of: str) -> str:
    """Calendar-quarter key from an ISO date, e.g. '2020-05-14' → '2020Q2'."""
    year, month = int(as_of[:4]), int(as_of[5:7])
    return f"{year}Q{(month - 1) // 3 + 1}"


# --- frame-level helpers -----------------------------------------------------


def spearman_by_quarter(
    df: pd.DataFrame, *, true=COL_TRUE, pred=COL_PRED, as_of="as_of"
) -> tuple[float, dict[str, float]]:
    """Cross-sectional Spearman per calendar quarter; (equal-weight mean, per-quarter).

    Quarters with <2 valid points or a constant side yield NaN and are excluded
    from the mean (DESIGN §7.1: a practitioner's per-quarter risk ranking).
    """
    quarters = df[as_of].map(quarter_of)
    per_quarter: dict[str, float] = {}
    for q, g in df.groupby(quarters):
        rho = spearman(g[true].to_numpy(), g[pred].to_numpy())
        if not np.isnan(rho):
            per_quarter[str(q)] = rho
    mean = float(np.mean(list(per_quarter.values()))) if per_quarter else float("nan")
    return mean, dict(sorted(per_quarter.items()))


def metrics_by_horizon(
    df: pd.DataFrame,
    *,
    true=COL_TRUE,
    pred=COL_PRED,
    baseline=COL_BASELINE,
    horizon="horizon",
    as_of="as_of",
) -> dict[int, dict[str, float]]:
    """Per-horizon MSE / MAE / R²_OOS / mean-quarter-Spearman over the frame.

    `baseline` (persistence) drives R²_OOS; pass a column the caller has filled
    appropriately for the target (v_pre for level-v, 0 for Δv).
    """
    out: dict[int, dict[str, float]] = {}
    for h, g in df.groupby(horizon):
        yt, yp = g[true].to_numpy(), g[pred].to_numpy()
        row = {"n": int(len(g)), "mse": mse(yt, yp), "mae": mae(yt, yp)}
        if baseline in g.columns:
            row["r2_oos"] = r2_oos(yt, yp, g[baseline].to_numpy())
        row["spearman_q"] = spearman_by_quarter(g, true=true, pred=pred, as_of=as_of)[0]
        out[int(h)] = row
    return dict(sorted(out.items()))
