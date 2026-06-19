"""Stage-0 econometric baselines — the honest floor (DESIGN §6 Stage 0).

Four forecasts of post-call log realized volatility `v_post(τ)`, using only
information available at day 0:

- **persistence** — `v_pre(τ)` (last period's vol predicts the next). For the Δv
  and HAR-residual targets the persistence forecast is 0 (handled by the caller).
- **EWMA** — RiskMetrics (λ=0.94) conditional variance from daily returns ≤ day0,
  held flat over the horizon → `0.5·ln(σ²)`.
- **HAR-RV** — Corsi log-HAR: OLS of `v_post` on `ln(RV_daily/weekly/monthly)`,
  **fit on the training split only** (the train-only fit deferred from T1.3), then
  applied to every call.
- **GARCH(1,1)** — `arch` fit per call on the ticker's returns ≤ day0; the mean
  per-day variance forecast over τ → `0.5·ln(σ̄²)`.

Persistence/EWMA/GARCH are per-call (split-independent); HAR is train-fit. All
take daily *simple* returns matching the §5.3 target convention. Pure functions;
GARCH returns None on non-convergence so the >95% gate can be measured.
"""

from __future__ import annotations

import warnings

import numpy as np

EWMA_LAMBDA = 0.94  # RiskMetrics daily decay
GARCH_MIN_OBS = 100  # too few returns → unstable fit, skip
GARCH_SCALE = 100.0  # fit on percent returns (arch convergence); unscale variance by SCALE²
_LOG_EPS = 1e-12


def log_rv_from_variance(variance: float) -> float:
    """Convert a return-variance forecast to the log-RV target scale: 0.5·ln(var)."""
    if not np.isfinite(variance) or variance <= 0.0:
        return float("nan")
    return 0.5 * float(np.log(variance))


# --- returns ----------------------------------------------------------------


def returns_through(close: dict[str, float], as_of: str) -> np.ndarray:
    """Daily simple returns over sessions up to and including `as_of` (ISO date).

    `close` is {ISO date → adjusted close} (from `prices.load_close_series`); the
    information rule (§5.4) — never read past `as_of` — is enforced by the slice.
    """
    items = sorted((d, p) for d, p in close.items() if d <= as_of)
    if len(items) < 2:
        return np.array([])
    prices = np.array([p for _, p in items], dtype=float)
    return np.diff(prices) / prices[:-1]


# --- EWMA -------------------------------------------------------------------


def ewma_variance(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """RiskMetrics EWMA conditional variance (forecast for the next period).

    `σ²_{t} = λ·σ²_{t-1} + (1−λ)·r²_{t-1}`, initialised at the sample variance.
    Fixed point for constant |r|: σ² → r².
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float("nan")
    var = float(np.var(r))
    for x in r:
        var = lam * var + (1.0 - lam) * x * x
    return var


def ewma_log_rv(close: dict[str, float], as_of: str, lam: float = EWMA_LAMBDA) -> float:
    """EWMA forecast of `v_post` (flat over the horizon)."""
    return log_rv_from_variance(ewma_variance(returns_through(close, as_of), lam))


# --- GARCH(1,1) -------------------------------------------------------------


def garch_log_rv_multi(
    close: dict[str, float], as_of: str, horizons: tuple[int, ...]
) -> dict[int, float] | None:
    """GARCH(1,1) `v_post(τ)` forecast for every τ from a single fit.

    Returns {τ → log-RV forecast}, or None if too little history / no convergence
    (so the >95%-convergence gate is measurable). Fit on percent returns ≤ day0;
    for each τ the forecast is the mean of the per-day variances over steps 1..τ.
    """
    from arch import arch_model

    r = returns_through(close, as_of)
    if r.size < GARCH_MIN_OBS:
        return None
    maxh = max(horizons)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r * GARCH_SCALE, mean="Constant", vol="GARCH", p=1, q=1).fit(
                disp="off"
            )
        if getattr(res, "convergence_flag", 0) != 0:
            return None
        step_var = res.forecast(horizon=maxh, reindex=False).variance.values[-1]
    except Exception:
        return None
    return {
        h: log_rv_from_variance(float(np.mean(step_var[:h])) / (GARCH_SCALE**2)) for h in horizons
    }


def garch_log_rv(close: dict[str, float], as_of: str, horizon: int) -> float | None:
    """Single-horizon GARCH(1,1) `v_post(τ)` forecast (see `garch_log_rv_multi`)."""
    out = garch_log_rv_multi(close, as_of, (horizon,))
    return None if out is None else out[horizon]


# --- HAR-RV (train-fit) -----------------------------------------------------


def har_design(rv_daily: np.ndarray, rv_weekly: np.ndarray, rv_monthly: np.ndarray) -> np.ndarray:
    """Design matrix [1, ln RV_d, ln RV_w, ln RV_m] for the Corsi log-HAR."""
    d = np.log(np.asarray(rv_daily, dtype=float) + _LOG_EPS)
    w = np.log(np.asarray(rv_weekly, dtype=float) + _LOG_EPS)
    m = np.log(np.asarray(rv_monthly, dtype=float) + _LOG_EPS)
    return np.column_stack([np.ones_like(d), d, w, m])


def har_fit(x_design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients (numpy least squares), dropping rows with any NaN."""
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x_design).any(axis=1) | np.isnan(y))
    coef, *_ = np.linalg.lstsq(x_design[keep], y[keep], rcond=None)
    return coef


def har_predict(x_design: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return x_design @ coef
