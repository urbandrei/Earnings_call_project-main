"""Statistical-significance tools for forecast comparison — DESIGN §7.2.

- **Diebold–Mariano** [R27] on per-call loss differentials vs. a reference model
  (HAR-RV, or the best preceding stage), with the Harvey–Leybourne–Newbold
  small-sample correction and a Student-t reference distribution.
- **Cluster bootstrap CIs** — resample whole clusters (ticker, or calendar
  quarter) with replacement; errors are not i.i.d. across either, so naive CIs
  understate uncertainty.
- **Holm correction** — step-down familywise control for the confirmatory
  comparisons (DESIGN §7.5), applied across horizons.

All functions are pure and seeded where stochastic, so `ecvol report` (T2.3)
regenerates identical numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DMResult:
    statistic: float  # HLN-corrected DM; >0 ⇒ model A has the larger loss (worse)
    p_value: float  # two-sided, Student-t with n−1 df
    n: int
    mean_loss_diff: float  # mean(loss_A − loss_B)


def _losses(errors: np.ndarray, loss: str) -> np.ndarray:
    if loss == "squared":
        return errors**2
    if loss == "absolute":
        return np.abs(errors)
    raise ValueError(f"unknown loss {loss!r} (expected 'squared' | 'absolute')")


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    loss: str = "squared",
    h: int = 1,
) -> DMResult:
    """DM test on paired forecast errors (residuals y−ŷ) of models A and B.

    `errors_*` are aligned per-call residuals over the *same* calls; for `h > 1`
    pass them in calendar order (the long-run variance uses the first `h−1`
    autocovariances of the loss differential, as an h-step forecast's errors are
    MA(h−1)). Returns the HLN-corrected statistic (Harvey–Leybourne–Newbold 1997)
    against `t_{n−1}`. A positive statistic with small p means A is significantly
    *worse* than B; negative means A is better.
    """
    ea = np.asarray(errors_a, dtype=float)
    eb = np.asarray(errors_b, dtype=float)
    keep = ~(np.isnan(ea) | np.isnan(eb))
    d = _losses(ea[keep], loss) - _losses(eb[keep], loss)
    n = d.size
    if n < 2:
        return DMResult(float("nan"), float("nan"), n, float("nan"))
    d_bar = float(np.mean(d))
    dev = d - d_bar
    gamma0 = float(np.mean(dev**2))
    # Long-run variance: γ0 + 2 Σ_{k=1..h-1} γ_k (truncated, unweighted — DM/HLN).
    lrv = gamma0
    for k in range(1, h):
        gamma_k = float(np.mean(dev[k:] * dev[:-k]))
        lrv += 2.0 * gamma_k
    if lrv <= 0.0:  # rare with the unweighted estimator; fall back to the variance
        lrv = gamma0
    if lrv <= 0.0:  # d is constant ⇒ no differential variability
        return DMResult(float("nan"), float("nan"), n, d_bar)
    dm = d_bar / np.sqrt(lrv / n)
    # Harvey–Leybourne–Newbold small-sample correction → t_{n-1}.
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2.0 * stats.t.sf(abs(stat), df=n - 1))
    return DMResult(stat, p, n, d_bar)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lo, hi) percentile CI by resampling whole clusters.

    Clusters (e.g. ticker or calendar quarter) are drawn with replacement; the
    statistic is recomputed over the pooled observations of the drawn clusters.
    DESIGN §7.2: errors cluster by ticker and by quarter, so i.i.d. CIs are wrong.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    keep = ~np.isnan(values)
    values, clusters = values[keep], clusters[keep]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    uniq = np.unique(clusters)
    by_cluster = {c: values[clusters == c] for c in uniq}
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = rng.choice(uniq, size=uniq.size, replace=True)
        pooled = np.concatenate([by_cluster[c] for c in drawn])
        estimates[i] = statistic(pooled)
    point = float(statistic(values))
    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1 - alpha / 2))
    return point, lo, hi


def holm_correction(pvalues: list[float] | np.ndarray) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values, in the input order.

    Adjusted p preserves the original ordering and is non-decreasing in rank;
    reject H_i at level α iff adjusted[i] ≤ α (DESIGN §7.5 confirmatory claims).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)  # enforce monotonic non-decreasing
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
