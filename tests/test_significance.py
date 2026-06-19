"""T2.1 significance: DM, cluster bootstrap, Holm (DESIGN §7.2).

Validation strategy (acceptance test):
- DM: the HLN-corrected statistic at h=1 equals the paired t-test on the loss
  differential — an independent published-reference check (scipy.stats.ttest_rel).
- bootstrap: on i.i.d. data (each obs its own cluster) the CI matches the
  analytic normal CI; clustering correlated obs widens it.
- Holm: matches the textbook / R `p.adjust(method="holm")` worked example.
"""

import math

import numpy as np
from scipy import stats

from ecvol.eval.significance import cluster_bootstrap_ci, diebold_mariano, holm_correction

# --- Diebold–Mariano ---------------------------------------------------------


def test_dm_h1_equals_paired_t_on_loss_diff():
    rng = np.random.default_rng(0)
    n = 200
    ea = rng.normal(0, 1.0, n)  # model A residuals
    eb = rng.normal(0, 1.3, n)  # model B residuals (noisier)
    res = diebold_mariano(ea, eb, loss="squared", h=1)

    # Reference: paired t-test on the squared-error differential.
    t_ref = stats.ttest_rel(ea**2, eb**2)
    assert math.isclose(res.statistic, t_ref.statistic, rel_tol=1e-9)
    assert math.isclose(res.p_value, t_ref.pvalue, rel_tol=1e-9)
    assert res.n == n


def test_dm_sign_convention_a_worse_is_positive():
    # A has systematically larger errors → A worse → positive statistic, small p.
    ea = np.array([2.0, -2.0] * 50)
    eb = np.array([0.1, -0.1] * 50)
    res = diebold_mariano(ea, eb)
    assert res.statistic > 0 and res.p_value < 0.05
    # symmetric: swapping flips the sign, same magnitude
    swapped = diebold_mariano(eb, ea)
    assert math.isclose(swapped.statistic, -res.statistic, rel_tol=1e-9)


def test_dm_identical_models_is_nan():
    e = np.array([1.0, -1.0, 0.5, -0.5])
    res = diebold_mariano(e, e)
    assert math.isnan(res.statistic) and math.isnan(res.p_value)


def test_dm_h_greater_than_one_runs_and_uses_autocovariance():
    rng = np.random.default_rng(1)
    ea = rng.normal(0, 1, 100)
    eb = rng.normal(0, 1, 100)
    r1 = diebold_mariano(ea, eb, h=1)
    r5 = diebold_mariano(ea, eb, h=5)
    assert math.isfinite(r5.statistic)
    # different long-run variance / HLN factor → generally a different statistic
    assert r1.statistic != r5.statistic


# --- cluster bootstrap -------------------------------------------------------


def test_bootstrap_matches_analytic_normal_ci_when_iid():
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(5.0, 2.0, n)
    clusters = np.arange(n)  # each obs its own cluster ⇒ ordinary bootstrap
    point, lo, hi = cluster_bootstrap_ci(x, clusters, n_resamples=2000, seed=0)
    se = x.std(ddof=1) / math.sqrt(n)
    assert math.isclose(point, x.mean(), rel_tol=1e-12)
    # half-width ≈ 1.96·SE (bootstrap ≈ analytic for a mean of many iid points)
    half = (hi - lo) / 2
    assert math.isclose(half, 1.96 * se, rel_tol=0.15)
    # the CI brackets the point estimate, and the point is within a few SE of truth
    assert lo < point < hi
    assert abs(point - 5.0) < 5 * se


def test_clustering_widens_ci_for_correlated_data():
    rng = np.random.default_rng(7)
    # 20 clusters of 50 identical-within values → strong intra-cluster correlation
    cluster_means = rng.normal(0, 1, 20)
    values = np.repeat(cluster_means, 50)
    clusters = np.repeat(np.arange(20), 50)

    _, lo_c, hi_c = cluster_bootstrap_ci(values, clusters, n_resamples=2000, seed=0)
    _, lo_i, hi_i = cluster_bootstrap_ci(values, np.arange(values.size), n_resamples=2000, seed=0)
    assert (hi_c - lo_c) > 3 * (hi_i - lo_i)  # ignoring clusters badly understates


def test_bootstrap_deterministic_under_seed():
    x = np.arange(100.0)
    c = np.arange(100)
    assert cluster_bootstrap_ci(x, c, seed=0) == cluster_bootstrap_ci(x, c, seed=0)
    assert cluster_bootstrap_ci(x, c, seed=0) != cluster_bootstrap_ci(x, c, seed=1)


# --- Holm --------------------------------------------------------------------


def test_holm_matches_r_p_adjust_example():
    # R: p.adjust(c(0.01, 0.04, 0.03, 0.005), "holm") == c(0.03, 0.06, 0.06, 0.02)
    adj = holm_correction([0.01, 0.04, 0.03, 0.005])
    assert [round(a, 10) for a in adj] == [0.03, 0.06, 0.06, 0.02]


def test_holm_clips_at_one_and_handles_singletons():
    assert holm_correction([0.6, 0.7, 0.8]) == [1.0, 1.0, 1.0]  # 3·0.6=1.8→1
    assert holm_correction([0.02]) == [0.02]
    assert holm_correction([]) == []


def test_holm_is_monotonic_in_rank():
    adj = holm_correction([0.001, 0.002, 0.003, 0.5])
    order = np.argsort([0.001, 0.002, 0.003, 0.5])
    ordered_adj = [adj[i] for i in order]
    assert ordered_adj == sorted(ordered_adj)
