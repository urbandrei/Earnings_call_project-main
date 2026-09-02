"""T7.2 lookahead: frozen imputation and the index-based cluster bootstrap of R²_OOS."""

import numpy as np

from ecvol.eval import lookahead as L
from ecvol.eval.significance import cluster_bootstrap_ci


def test_impute_uses_given_medians_only():
    X = np.array([[1.0, np.nan], [np.nan, 4.0]])
    out = L._impute(X, np.array([10.0, 20.0]))
    assert out.tolist() == [[1.0, 20.0], [10.0, 4.0]]
    assert np.isnan(X).sum() == 2  # input untouched


def test_bootstrap_over_indices_reproduces_r2():
    err = np.array([1.0, 1.0, 1.0, 1.0])
    err_b = np.array([2.0, 2.0, 2.0, 2.0])
    idx = np.arange(4)
    clusters = np.array(["a", "a", "b", "b"])

    def stat(sub, _e=err, _b=err_b):
        sub = sub.astype(int)
        return 1.0 - _e[sub].sum() / _b[sub].sum()

    point, lo, hi = cluster_bootstrap_ci(idx, clusters, statistic=stat, n_resamples=50, seed=0)
    assert point == 0.5 and lo == 0.5 and hi == 0.5  # identical clusters ⇒ degenerate CI
    assert set(L.STAGES) >= {"persistence", "har", "ridge_fusion_pastvol"}


def test_load_eval_frame_anchor_selects_measured_files(tmp_path):
    import pandas as pd

    from ecvol.eval import evaluate as E

    d = tmp_path / "x"
    d.mkdir()
    base = {"horizon": [3], "status": ["ok"], "n_post": [3]}
    pd.DataFrame({"call_id": ["a"], **base}).to_parquet(d / "targets.parquet")
    pd.DataFrame({"call_id": ["b"], **base}).to_parquet(d / "targets_measured.parquet")
    pd.DataFrame({"call_id": ["a", "b"], "n_turns": [1, 1], "n_chars": [1, 1]}).to_parquet(
        d / "calls.parquet"
    )
    assert E.load_eval_frame(tmp_path, "x")["call_id"].tolist() == ["a"]
    assert E.load_eval_frame(tmp_path, "x", anchor="measured")["call_id"].tolist() == ["b"]
    assert set(L.ANCHOR_FILES) == {"assumed", "measured"}
