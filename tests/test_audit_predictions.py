"""`ecvol audit predictions`: input contract, checks on a frame with a known structure."""

import numpy as np
import pandas as pd
import pytest

from ecvol.eval import audit_predictions as A


def _frame(seed=0, identity_model=True):
    """8 tickers × 10 quarterly calls (6 train, 4 test); T6/T7 appear only in test. A ticker-level
    target (per-ticker offset + noise) and a model that either predicts the ticker offset
    (identity_model) or the true value with independent noise (content model)."""
    rng = np.random.default_rng(seed)
    rows = []
    offsets = {f"T{i}": float(v) for i, v in enumerate(rng.normal(0, 1, 8))}
    dates = pd.bdate_range("2019-01-01", periods=10 * 70, freq="B")[::70]
    y_all = {(t, i): off + rng.normal(0, 0.3) for t, off in offsets.items() for i in range(10)}
    seen = {t: np.mean([y_all[t, i] for i in range(6)]) for t in offsets if t not in ("T6", "T7")}
    global_mean = float(np.mean(list(seen.values())))
    for i, d in enumerate(dates):
        split = "train" if i < 6 else "test"
        for t in offsets:
            if split == "train" and t in ("T6", "T7"):
                continue
            y = y_all[t, i]
            # identity model = what a model that memorised the company would output
            pred = (
                seen.get(t, global_mean) + rng.normal(0, 0.02)
                if identity_model
                else y + rng.normal(0, 0.2)
            )
            rows.append(
                {"call_id": f"{t}_{d:%Y%m%d}", "ticker": t, "date": f"{d:%Y-%m-%d}",
                 "split": split, "horizon": 3, "y_true": y, "y_pred": pred,
                 "y_persistence": offsets[t] + rng.normal(0, 0.4)}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_load_frame_enforces_the_contract(tmp_path):
    f = _frame()
    p = tmp_path / "ok.csv"
    f.to_csv(p, index=False)
    assert len(A.load_frame(p)) == len(f)
    for bad, msg in (
        (f.drop(columns="ticker"), "missing columns"),
        (pd.concat([f, f.iloc[:1]]), "duplicate"),
        (f.assign(split=f["split"].replace("test", "eval")), "unknown split"),
        (f.assign(date=f["date"].str.replace("-", "/")), "YYYY-MM-DD"),
    ):
        bad.to_csv(p, index=False)
        with pytest.raises(A.FrameError, match=msg):
            A.load_frame(p)


def test_split_integrity_counts_seen_tickers_and_gap():
    f = _frame()
    rep = A.audit(f)
    g = lambda m: float(rep.loc[rep["metric"] == m, "value"].iloc[0])  # noqa: E731
    assert g("test_calls_ticker_in_fit_pct") == pytest.approx(75.0)  # 6 of 8 tickers seen
    assert g("gap_business_days") == 70 and g("fit_windows_reaching_test") == 0
    assert g("n_train") == 36 and g("n_test") == 32


def test_identity_model_is_caught_and_content_model_is_not():
    rep_id, rep_ct = A.audit(_frame(identity_model=True)), A.audit(_frame(identity_model=False))
    g = lambda rep, m: float(rep.loc[rep["metric"] == m, "value"].iloc[0])  # noqa: E731
    # identity model: same-company swap changes nothing, other-company swap hurts,
    # predictions are (almost) all ticker variance, and it does not beat ticker-only
    assert abs(g(rep_id, "within_swap_change_pct")) < 5 < g(rep_id, "other_swap_change_pct")
    assert g(rep_id, "ticker_variance_share_y_pred") > 0.9
    assert abs(g(rep_id, "r2_oos_vs_ticker_only")) < 0.05
    # content model: swapping predictions within a company costs a lot, and it beats ticker-only
    assert g(rep_ct, "within_swap_change_pct") > 20
    assert g(rep_ct, "r2_oos_vs_ticker_only") > 0.3 and g(rep_ct, "dm_p_vs_ticker_only") < 0.05
    assert g(rep_ct, "mse_unseen_ticker") < g(rep_id, "mse_unseen_ticker")


def test_shuffle_never_keeps_own_ticker_globally_and_moves_only_multi_call_tickers():
    tickers = np.array(["A", "A", "B", "C", "C", "C"])
    pred = np.arange(6, dtype=float)
    rng = np.random.default_rng(1)
    w = A.shuffle_predictions(pred, tickers, "within", rng)
    assert w[2] == 2 and set(w[:2]) == {0, 1} and set(w[3:]) == {3, 4, 5}
    g = A.shuffle_predictions(pred, tickers, "global", rng)
    assert all(tickers[int(v)] != t for v, t in zip(g, tickers, strict=True))


def test_run_audit_writes_csv_and_markdown(tmp_path):
    p = tmp_path / "preds.csv"
    _frame().to_csv(p, index=False)
    out = A.run_audit(p, tmp_path / "out")
    rep = pd.read_csv(out / "report.csv")
    md = (out / "report.md").read_text(encoding="utf-8")
    assert {
        "split_integrity",
        "floors",
        "seen_vs_unseen",
        "prediction_shuffle",
        "identity_share",
        "significance",
    } <= set(rep["check"])
    assert "τ=3" in md and "ticker-only" in md
