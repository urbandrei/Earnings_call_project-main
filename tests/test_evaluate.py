"""T2.2 evaluation orchestration: the sanity gate logic + a small end-to-end run."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.data.prices import write_price_parquet
from ecvol.eval import evaluate as E

# --- the corroborated sanity gate (DESIGN §6 Stage 0) ------------------------


def _gate_table(fincall_temporal, fincall_disjoint, maec_temporal) -> pd.DataFrame:
    """A table carrying just the HAR level-v τ=30 test cells the gate reads."""
    rows = [
        ("fincall", "temporal", fincall_temporal),
        ("fincall", "ticker_disjoint", fincall_disjoint),
        ("maec", "temporal", maec_temporal),
    ]
    return pd.DataFrame(
        [
            {
                "dataset": ds,
                "convention": "trading",
                "split": sp,
                "target": "v",
                "horizon": 30,
                "model": "har",
                "segment": "test",
                "r2_oos": r2,
            }
            for ds, sp, r2 in rows
        ]
    )


def test_gate_literal_pass():
    g = E._check_gate(_gate_table(0.15, 0.2, 0.2))
    assert g["passed"] and g["literal_pass"] and not g["covid_regime_exception"]


def test_gate_covid_regime_exception():
    # FinCall temporal fails, but regime-stable cells corroborate sound targets
    g = E._check_gate(_gate_table(-0.29, 0.23, 0.21))
    assert g["passed"] and not g["literal_pass"] and g["covid_regime_exception"]


def test_gate_genuine_failure():
    # FinCall temporal fails AND the corroborating cells fail → real problem
    g = E._check_gate(_gate_table(-0.29, -0.1, 0.21))
    assert not g["passed"] and not g["covid_regime_exception"]


# --- small end-to-end --------------------------------------------------------


def _build_fixture(root: Path) -> None:
    from ecvol.data import calendar as cal

    sessions = [d.isoformat() for d in cal.sessions_in_range(date(2019, 1, 2), date(2021, 6, 30))]
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rng = np.random.default_rng(0)

    # prices: ~600 sessions each so EWMA/GARCH have history
    for t in tickers:
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.012, len(sessions)))
        rows = [
            {"date": d, "open": p, "high": p, "low": p, "close": p, "volume": 1000}
            for d, p in zip(sessions, prices, strict=True)
        ]
        write_price_parquet(rows, root / "prices" / f"{t}.parquet")

    # 80 calls spread across the calendar and tickers
    call_dates = sessions[200::4][:80]
    trows, crows, srows = [], [], []
    for i, as_of in enumerate(call_dates):
        cid = 1000 + i
        ticker = tickers[i % len(tickers)]
        v_pre = float(rng.normal(-4, 0.3))
        v_post = v_pre + float(rng.normal(0, 0.3))
        crows.append({"call_id": cid, "n_turns": 50 + i, "n_chars": 1000 + 10 * i})
        # temporal split by index order
        seg = "train" if i < 56 else ("val" if i < 64 else "test")
        srows.append({"call_id": cid, "ticker": ticker, "as_of": as_of, "split": seg})
        for h in E.HORIZONS:
            trows.append(
                {
                    "call_id": cid,
                    "ticker": ticker,
                    "call_type": "earnings",
                    "call_date": as_of,
                    "as_of": as_of,
                    "horizon": h,
                    "v_pre": v_pre,
                    "v_post": v_post,
                    "delta_v": v_post - v_pre,
                    "rv_daily": float(abs(rng.normal(1e-4, 1e-5))),
                    "rv_weekly": float(abs(rng.normal(1e-4, 1e-5))),
                    "rv_monthly": float(abs(rng.normal(1e-4, 1e-5))),
                    "assumed_after_hours": True,
                    "status": "ok",
                    "reason": "",
                }
            )
    (root / "fincall").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trows).to_parquet(root / "fincall" / "targets.parquet")
    pd.DataFrame(crows).to_parquet(root / "fincall" / "calls.parquet")
    (root / "splits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(srows).to_csv(root / "splits" / "fincall_temporal.csv", index=False)


def test_end_to_end_run(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "SPLIT_SCHEMES", ("temporal",))
    root = tmp_path / "data"
    _build_fixture(root)

    summary = E.run_evaluate(root, seeds=(0,))
    out = root / "results" / "result_table_1.csv"
    assert out.is_file()
    t = pd.read_csv(out)

    assert set(t["model"]) == {"persistence", "ewma", "har", "garch", "gbdt_tickerFE"}
    assert set(t["segment"]) <= {"val", "test"}
    # persistence is the R²_OOS baseline on level-v ⇒ exactly 0
    pv = t[(t.model == "persistence") & (t.target == "v")]
    assert np.allclose(pv["r2_oos"], 0.0, atol=1e-9)
    # GARCH converged on the long price histories
    assert summary.garch_convergence["fincall"] > 0.95


def test_end_to_end_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "SPLIT_SCHEMES", ("temporal",))
    root = tmp_path / "data"
    _build_fixture(root)
    E.run_evaluate(root, seeds=(0,))
    first = (root / "results" / "result_table_1.csv").read_bytes()
    E.run_evaluate(root, seeds=(0,))
    assert (root / "results" / "result_table_1.csv").read_bytes() == first


def test_calendar_convention_rows_leave_trading_rows_unchanged(tmp_path, monkeypatch):
    """T9.1: with a calendar-day target file present, Result Table 1 gains calendar rows
    and the trading rows are byte-for-byte what a trading-only run produces."""
    monkeypatch.setattr(E, "SPLIT_SCHEMES", ("temporal",))
    root = tmp_path / "data"
    _build_fixture(root)
    E.run_evaluate(root, seeds=(0,))
    trading_only = pd.read_csv(root / "results" / "result_table_1.csv")
    assert set(trading_only["convention"]) == {"trading"}

    # A calendar file: same calls, shorter windows (n_post < τ), perturbed targets.
    cal = pd.read_parquet(root / "fincall" / "targets.parquet")
    cal["n_post"] = (cal["horizon"] * 0.7).round().astype(int).clip(lower=2)
    cal["v_post"] = cal["v_post"] + 0.05
    cal["delta_v"] = cal["v_post"] - cal["v_pre"]
    cal.to_parquet(root / "fincall" / "targets_calendar.parquet")

    summary = E.run_evaluate(root, seeds=(0,))
    both = pd.read_csv(root / "results" / "result_table_1.csv")
    assert set(both["convention"]) == {"trading", "calendar"}
    assert list(both["convention"].drop_duplicates()) == ["trading", "calendar"]  # order
    tr = both[both["convention"] == "trading"].reset_index(drop=True)
    pd.testing.assert_frame_equal(tr, trading_only)
    # the sanity gate reads trading rows only
    assert summary.gate_detail["har_r2_oos_vs_persistence"]["fincall_temporal"] == float(
        trading_only[
            (trading_only.model == "har")
            & (trading_only.target == "v")
            & (trading_only.horizon == 30)
            & (trading_only.segment == "test")
        ]["r2_oos"].iloc[0]
    )
