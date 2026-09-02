"""T1.3/T9.1 target computation: RV math, day-0 rule, both horizon conventions, exclusions.

Numeric checks use zero-mean return windows so the expected log-RV has the closed
form `0.5·ln(mean(r²))` — an independent check, not a re-run of `realized_vol`.
"""

import math
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq

from ecvol.data.prices import write_price_parquet
from ecvol.data.targets import (
    anchor_day0,
    build_targets,
    calendar_window,
    compute_call_targets,
    convention_delta,
    realized_vol,
    session_offsets,
)

# A clean calendar neighbourhood: Fri 2021-01-15 is a session; 16/17 are the
# weekend, Mon 2021-01-18 is MLK (holiday), Tue 2021-01-19 trades.
DAY0 = date(2021, 1, 15)


# --- pure RV math (analytically known) ---------------------------------------


def test_realized_vol_zero_mean_closed_form():
    # mean 0 ⇒ var = mean(r²); v = ln(sqrt(var)) = 0.5·ln(mean r²)
    assert math.isclose(realized_vol([0.2, -0.1, -0.1]), 0.5 * math.log(0.02), abs_tol=1e-12)
    assert math.isclose(realized_vol([0.1, -0.1, 0.1, -0.1]), 0.5 * math.log(0.01), abs_tol=1e-12)


def test_realized_vol_degenerate_cases():
    # A flat price window yields returns of exactly 0.0 → zero variance → ln(0)
    assert math.isnan(realized_vol([0.0, 0.0, 0.0]))
    assert math.isnan(realized_vol([0.1]))  # <2 points
    assert math.isnan(realized_vol([]))


# --- after-hours / day-0 rule ------------------------------------------------


def test_anchor_day0_fallback_rolls_back_to_last_session():
    assert anchor_day0(DAY0) == DAY0  # a session → itself
    assert anchor_day0(date(2021, 1, 16)) == DAY0  # Saturday → Friday
    assert anchor_day0(date(2021, 1, 17)) == DAY0  # Sunday → Friday
    assert anchor_day0(date(2021, 1, 18)) == DAY0  # MLK holiday → Friday


def test_anchor_day0_timestamp_branches():
    # ≥16:00 ET → after-hours: day0 is the call's own session
    assert anchor_day0(DAY0, timestamp=datetime(2021, 1, 15, 16, 0)) == DAY0
    # <09:30 ET → call date is day +1, so day0 is the prior session
    assert anchor_day0(date(2021, 1, 19), timestamp=datetime(2021, 1, 19, 8, 0)) == DAY0


# --- HAR inputs --------------------------------------------------------------


def _close_at(offsets, prices_by_offset):
    return {offsets[o].isoformat(): p for o, p in prices_by_offset.items()}


def test_har_daily_input_and_missing_windows():
    off = session_offsets(DAY0, back=3, fwd=3)
    # r_0 = (97.2-108)/108 = -0.1 → rv_daily = 0.01; only 7 sessions present →
    # weekly (5) / monthly (22) lookbacks lack history → NaN.
    close = _close_at(
        off, {-3: 100.0, -2: 120.0, -1: 108.0, 0: 97.2, 1: 106.92, 2: 101.574, 3: 96.4953}
    )
    rows = compute_call_targets(
        {"call_id": 1, "ticker": "AAA", "date": DAY0.isoformat(), "call_type": "earnings"},
        close,
        horizons=(3,),
    )
    r = rows[0]
    assert math.isclose(r.rv_daily, 0.01, abs_tol=1e-9)
    assert math.isnan(r.rv_weekly) and math.isnan(r.rv_monthly)


# --- 3 calls hand-verified end-to-end ----------------------------------------


def test_handverified_call_values():
    # Call A: ok row; pre returns [0.2,-0.1,-0.1] (var 0.02), post [0.1,-0.05,-0.05] (var 0.005)
    off = session_offsets(DAY0, back=3, fwd=3)
    close = _close_at(
        off, {-3: 100.0, -2: 120.0, -1: 108.0, 0: 97.2, 1: 106.92, 2: 101.574, 3: 96.4953}
    )
    r = compute_call_targets(
        {"call_id": 10, "ticker": "AAA", "date": DAY0.isoformat(), "call_type": "earnings"},
        close,
        horizons=(3,),
    )[0]
    assert r.status == "ok" and r.reason == "" and r.assumed_after_hours is True
    assert r.as_of == DAY0.isoformat()
    assert math.isclose(r.v_pre, 0.5 * math.log(0.02), abs_tol=1e-6)
    assert math.isclose(r.v_post, 0.5 * math.log(0.005), abs_tol=1e-6)
    assert math.isclose(r.delta_v, 0.5 * math.log(0.25), abs_tol=1e-6)  # ln(0.005/0.02)/2


def test_handverified_weekend_call_anchors_to_friday():
    """Call B: a Saturday call uses the Friday-anchored windows (same prices, same targets)."""
    off = session_offsets(DAY0, back=3, fwd=3)
    close = _close_at(
        off, {-3: 100.0, -2: 120.0, -1: 108.0, 0: 97.2, 1: 106.92, 2: 101.574, 3: 96.4953}
    )
    r = compute_call_targets(
        {"call_id": 11, "ticker": "AAA", "date": "2021-01-16", "call_type": "earnings"},
        close,
        horizons=(3,),
    )[0]
    assert r.as_of == DAY0.isoformat()  # Saturday → Friday day0
    assert r.status == "ok"
    assert math.isclose(r.v_post, 0.5 * math.log(0.005), abs_tol=1e-6)


def test_handverified_call_excluded_insufficient_post():
    """Call C: prices stop at offset +1, so a τ=3 post-window can't form."""
    off = session_offsets(DAY0, back=3, fwd=3)
    close = _close_at(off, {-3: 100.0, -2: 120.0, -1: 108.0, 0: 97.2, 1: 106.92})
    r = compute_call_targets(
        {"call_id": 12, "ticker": "AAA", "date": DAY0.isoformat(), "call_type": "earnings"},
        close,
        horizons=(3,),
    )[0]
    assert r.status == "excluded" and r.reason == "insufficient_post_history"
    assert math.isnan(r.v_post) and math.isnan(r.delta_v)
    assert math.isclose(r.v_pre, 0.5 * math.log(0.02), abs_tol=1e-6)  # pre still recorded


# --- exclusion reason codes --------------------------------------------------


def test_exclusion_reason_codes_and_one_row_per_horizon():
    off = session_offsets(DAY0, back=3, fwd=3)
    full = _close_at(
        off, {-3: 100.0, -2: 120.0, -1: 108.0, 0: 97.2, 1: 106.92, 2: 101.574, 3: 96.4953}
    )
    horizons = (3, 7)

    # unresolved ticker → all horizons excluded, no day0 needed
    unresolved = compute_call_targets(
        {"call_id": 1, "ticker": "", "date": DAY0.isoformat(), "call_type": "earnings"},
        full,
        horizons=horizons,
    )
    assert [r.reason for r in unresolved] == ["unresolved_ticker", "unresolved_ticker"]
    assert len(unresolved) == len(horizons)  # one row per (call, τ)

    # resolved ticker but no price data → as_of still stamped
    nodata = compute_call_targets(
        {"call_id": 2, "ticker": "AAA", "date": DAY0.isoformat(), "call_type": "earnings"},
        {},
        horizons=horizons,
    )
    assert {r.reason for r in nodata} == {"no_price_data"}
    assert all(r.as_of == DAY0.isoformat() for r in nodata)

    # invalid date
    bad = compute_call_targets(
        {"call_id": 3, "ticker": "AAA", "date": "not-a-date", "call_type": "earnings"},
        full,
        horizons=horizons,
    )
    assert {r.reason for r in bad} == {"invalid_date"}

    # flat post window → zero_variance_post (pre varies)
    flat_post = _close_at(
        off, {-3: 100.0, -2: 120.0, -1: 108.0, 0: 97.2, 1: 97.2, 2: 97.2, 3: 97.2}
    )
    zv = compute_call_targets(
        {"call_id": 4, "ticker": "AAA", "date": DAY0.isoformat(), "call_type": "earnings"},
        flat_post,
        horizons=(3,),
    )[0]
    assert zv.status == "excluded" and zv.reason == "zero_variance_post"


# --- orchestration + determinism ---------------------------------------------


def _seed_root(tmp_path: Path) -> Path:
    (tmp_path / "identity").mkdir(parents=True)
    (tmp_path / "identity" / "fincall_identity.csv").write_text(
        "call_id,ticker,date,call_type\n"
        f"100,AAA,{DAY0.isoformat()},earnings\n"
        f"101,BBB,{DAY0.isoformat()},earnings\n"  # BBB has no parquet → no_price_data
        f"102,,{DAY0.isoformat()},conference\n",  # unresolved
        encoding="utf-8",
    )
    off = session_offsets(DAY0, back=30, fwd=30)
    rows = [
        {"date": off[o].isoformat(), "open": p, "high": p, "low": p, "close": p, "volume": 1}
        for o, p in ((o, 100.0 + o) for o in sorted(off))
    ]
    write_price_parquet(rows, tmp_path / "prices" / "AAA.parquet")
    return tmp_path


def test_build_targets_summary_and_determinism(tmp_path):
    root = _seed_root(tmp_path)
    summaries = build_targets(root, horizons=(3, 7))
    summary = summaries["trading"]

    assert set(summaries) == {"trading", "calendar"}
    assert summary.total_calls == 3
    assert summary.resolved_calls == 2  # AAA, BBB (102 unresolved)
    assert summary.rows_total == 3 * 2  # one row per (call, τ)
    assert (root / "fincall" / "targets.parquet").is_file()
    assert (root / "coverage" / "targets_report.csv").is_file()
    assert (root / "manifests" / "fincall_targets.json").is_file()
    # AAA is a clean ascending series → both horizons ok; BBB has no prices
    assert summary.reason_counts.get("no_price_data") == 2
    assert summary.calls_with_any_ok == 1

    # deterministic: same inputs → byte-identical parquet (both conventions)
    first = (root / "fincall" / "targets.parquet").read_bytes()
    first_cal = (root / "fincall" / "targets_calendar.parquet").read_bytes()
    build_targets(root, horizons=(3, 7))
    assert (root / "fincall" / "targets.parquet").read_bytes() == first
    assert (root / "fincall" / "targets_calendar.parquet").read_bytes() == first_cal

    # each file is stamped with its convention; the manifest lists both
    cal = pq.read_table(root / "fincall" / "targets_calendar.parquet")
    assert set(cal.column("convention").to_pylist()) == {"calendar"}
    assert (root / "coverage" / "targets_calendar_report.csv").is_file()
    manifest = (root / "manifests" / "fincall_targets.json").read_text(encoding="utf-8")
    assert "fincall/targets.parquet" in manifest and "targets_calendar.parquet" in manifest

    # the delta report covers every horizon with both files present
    delta = convention_delta(root, datasets=("fincall",))
    assert list(delta["horizon"]) == [3, 7]
    assert set(delta.columns) >= {"n_ok_both", "corr_v_post", "sessions_calendar_mean"}

    # schema sanity + deterministic ordering (sorted by call_id, then horizon)
    table = pq.read_table(root / "fincall" / "targets.parquet")
    keys = list(
        zip(table.column("call_id").to_pylist(), table.column("horizon").to_pylist(), strict=True)
    )
    assert keys == sorted(keys)
    assert set(table.column_names) >= {"call_id", "horizon", "v_pre", "v_post", "delta_v", "reason"}


def test_insufficient_pre_history_reason(tmp_path):
    # AAA priced only from day0 forward → pre window can't form, post can
    root = tmp_path
    (root / "identity").mkdir(parents=True)
    (root / "identity" / "fincall_identity.csv").write_text(
        f"call_id,ticker,date,call_type\n200,AAA,{DAY0.isoformat()},earnings\n", encoding="utf-8"
    )
    off = session_offsets(DAY0, back=0, fwd=10)
    rows = [
        {
            "date": off[o].isoformat(),
            "open": 100.0 + o,
            "high": 100.0 + o,
            "low": 100.0 + o,
            "close": 100.0 + o,
            "volume": 1,
        }
        for o in sorted(off)
        if o >= 0
    ]
    write_price_parquet(rows, root / "prices" / "AAA.parquet")
    summary = build_targets(root, horizons=(3,))["trading"]
    assert summary.reason_counts.get("insufficient_pre_history") == 1


# --- calendar-day convention (T9.1) ------------------------------------------

# Prices by session offset around DAY0 (Fri 2021-01-15). Pre returns over
# offsets -4..0 are [0.1,-0.1,0.1,-0.1,0.0] (zero-mean, Σr²=0.04); post returns
# over offsets 1..4 are [0.2,-0.2,0.1,-0.1] (zero-mean, Σr²=0.10).
CAL_PRICES = {
    -5: 100.0,
    -4: 110.0,
    -3: 99.0,
    -2: 108.9,
    -1: 98.01,
    0: 98.01,
    1: 117.612,
    2: 94.0896,
    3: 103.49856,
    4: 93.148704,
}


def test_calendar_window_bounds_hand_counted():
    """Sessions inside τ calendar days of Fri 2021-01-15 (MLK Monday is closed)."""
    off = session_offsets(DAY0, back=30, fwd=30)
    # post: Sat/Sun/MLK → none; 7d → Tue19..Fri22; 15d → +Jan25..29; 30d → through Feb 12
    assert calendar_window(off, DAY0, 3, post=True) == (1, 0)
    assert calendar_window(off, DAY0, 7, post=True) == (1, 4)
    assert calendar_window(off, DAY0, 15, post=True) == (1, 9)
    assert calendar_window(off, DAY0, 30, post=True) == (1, 19)
    # pre (window ends at day 0 inclusive): 3d → Wed13..Fri15; 7d → Mon11..; 15d →
    # Jan 4.. (Jan 1 closed); 30d → Dec 17.. (Dec 25 closed)
    assert calendar_window(off, DAY0, 3, post=False) == (-2, 0)
    assert calendar_window(off, DAY0, 7, post=False) == (-4, 0)
    assert calendar_window(off, DAY0, 15, post=False) == (-9, 0)
    assert calendar_window(off, DAY0, 30, post=False) == (-19, 0)


def test_handverified_calendar_call_values():
    """τ=7 calendar days: 5 pre sessions / 4 post sessions, closed-form log-RV."""
    off = session_offsets(DAY0, back=7, fwd=7)
    close = _close_at(off, CAL_PRICES)
    call = {"call_id": 20, "ticker": "AAA", "date": DAY0.isoformat(), "call_type": "earnings"}
    r = compute_call_targets(call, close, horizons=(7,), convention="calendar")[0]
    assert r.status == "ok" and r.convention == "calendar"
    assert (r.n_pre, r.n_post) == (5, 4)
    assert math.isclose(r.v_pre, 0.5 * math.log(0.04 / 5), abs_tol=1e-9)
    assert math.isclose(r.v_post, 0.5 * math.log(0.10 / 4), abs_tol=1e-9)
    assert math.isclose(r.delta_v, 0.5 * math.log((0.10 / 4) / (0.04 / 5)), abs_tol=1e-9)
    # the HAR inputs are session-based and identical under both conventions
    t = compute_call_targets(call, close, horizons=(3,), convention="trading")[0]
    assert (t.rv_daily, t.rv_weekly) == (r.rv_daily, r.rv_weekly)
    assert t.convention == "trading" and (t.n_pre, t.n_post) == (3, 3)
    assert not math.isclose(t.v_post, r.v_post)  # a different window ⇒ a different target


def test_calendar_short_and_missing_windows():
    off = session_offsets(DAY0, back=7, fwd=7)
    close = _close_at(off, CAL_PRICES)
    call = {"call_id": 21, "ticker": "AAA", "date": DAY0.isoformat(), "call_type": "earnings"}
    rows = compute_call_targets(call, close, horizons=(3, 15), convention="calendar")
    # τ=3 over a holiday weekend: zero post sessions → excluded, not silently NaN
    assert rows[0].status == "excluded" and rows[0].reason == "short_window_post"
    assert rows[0].n_post == 0 and rows[0].n_pre == 3
    # τ=15: the post window (9 sessions) runs past the last price → insufficient history
    assert rows[1].reason == "insufficient_post_history"


def test_unknown_convention_rejected():
    import pytest

    call = {"call_id": 1, "ticker": "AAA", "date": "2021-01-15"}
    with pytest.raises(ValueError):
        compute_call_targets(call, {}, convention="x")
