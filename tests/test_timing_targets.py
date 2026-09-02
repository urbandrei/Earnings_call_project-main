"""T9.2: measured-anchor target variant and the fallback-vs-measured sensitivity table."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.data import timing as T
from ecvol.data.calendar import sessions_in_range
from ecvol.data.calls import CallRecord, write_calls_parquet
from ecvol.data.prices import write_price_parquet
from ecvol.data.targets import compute_call_targets, write_targets_parquet

DAY0 = date(2021, 1, 15)  # Friday session


def _call(cid, ticker, day):
    return CallRecord(
        call_id=cid,
        source="x",
        ticker=ticker,
        call_date=day,
        time_known=False,
        assumed_after_hours=True,
        call_type="earnings",
        label=-1,
        n_turns=1,
        n_chars=100,
        transcript_json="[]",
        speaker_metadata="{}",
        audio_path="",
        audio_exists=False,
        audio_duration_sec=float("nan"),
        parsed=True,
        status="ok",
        reason="",
    )


def _seed(root: Path) -> None:
    sessions = sessions_in_range(DAY0 - timedelta(days=120), DAY0 + timedelta(days=120))
    prices = [100.0 + (i % 5) * 0.7 + (i % 3) * 0.2 for i in range(len(sessions))]
    rows = [
        {"date": d.isoformat(), "open": p, "high": p, "low": p, "close": p, "volume": 1}
        for d, p in zip(sessions, prices, strict=True)
    ]
    write_price_parquet(rows, root / "prices" / "AAA.parquet")
    calls = [_call("1", "AAA", DAY0.isoformat()), _call("2", "AAA", DAY0.isoformat())]
    write_calls_parquet(calls, root / "maec" / "calls.parquet", id_type=pa.string())
    # primary (fallback) targets for both calls
    trows = []
    for c in calls:
        close = {r["date"]: r["close"] for r in rows}
        trows += compute_call_targets(
            {"call_id": c.call_id, "ticker": "AAA", "date": c.call_date, "call_type": "earnings"},
            close,
            horizons=(3, 30),
        )
    write_targets_parquet(trows, root / "maec" / "targets.parquet", id_type=pa.string())
    # timing: call 1 released before the open → day 0 is the prior session; call 2 fallback
    (root / "coverage").mkdir(parents=True, exist_ok=True)
    T.write_call_times(
        [
            T.TimingRow(
                "1", "AAA", DAY0.isoformat(), "edgar_8k", "2021-01-15T07:30", "before_open", "acc"
            ),
            T.TimingRow(
                "2", "AAA", DAY0.isoformat(), "assumed_after_hours", "", "assumed_after_hours", ""
            ),
        ],
        root / "coverage" / "maec_timing.csv",
    )
    # a ticker-disjoint split with both calls in test and a train set of the same calls'
    # siblings is impossible with 2 calls; give the sensitivity a train set via extra calls
    (root / "splits").mkdir(exist_ok=True)


def test_measured_targets_move_day0_only_for_measured_calls(tmp_path: Path):
    root = tmp_path / "data"
    _seed(root)
    s = T.build_measured_targets(root, "maec", horizons=(3, 30))
    assert s == {"calls": 2, "measured": 1, "ok_rows": 4}
    m = pq.read_table(root / "maec" / "targets_measured.parquet").to_pandas()
    p = pq.read_table(root / "maec" / "targets.parquet").to_pandas()
    prior = sessions_in_range(DAY0 - timedelta(days=5), DAY0 - timedelta(days=1))[-1]
    assert set(m[m.call_id == "1"].as_of) == {prior.isoformat()}  # moved one session back
    assert set(m[m.call_id == "2"].as_of) == {DAY0.isoformat()}  # fallback unchanged
    assert set(p.as_of) == {DAY0.isoformat()}
    assert bool(m[m.call_id == "1"].assumed_after_hours.iloc[0]) is False
    assert (root / "maec" / "targets_measured_calendar.parquet").is_file()
    assert (root / "manifests" / "maec_targets_measured.json").is_file()
    # deterministic
    b = (root / "maec" / "targets_measured.parquet").read_bytes()
    T.build_measured_targets(root, "maec", horizons=(3, 30))
    assert (root / "maec" / "targets_measured.parquet").read_bytes() == b


def test_sensitivity_reports_shift_share_and_baselines(tmp_path: Path):
    root = tmp_path / "data"
    _seed(root)
    T.build_measured_targets(root, "maec", horizons=(3, 30))
    # a split assigning both calls to test; no train rows → baseline rows skipped, shift kept
    pd.DataFrame({"call_id": ["1", "2"], "split": ["test", "test"]}).to_csv(
        root / "splits" / "maec_ticker_disjoint.csv", index=False
    )
    out = T.sensitivity(root, datasets=("maec",))
    shift = out[out.anchor == "shift"].iloc[0]
    assert shift.n == 2 and abs(shift.r2_oos - 0.5) < 1e-9  # one of two calls moved
    assert (root / "results" / "timing_sensitivity.csv").is_file()
