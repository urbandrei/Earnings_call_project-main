"""T9.2 call-timestamp retrofit: EDGAR release filing selection, the day-0 reading of a
measured timestamp, the provenance tiers, and the table the target builders consume."""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa

from ecvol.data import timing as T
from ecvol.data.calls import CallRecord, write_calls_parquet


def _filings(rows):
    return pd.DataFrame(
        rows, columns=["form", "filingDate", "acceptanceDateTime", "items", "accessionNumber"]
    )


def test_release_filing_picks_earliest_item_202_in_window():
    f = _filings(
        [
            ("8-K", "2021-07-15", "2021-07-15T20:05:00.000Z", "8.01", "a"),  # not 2.02
            ("8-K", "2021-07-21", "2021-07-21T20:17:00.000Z", "2.02,9.01", "b"),  # evening before
            ("8-K", "2021-07-22", "2021-07-22T11:00:00.000Z", "2.02", "c"),  # later duplicate
            ("10-Q", "2021-07-22", "2021-07-22T12:00:00.000Z", "", "d"),
        ]
    )
    hit = T.release_filing(f, date(2021, 7, 22))
    assert hit["accession"] == "b"
    assert hit["anchor_et"] == datetime(2021, 7, 21, 16, 17)  # 20:17Z → 16:17 EDT
    assert T.release_filing(f, date(2021, 9, 1)) is None  # nothing in window
    assert T.release_filing(_filings([]), date(2021, 7, 22)) is None


def test_session_rule_and_anchor_timestamp():
    d = date(2021, 7, 22)
    assert T.session_rule(datetime(2021, 7, 22, 7, 30), d) == "before_open"
    assert T.session_rule(datetime(2021, 7, 22, 9, 30), d) == "intraday"
    assert T.session_rule(datetime(2021, 7, 22, 16, 0), d) == "after_close"
    assert T.session_rule(datetime(2021, 7, 21, 16, 17), d) == "before_open"  # prior evening
    assert T.session_rule(datetime(2021, 7, 23, 8, 0), d) == "after_call_date"
    # anchors on other days collapse onto the call date so anchor_day0 branches correctly
    assert T.anchor_timestamp(datetime(2021, 7, 21, 16, 17), d) == datetime(2021, 7, 22, 0, 0)
    assert T.anchor_timestamp(datetime(2021, 7, 23, 8, 0), d) == datetime(2021, 7, 22, 16, 0)
    assert T.anchor_timestamp(datetime(2021, 7, 22, 10, 0), d) == datetime(2021, 7, 22, 10, 0)


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


def test_build_call_times_tiers_offline(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    write_calls_parquet(
        [
            _call("1", "AAA", "2021-07-22"),
            _call("2", "BBB", "2021-07-22"),
            _call("3", "CCC", "2021-07-22"),
        ],
        root / "x" / "calls.parquet",
        id_type=pa.string(),
    )
    # AAA: a cached EDGAR submissions file with an Item 2.02 8-K the evening before.
    cache = root / T.CACHE_REL
    cache.mkdir(parents=True)
    (cache / "CIK0000000001.json").write_text(
        json.dumps(
            {
                "filings": {
                    "recent": {
                        "form": ["8-K"],
                        "filingDate": ["2021-07-21"],
                        "acceptanceDateTime": ["2021-07-21T20:17:00.000Z"],
                        "items": ["2.02,9.01"],
                        "accessionNumber": ["0001-21-1"],
                    },
                    "files": [],
                }
            }
        ),
        encoding="utf-8",
    )
    # BBB: no EDGAR cache, but a DEC flag.
    dec = root / T.DEC_REL
    dec.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": ["BBB"],
            "day_earnings": ["2021-07-22"],
            "beforeAfterMarket": ["BeforeMarket"],
            "id": ["BBB_2021_Q3"],
        }
    ).to_csv(dec, index=False)
    monkeypatch.setattr(
        T, "_cik_by_ticker", lambda root: {"AAA": "0000000001", "BBB": "0000000002"}
    )

    rows, counts = T.build_call_times(root, "x", fetch=False)  # offline: cache only
    by = {r.call_id: r for r in rows}
    assert by["1"].tier == "edgar_8k" and by["1"].session_rule == "before_open"
    assert by["1"].anchor_et == "2021-07-21T16:17" and by["1"].source_ref == "0001-21-1"
    assert by["2"].tier == "dec_flag" and by["2"].session_rule == "before_open"
    assert by["3"].tier == "assumed_after_hours"
    assert counts == {
        "edgar_8k": 1,
        "earnings25_metadata": 0,
        "dec_flag": 1,
        "assumed_after_hours": 1,
    }

    out = root / "coverage" / "x_timing.csv"
    T.write_call_times(rows, out)
    stamps = T.load_call_times(out)
    assert stamps["1"] == datetime(2021, 7, 22, 0, 0)  # prior-evening release → before-open branch
    assert stamps["2"] == datetime(2021, 7, 22, 0, 0)  # DEC BeforeMarket → 00:00 on the call date
    assert "3" not in stamps  # the fallback tier stays a fallback

    # deterministic
    first = out.read_bytes()
    T.write_call_times(T.build_call_times(root, "x", fetch=False)[0], out)
    assert out.read_bytes() == first
