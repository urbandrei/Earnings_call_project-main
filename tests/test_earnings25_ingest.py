"""T7.1 Earnings25 ingestion: record parsing, universe rule, timestamps, targets, determinism.

Fixtures build a tiny `testset-full/data.jsonl` + a fake S&P 500 page revision in
tmp dirs (no network); the join test adds an Earnings25 price parquet so one call
computes real targets under measured call times.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ecvol.data import earnings25_ingest as E
from ecvol.data.calendar import sessions_in_range
from ecvol.data.prices import write_price_parquet

# --- pure helpers ------------------------------------------------------------


def test_release_to_et_converts_utc():
    # 2025-12-17T23:00Z → 18:00 EST (after the 16:00 close); 2025-12-19T14:30Z → 09:30 EST
    assert E.release_to_et("2025-12-17T23:00:00").isoformat() == "2025-12-17T18:00:00"
    assert E.release_to_et("2025-12-19T14:30:00").isoformat() == "2025-12-19T09:30:00"
    # DST: 2025-10-22T20:30Z → 16:30 EDT
    assert E.release_to_et("2025-10-22T20:30:00").isoformat() == "2025-10-22T16:30:00"


def test_segments_to_turns_merges_speakers_and_tags_operator():
    attributions = {"1": "Operator", "2": "Jane Doe"}
    segs = [
        {"segment_transcript": " Good day,", "speaker_ids": ["1"]},
        {"segment_transcript": "everyone.", "speaker_ids": ["1"]},
        {"segment_transcript": "Thank you.", "speaker_ids": ["2"]},
        {"segment_transcript": "", "speaker_ids": ["2"]},  # empty segments dropped
    ]
    turns = E.segments_to_turns(segs, attributions)
    assert [t["role"] for t in turns] == ["operator", "unknown"]
    assert turns[0]["text"] == "Good day, everyone."
    assert turns[1]["speaker"] == "2"


def test_bitrate_stratum():
    assert E.bitrate_stratum(64.001) == "64k"
    assert E.bitrate_stratum(24.0) == "le24k"
    assert E.bitrate_stratum(16.0) == "le24k"
    assert E.bitrate_stratum(32.0) == "other"


def test_resolve_ticker_overrides_and_lookalikes():
    sec = {"acme": ("ACME", "0000000001")}
    assert E.resolve_ticker("Acme Corp", sec) == ("ACME", "")
    assert E.resolve_ticker("First Solar, Inc.", sec) == ("FSLR", "")  # curated override
    assert E.resolve_ticker("Grainger PLC", sec) == ("", "lookalike")
    assert E.resolve_ticker("Nobody Ltd", sec) == ("", "unresolved")


# --- fixtures ----------------------------------------------------------------

DAY0 = date(2025, 10, 30)  # Thursday; a session


def _wiki_page(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><td>{t}</td><td>{c}</td><td>Industrials</td></tr>" for t, c in rows)
    return (
        '<html><body><table id="constituents"><tr><th>Symbol</th><th>Security</th>'
        f"<th>GICS Sector</th></tr>{body}</table></body></html>"
    )


def _record(cid: str, company: str, release: str, text: str = "Thank you. " * 20) -> dict:
    return {
        "id": cid,
        "audio_file_path": f"audio/{cid}.mp3",
        "audio_info": {"format_name": "mp3", "sample_rate": 16000, "duration_seconds": 3600.0},
        "extra_fields": {
            "Company": company,
            "Country": "US",
            "ReleaseDate": release,
            "Industry": "Widgets",
            "MarketCap": "1000",
            "speaker_attributions": json.dumps({"1": "Operator", "2": "Pat Lee"}),
        },
        "transcript": text,
        "speech_segments": [
            {"start": 0.0, "end": 1.0, "segment_transcript": "Good day.", "speaker_ids": ["1"]},
            {"start": 1.0, "end": 2.0, "segment_transcript": text, "speaker_ids": ["2"]},
        ],
    }


def _seed(root: Path, monkeypatch) -> None:
    ds = root / E.DATASET_REL
    ds.mkdir(parents=True)
    records = [
        _record("1", "Acme Corporation", f"{DAY0.isoformat()}T20:30:00"),  # 16:30 ET → after hours
        _record("2", "Acme Corporation", f"{DAY0.isoformat()}T12:30:00"),  # 08:30 ET → same-day
        _record("3", "Zed Holdings", f"{DAY0.isoformat()}T20:30:00"),  # resolves, not S&P
        _record("4", "Grainger PLC", f"{DAY0.isoformat()}T20:30:00"),  # look-alike
        _record("5", "Nobody Ltd", f"{DAY0.isoformat()}T20:30:00"),  # unresolved
    ]
    records[1]["id"] = "2"
    with (ds / "data.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    # A cached S&P 500 revision for 2025-10 listing ACME only.
    rev = root / E.REVISIONS_REL
    rev.mkdir(parents=True)
    (rev / "wiki_sp500_2025-10_rev1.html").write_text(_wiki_page([("ACME", "Acme")]), "utf-8")
    # SEC table + CIK map, no network.
    sec = {"acme": ("ACME", "0000000001"), "zed": ("ZED", "0000000002")}
    ciks = {"ACME": "0000000001", "ZED": "0000000002"}
    monkeypatch.setattr(E, "load_sec_table", lambda root: (sec, frozenset()))
    monkeypatch.setattr(E, "_cik_by_ticker", lambda root: ciks)
    # Prices for ACME around DAY0 (own store).
    sessions = sessions_in_range(DAY0 - timedelta(days=90), DAY0 + timedelta(days=90))
    prices = [100.0 + (i % 7) * 0.5 for i in range(len(sessions))]
    rows = [
        {"date": d.isoformat(), "open": p, "high": p, "low": p, "close": p, "volume": 1}
        for d, p in zip(sessions, prices, strict=True)
    ]
    write_price_parquet(rows, root / E.PRICES_REL / "ACME.parquet")


# --- end to end --------------------------------------------------------------


def test_ingest_universe_rule_timestamps_and_targets(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    _seed(root, monkeypatch)
    s = E.ingest_earnings25(root, horizons=(3,), probe_audio=False)

    assert s.total_calls == 5 and s.ok == 2
    assert s.reason_counts == {"lookalike": 1, "not_sp500": 1, "unresolved": 1}
    assert s.months == ["2025-10"]
    assert s.joined == 2 and s.join_rate_pct == 100.0

    calls = pq.read_table(root / "earnings25" / "calls.parquet").to_pandas()
    assert set(calls.time_known) == {True}
    assert set(calls.assumed_after_hours) == {False}
    by_id = calls.set_index("call_id")
    assert by_id.loc["1", "status"] == "ok" and by_id.loc["4", "reason"] == "lookalike"
    assert by_id.loc["1", "audio_path"].endswith("testset-full/audio/1.mp3")
    assert not by_id.loc["1", "audio_exists"]  # fixture ships no mp3 → checked on disk
    turns = json.loads(by_id.loc["1", "transcript_json"])
    assert turns[0]["role"] == "operator" and turns[1]["role"] == "unknown"

    # Measured times drive the after-hours rule: 16:30 ET → day0 = call date;
    # 08:30 ET → the call date is day +1, so day0 is the prior session.
    t = pq.read_table(root / "earnings25" / "targets.parquet").to_pandas().set_index("call_id")
    assert t.loc["1", "as_of"] == DAY0.isoformat()
    assert t.loc["2", "as_of"] == (DAY0 - timedelta(days=1)).isoformat()
    assert set(t["assumed_after_hours"]) == {False}
    assert (root / "earnings25" / "targets_calendar.parquet").is_file()

    # Coverage artifacts + manifests.
    cov = root / "coverage"
    times = (cov / "earnings25_call_times.csv").read_text(encoding="utf-8")
    assert "2025-10-30T16:30,earnings25_metadata" in times
    for name in ("earnings25_ingest_report.csv", "earnings25_inventory.csv"):
        assert (cov / name).is_file()
    for name in ("earnings25_raw.json", "earnings25_calls.json", "earnings25_targets.json"):
        assert (root / "manifests" / name).is_file()


def test_ingest_deterministic(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    _seed(root, monkeypatch)
    E.ingest_earnings25(root, horizons=(3,), probe_audio=False)
    first = {
        n: (root / "earnings25" / n).read_bytes()
        for n in ("calls.parquet", "targets.parquet", "targets_calendar.parquet")
    }
    E.ingest_earnings25(root, horizons=(3,), probe_audio=False)
    for n, b in first.items():
        assert (root / "earnings25" / n).read_bytes() == b


def test_missing_dataset_is_an_error(tmp_path: Path):
    with pytest.raises(ValueError):
        E.ingest_earnings25(tmp_path, probe_audio=False)


def test_calls_parquet_string_ids(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    _seed(root, monkeypatch)
    E.ingest_earnings25(root, horizons=(3,), probe_audio=False)
    schema = pq.read_schema(root / "earnings25" / "calls.parquet")
    assert schema.field("call_id").type == pa.string()
