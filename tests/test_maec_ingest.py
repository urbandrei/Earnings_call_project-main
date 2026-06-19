"""T1.5 MAEC ingestion: folder parsing, schema records, audio-feature audit, join.

Fixtures build a tiny MAEC_Dataset tree in tmp dirs; the join test adds a price
parquet so one ticker computes real targets and the other is reason-coded.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.data.maec_ingest import (
    build_records,
    features_audio_length,
    ingest_maec,
    parse_folder_name,
    read_sentences,
)
from ecvol.data.prices import write_price_parquet

# --- pure parsing ------------------------------------------------------------


def test_parse_folder_name():
    assert parse_folder_name("20150226_ADSK") == ("2015-02-26", "ADSK")
    assert parse_folder_name("20180621_BRK.B") == ("2018-06-21", "BRK.B")
    assert parse_folder_name("not_a_call") is None
    assert parse_folder_name("2015_ADSK") is None


def test_read_sentences(tmp_path: Path):
    f = tmp_path / "text.txt"
    f.write_text("First sentence.\n\n  Second.  \nThird.\n", encoding="utf-8")
    assert read_sentences(f) == ["First sentence.", "Second.", "Third."]
    assert read_sentences(tmp_path / "missing.txt") == []


def test_features_audio_length(tmp_path: Path):
    f = tmp_path / "features.csv"
    f.write_text(
        "Mean pitch,Audio Length\n100,1.5 \n200,--undefined--\n300,2.5\n", encoding="utf-8"
    )
    n_rows, total = features_audio_length(f)
    assert n_rows == 3  # all rows counted
    assert total == 4.0  # 1.5 + 2.5; --undefined-- skipped


# --- record assembly ---------------------------------------------------------


def _make_call(ds: Path, name: str, *, sentences: list[str], audio_lengths: list[float]) -> None:
    d = ds / name
    d.mkdir(parents=True)
    (d / "text.txt").write_text("\n".join(sentences), encoding="utf-8")
    lines = ["Mean pitch,Audio Length"]
    lines += [f"100,{al}" for al in audio_lengths]
    (d / "features.csv").write_text("\n".join(lines), encoding="utf-8")


def test_build_records_schema_and_status(tmp_path: Path):
    ds = tmp_path / "MAEC_Dataset"
    _make_call(
        ds, "20160321_AAA", sentences=["A long enough sentence here." * 2], audio_lengths=[12.5]
    )
    _make_call(ds, "20160322_BBB", sentences=["x"], audio_lengths=[0.3])  # too short → empty
    (ds / "weird_folder").mkdir()  # invalid name

    records = {r.call_id: r for r in build_records(ds)}
    aaa = records["20160321_AAA"]
    assert aaa.status == "ok" and aaa.ticker == "AAA" and aaa.call_date == "2016-03-21"
    assert aaa.source == "maec" and aaa.call_type == "earnings" and aaa.label == -1
    assert aaa.audio_exists is False and aaa.audio_path == ""
    assert aaa.audio_duration_sec == 12.5  # from features Audio Length
    assert aaa.assumed_after_hours is True and aaa.time_known is False
    assert records["20160322_BBB"].reason == "empty_transcript"
    assert records["weird_folder"].reason == "invalid_folder_name"


# --- end-to-end --------------------------------------------------------------


def _price_rows(center: date, days: int = 90) -> list[dict]:
    rows = []
    p = 100.0
    for i in range(-days, days + 1):
        d = center + timedelta(days=i)
        p += 0.5 if i % 2 else -0.3  # nonzero variance
        rows.append(
            {"date": d.isoformat(), "open": p, "high": p, "low": p, "close": p, "volume": 1000}
        )
    return rows


def test_ingest_maec_join_and_artifacts(tmp_path: Path):
    root = tmp_path / "data"
    ds = root / "raw" / "maec" / "repo" / "MAEC_Dataset"
    _make_call(
        ds,
        "20160321_AAA",
        sentences=["This is a genuinely long earnings-call sentence with plenty of characters."],
        audio_lengths=[30.0],
    )
    _make_call(
        ds,
        "20160321_ZZZ",
        sentences=["Another genuinely long earnings-call sentence with plenty of words here."],
        audio_lengths=[40.0],
    )  # no price data → reason-coded

    # AAA has prices; ZZZ does not.
    write_price_parquet(_price_rows(date(2016, 3, 21)), root / "prices" / "AAA.parquet")

    m = ingest_maec(root, horizons=(3,))

    assert m.total_calls == 2 and m.parsed == 2 and m.ok == 2
    assert m.joined == 1 and m.join_rate_pct == 50.0  # only AAA joins
    assert m.missing_price_tickers == 1

    # Artifacts.
    assert (root / "maec" / "calls.parquet").is_file()
    assert (root / "maec" / "targets.parquet").is_file()
    assert (root / "manifests" / "maec_calls.json").is_file()
    assert (root / "manifests" / "maec_targets.json").is_file()
    for name in ("maec_ingest_report.csv", "maec_audio_features.csv", "maec_join_audit.csv"):
        assert (root / "coverage" / name).is_file()

    # Missing-ticker reason coding (zero silent drops).
    missing = (root / "coverage" / "maec_missing_tickers.csv").read_text()
    assert "ZZZ,no_price_data" in missing

    # call_id is a string column; common-schema columns present.
    schema = pq.read_schema(root / "maec" / "calls.parquet")
    assert schema.field("call_id").type == pa.string()
    for col in ("ticker", "transcript_json", "audio_path", "speaker_metadata", "source"):
        assert col in schema.names


def test_ingest_maec_deterministic(tmp_path: Path):
    root = tmp_path / "data"
    ds = root / "raw" / "maec" / "repo" / "MAEC_Dataset"
    _make_call(
        ds,
        "20160321_AAA",
        sentences=["This is a genuinely long earnings-call sentence with plenty of characters."],
        audio_lengths=[30.0],
    )
    ingest_maec(root, horizons=(3,))
    c1 = (root / "maec" / "calls.parquet").read_bytes()
    t1 = (root / "maec" / "targets.parquet").read_bytes()
    ingest_maec(root, horizons=(3,))
    assert (root / "maec" / "calls.parquet").read_bytes() == c1
    assert (root / "maec" / "targets.parquet").read_bytes() == t1


def test_maec_calls_parquet_every_folder_accounted(tmp_path: Path):
    root = tmp_path / "data"
    ds = root / "raw" / "maec" / "repo" / "MAEC_Dataset"
    _make_call(
        ds,
        "20160321_AAA",
        sentences=["This is a genuinely long earnings-call sentence with plenty of characters."],
        audio_lengths=[30.0],
    )
    _make_call(ds, "20160322_BBB", sentences=["y"], audio_lengths=[0.1])
    ingest_maec(root, horizons=(3,))
    df = pd.read_parquet(root / "maec" / "calls.parquet")
    assert len(df) == 2  # one row per folder, including the excluded one
    assert set(df["status"]) == {"ok", "excluded"}
