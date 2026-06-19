"""T1.4 FinCall ingestion: transcript parsing, schema records, reports, join audit.

Filesystem fixtures are built in tmp dirs (no network, no ffprobe); audio
durations are injected so the tests don't depend on a real mp3 decoder.
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ecvol.data.fincall_ingest import (
    ROLE_MARKERS,
    build_records,
    ingest_fincall,
    parse_transcript,
    speaker_summary,
)

# --- transcript parsing ------------------------------------------------------


def test_parse_transcript_splits_on_role_markers():
    text = (
        "Operator: Good morning.Executives: Thanks for joining."
        "Analysts: My first question?Executives: Great question."
    )
    turns = parse_transcript(text)
    assert [t["role"] for t in turns] == ["operator", "management", "analyst", "management"]
    assert turns[0]["text"] == "Good morning."
    assert turns[2]["text"] == "My first question?"


def test_parse_transcript_preamble_is_unknown_role():
    turns = parse_transcript("Welcome everyone.Executives: Hello.")
    assert turns[0]["role"] == "unknown"
    assert turns[0]["text"] == "Welcome everyone."
    assert turns[1]["role"] == "management"


def test_parse_transcript_handles_glued_marker():
    # The corpus glues markers to the prior word: "...North AmericaAnalysts:".
    turns = parse_transcript("Executives: We grew in North AmericaAnalysts: Thanks.")
    assert turns[0] == {"role": "management", "text": "We grew in North America"}
    assert turns[1] == {"role": "analyst", "text": "Thanks."}


def test_parse_transcript_empty():
    assert parse_transcript("") == []
    assert parse_transcript("   ") == []


def test_speaker_summary_counts():
    turns = parse_transcript("Executives: abcde.Analysts: fg.Executives: hij.")
    s = speaker_summary(turns)
    assert s["n_turns"] == 3
    assert s["roles"]["management"]["turns"] == 2
    assert s["roles"]["analyst"]["turns"] == 1


def test_role_markers_cover_known_roles():
    assert set(ROLE_MARKERS.values()) >= {"management", "analyst", "operator"}


# --- record assembly ---------------------------------------------------------


def _write_corpus(raw: Path, calls: dict[int, dict]) -> None:
    """Write a minimal transcripts_2019.json (other years empty)."""
    raw.mkdir(parents=True, exist_ok=True)
    by_year: dict[int, dict] = {2019: {}, 2020: {}, 2021: {}}
    for cid, rec in calls.items():
        by_year[rec.pop("_year", 2019)][str(cid)] = rec
    for year, data in by_year.items():
        (raw / f"transcripts_{year}.json").write_text(json.dumps(data), encoding="utf-8")


def test_build_records_status_and_reasons():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td)
        _write_corpus(
            raw,
            {
                100: {"input": "Operator: Hi.Executives: " + "x" * 60, "mp3_id": "m1", "label": 1},
                101: {"input": "Operator: Hi.Executives: " + "y" * 60, "mp3_id": "m2", "label": 0},
                102: {"input": "tiny", "mp3_id": "m3", "label": 1},  # empty_transcript
                103: {"input": "Operator: Hi.Executives: " + "z" * 60, "mp3_id": "m4", "label": 1},
            },
        )
        identity = {
            "100": {"ticker": "AAA", "date": "2019-03-01", "call_type": "earnings"},
            "101": {"ticker": "", "date": "2019-03-02", "call_type": "earnings"},  # unresolved
            "103": {
                "ticker": "DDD",
                "date": "2019-03-04",
                "call_type": "conference",
            },  # non_earnings
        }
        records = build_records(raw, identity)
        by_id = {r.call_id: r for r in records}

        assert by_id[100].status == "ok" and by_id[100].reason == ""
        assert by_id[100].label == 1 and by_id[100].time_known is False
        assert by_id[100].assumed_after_hours is True
        assert by_id[101].reason == "unresolved_ticker"
        assert by_id[102].reason == "empty_transcript" and by_id[102].parsed is False
        assert by_id[103].reason == "non_earnings"


def test_build_records_no_date_reason():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td)
        _write_corpus(raw, {200: {"input": "Executives: " + "a" * 60, "mp3_id": "m"}})
        identity = {"200": {"ticker": "AAA", "date": "", "call_type": "earnings"}}
        records = build_records(raw, identity)
        assert records[0].reason == "no_date"


def test_build_records_audio_duration_injected_and_transcript_roundtrips():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td)
        _write_corpus(raw, {300: {"input": "Executives: " + "a" * 60, "mp3_id": "m"}})
        identity = {"300": {"ticker": "AAA", "date": "2019-05-01", "call_type": "earnings"}}
        records = build_records(raw, identity, durations={"300": 1234.5})
        r = records[0]
        assert r.audio_duration_sec == 1234.5
        turns = json.loads(r.transcript_json)
        assert turns[0]["role"] == "management"
        assert json.loads(r.speaker_metadata)["n_turns"] == 1


# --- end-to-end ingest -------------------------------------------------------


def _full_corpus(root: Path) -> None:
    raw = root / "raw" / "fincall"
    _write_corpus(
        raw,
        {
            1: {"input": "Operator: Hi.Executives: " + "a" * 60, "mp3_id": "z1", "label": 1},
            2: {"input": "Operator: Hi.Executives: " + "b" * 60, "mp3_id": "z2", "label": 0},
            3: {"input": "nope", "mp3_id": "z3", "label": 1},
        },
    )
    (root / "identity").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"call_id": "1", "ticker": "AAA", "date": "2019-03-01", "call_type": "earnings"},
            {"call_id": "2", "ticker": "BBB", "date": "2019-03-02", "call_type": "earnings"},
            {"call_id": "3", "ticker": "CCC", "date": "2019-03-03", "call_type": "earnings"},
        ]
    ).to_csv(root / "identity" / "fincall_identity.csv", index=False)


def test_ingest_writes_artifacts_and_audits_join(tmp_path: Path):
    root = tmp_path / "data"
    _full_corpus(root)
    # Targets parquet: call 1 has an ok row, call 2 does not.
    (root / "fincall").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"call_id": 1, "status": "ok"},
            {"call_id": 2, "status": "excluded"},
        ]
    ).to_parquet(root / "fincall" / "targets.parquet")

    s = ingest_fincall(root, probe_audio=False)

    assert s.total_calls == 3
    assert s.ok == 2  # calls 1 and 2 (resolved earnings); call 3 has empty transcript
    assert s.reason_counts.get("empty_transcript") == 1
    # Join audit: cohort = {1,2}; only call 1 joins → 50%.
    assert s.earnings_resolved == 2
    assert s.earnings_joined == 1
    assert s.join_rate_pct == 50.0

    # Artifacts exist.
    assert (root / "fincall" / "calls.parquet").is_file()
    assert (root / "manifests" / "fincall_calls.json").is_file()
    for name in (
        "fincall_ingest_report.csv",
        "fincall_audio_durations.csv",
        "fincall_join_audit.csv",
    ):
        assert (root / "coverage" / name).is_file()

    # 100% accounting: every call is a row.
    df = pd.read_parquet(root / "fincall" / "calls.parquet")
    assert len(df) == 3
    assert df["call_id"].tolist() == [1, 2, 3]


def test_ingest_parquet_deterministic(tmp_path: Path):
    root = tmp_path / "data"
    _full_corpus(root)
    ingest_fincall(root, probe_audio=False)
    b1 = (root / "fincall" / "calls.parquet").read_bytes()
    ingest_fincall(root, probe_audio=False)
    b2 = (root / "fincall" / "calls.parquet").read_bytes()
    assert b1 == b2


def test_ingest_parquet_schema_is_common(tmp_path: Path):
    root = tmp_path / "data"
    _full_corpus(root)
    ingest_fincall(root, probe_audio=False)
    schema = pq.read_schema(root / "fincall" / "calls.parquet")
    for col in ("call_id", "ticker", "transcript_json", "audio_path", "speaker_metadata", "source"):
        assert col in schema.names
