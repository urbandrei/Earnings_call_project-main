"""The common call schema shared by every dataset ingester (T1.4, T1.5).

A `CallRecord` is one normalized earnings-call record:

    (call_id, ticker, utc_timestamp, transcript_json, audio_path,
     speaker_metadata, source)

plus the accounting columns (`status`/`reason`, audio/parse flags) that keep
ingestion honest — every call yields exactly one row, never a silent drop.

`call_id` is the dataset's native id: an int for FinCall (numeric ids), a string
`YYYYMMDD_TICKER` for MAEC. The two `calls.parquet` files therefore differ only
in the call_id column type; `write_calls_parquet(..., id_type=...)` selects it,
and the `source` column disambiguates when the datasets are later unioned (cast
call_id to string then).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

NAN = float("nan")


@dataclass
class CallRecord:
    call_id: int | str
    source: str
    ticker: str
    call_date: str  # ISO yyyy-mm-dd; "" when unknown
    time_known: bool  # call time-of-day available? (never, for these corpora)
    assumed_after_hours: bool  # the §5.3 documented fallback was applied
    call_type: str
    label: int  # dataset-native label; -1 when absent (MAEC has none)
    n_turns: int  # FinCall: speaker turns; MAEC: sentences (see source)
    n_chars: int
    transcript_json: str  # JSON list of {role, text} turns/sentences
    speaker_metadata: str  # JSON {n_turns, roles:{role:{turns,chars}}}
    audio_path: str  # relative; "" when no raw audio file is held
    audio_exists: bool  # a raw audio file is present (MAEC ships none)
    audio_duration_sec: float  # NaN when unknown
    parsed: bool  # transcript produced a usable record
    status: str  # "ok" (usable earnings record) | "excluded"
    reason: str  # parse/cohort reason code; "" when status == ok


def write_calls_parquet(
    records: list[CallRecord], path: Path, *, id_type: pa.DataType | None = None
) -> None:
    """Deterministic parquet (sorted by call_id), matching the T0.3 convention."""
    id_type = id_type or pa.int64()  # FinCall default; MAEC passes pa.string()
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: r.call_id)
    cols = {f: [getattr(r, f) for r in records] for f in CallRecord.__dataclass_fields__}
    table = pa.table(
        {
            "call_id": pa.array(cols["call_id"], id_type),
            "source": pa.array(cols["source"], pa.string()),
            "ticker": pa.array(cols["ticker"], pa.string()),
            "call_date": pa.array(cols["call_date"], pa.string()),
            "time_known": pa.array(cols["time_known"], pa.bool_()),
            "assumed_after_hours": pa.array(cols["assumed_after_hours"], pa.bool_()),
            "call_type": pa.array(cols["call_type"], pa.string()),
            "label": pa.array(cols["label"], pa.int64()),
            "n_turns": pa.array(cols["n_turns"], pa.int64()),
            "n_chars": pa.array(cols["n_chars"], pa.int64()),
            "transcript_json": pa.array(cols["transcript_json"], pa.string()),
            "speaker_metadata": pa.array(cols["speaker_metadata"], pa.string()),
            "audio_path": pa.array(cols["audio_path"], pa.string()),
            "audio_exists": pa.array(cols["audio_exists"], pa.bool_()),
            "audio_duration_sec": pa.array(cols["audio_duration_sec"], pa.float64()),
            "parsed": pa.array(cols["parsed"], pa.bool_()),
            "status": pa.array(cols["status"], pa.string()),
            "reason": pa.array(cols["reason"], pa.string()),
        }
    )
    pq.write_table(table, path, compression="none", store_schema=True)


def write_metric_csv(rows: list[tuple[str, object]], path: Path) -> None:
    """Committed human-readable (metric,value) report — int/float types preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["metric", "value"])
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")
