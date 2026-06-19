"""MAEC ingestion onto the common call schema (T1.5).

MAEC ships one folder per call, named `YYYYMMDD_TICKER`, each containing:
- `text.txt` — the transcript, one sentence per line, **no speaker labels**;
- `features.csv` — per-sentence acoustic functionals (pitch/intensity/jitter/
  shimmer/… + `Audio Length`), one row per sentence. MAEC ships **no raw audio**
  (the 59 GB MFCC archive link-rotted; DECISIONS 2026-06-12), so the audio
  signal here is these precomputed features, not waveforms.

So unlike FinCall (T1.4), identity needs no reconstruction — date and ticker are
in the folder name — but there is no speaker structure (transcript stored as
"unknown"-role sentences) and no raw-audio file. The record carries the same
common schema (`ecvol.data.calls.CallRecord`); `call_id` is the folder name
(string), `audio_path` is empty (`audio_exists=False`), and `audio_duration_sec`
is the sum of the sentence `Audio Length`s — the genuine call audio length the
features were computed over.

Targets are computed with the same tested machinery as FinCall (`targets.py`,
assume-after-hours fallback — MAEC has no call times either), giving a symmetric
price/target join rate. `ecvol data ingest maec` writes deterministic
`data/maec/{calls,targets}.parquet` (gitignored payloads) + committed manifests
and coverage reports; every call yields exactly one row with a `status`/`reason`
and every non-joining ticker is reason-coded — zero silent drops.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from ecvol.data.calls import NAN, CallRecord, write_calls_parquet, write_metric_csv
from ecvol.data.manifests import make_entry, write_manifest
from ecvol.data.prices import load_close_series
from ecvol.data.targets import HORIZONS, compute_call_targets, write_targets_parquet

SOURCE = "maec"
DATASET_REL = "raw/maec/repo/MAEC_Dataset"
FOLDER_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(.+)$")
MIN_TRANSCRIPT_CHARS = 50

CALLS_LICENSE = "Derived artifact — normalized from MAEC (CC-BY-SA-4.0)"
CALLS_SOURCE = "computed: ecvol data ingest maec (T1.5)"
TARGETS_LICENSE = "Derived artifact — computed from price data (DESIGN §5.3); no external source"
TARGETS_SOURCE = "computed: ecvol data ingest maec / targets (DESIGN §5.3)"


# --- pure parsing ------------------------------------------------------------


def parse_folder_name(name: str) -> tuple[str, str] | None:
    """`YYYYMMDD_TICKER` → (ISO date, ticker); None if the name doesn't match."""
    m = FOLDER_RE.match(name)
    if not m:
        return None
    y, mo, d, ticker = m.groups()
    return f"{y}-{mo}-{d}", ticker


def read_sentences(text_path: Path) -> list[str]:
    """Non-empty, stripped lines of a MAEC `text.txt` (one sentence per line)."""
    if not text_path.is_file():
        return []
    return [ln.strip() for ln in text_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def features_audio_length(features_path: Path) -> tuple[int, float]:
    """(#sentence rows, summed `Audio Length` seconds) from a MAEC `features.csv`.

    Unparseable / `--undefined--` lengths are skipped in the sum but still
    counted as sentence rows.
    """
    if not features_path.is_file():
        return 0, NAN
    n_rows = 0
    total = 0.0
    with open(features_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            raw = (row.get("Audio Length") or "").strip()
            try:
                total += float(raw)
            except ValueError:
                continue
    return n_rows, (total if n_rows else NAN)


# --- record assembly ---------------------------------------------------------


def _sentence_turns(sentences: list[str]) -> list[dict]:
    """MAEC has no speakers; each sentence is an "unknown"-role unit (schema parity)."""
    return [{"role": "unknown", "text": s} for s in sentences]


def build_records(dataset_dir: Path) -> list[CallRecord]:
    """One CallRecord per MAEC call folder."""
    records: list[CallRecord] = []
    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        parsed_name = parse_folder_name(folder.name)
        if parsed_name is None:
            records.append(_excluded_record(folder.name, "invalid_folder_name"))
            continue
        call_date, ticker = parsed_name

        sentences = read_sentences(folder / "text.txt")
        turns = _sentence_turns(sentences)
        n_chars = sum(len(s) for s in sentences)
        parsed = bool(sentences) and n_chars >= MIN_TRANSCRIPT_CHARS

        n_sentences, audio_len = features_audio_length(folder / "features.csv")
        speaker_meta = {
            "n_turns": len(turns),
            "roles": {"unknown": {"turns": len(turns), "chars": n_chars}},
            "speakers": "unavailable",  # MAEC text.txt has no speaker labels
        }

        status, reason = ("ok", "") if parsed else ("excluded", "empty_transcript")
        records.append(
            CallRecord(
                call_id=folder.name,
                source=SOURCE,
                ticker=ticker,
                call_date=call_date,
                time_known=False,
                assumed_after_hours=True,
                call_type="earnings",  # MAEC is all earnings calls
                label=-1,  # MAEC ships no FinCall-style surprise label
                n_turns=len(turns),  # sentences (no speaker turns) — see `source`
                n_chars=n_chars,
                transcript_json=json.dumps(turns, ensure_ascii=False),
                speaker_metadata=json.dumps(speaker_meta, ensure_ascii=False),
                audio_path="",  # MAEC ships no raw audio
                audio_exists=False,
                audio_duration_sec=audio_len,  # summed sentence Audio Length (features)
                parsed=parsed,
                status=status,
                reason=reason,
            )
        )
    return records


def _excluded_record(call_id: str, reason: str) -> CallRecord:
    return CallRecord(
        call_id=call_id,
        source=SOURCE,
        ticker="",
        call_date="",
        time_known=False,
        assumed_after_hours=True,
        call_type="earnings",
        label=-1,
        n_turns=0,
        n_chars=0,
        transcript_json="[]",
        speaker_metadata="{}",
        audio_path="",
        audio_exists=False,
        audio_duration_sec=NAN,
        parsed=False,
        status="excluded",
        reason=reason,
    )


# --- targets -----------------------------------------------------------------


def compute_targets(records: list[CallRecord], prices_dir: Path, *, horizons=HORIZONS):
    """Targets for every parsed MAEC call, reusing the FinCall target math."""
    close_cache: dict[str, dict[str, float]] = {}
    rows = []
    for r in records:
        if not r.parsed:
            continue
        if r.ticker not in close_cache:
            close_cache[r.ticker] = load_close_series(prices_dir, r.ticker)
        call = {
            "call_id": r.call_id,
            "ticker": r.ticker,
            "date": r.call_date,
            "call_type": r.call_type,
        }
        rows.extend(compute_call_targets(call, close_cache[r.ticker], horizons=horizons))
    return rows


# --- artifacts ---------------------------------------------------------------


@dataclass
class MaecSummary:
    total_calls: int
    parsed: int
    ok: int
    reason_counts: dict[str, int]
    calls_with_features: int
    total_sentences: int
    joined: int  # calls with >=1 ok target row
    join_rate_pct: float
    missing_price_tickers: int


def _is_decoded(x: float) -> bool:
    return x == x  # False only for NaN


def _audio_feature_rows(records: list[CallRecord]) -> list[tuple[str, object]]:
    durs = sorted(r.audio_duration_sec for r in records if _is_decoded(r.audio_duration_sec))
    sentences = sum(r.n_turns for r in records if r.parsed)
    rows: list[tuple[str, object]] = [
        ("calls_total", len(records)),
        ("calls_with_features", sum(1 for r in records if r.n_turns > 0)),
        ("calls_with_raw_audio", sum(1 for r in records if r.audio_exists)),  # MAEC: 0
        ("total_sentences", sentences),
        ("mean_sentences_per_call", round(sentences / max(1, sum(r.parsed for r in records)), 1)),
    ]
    if durs:
        n = len(durs)
        rows += [
            ("duration_source", "summed sentence Audio Length (MAEC ships no raw audio)"),
            ("duration_min_sec", round(durs[0], 1)),
            ("duration_median_sec", round(durs[n // 2], 1)),
            ("duration_max_sec", round(durs[-1], 1)),
            ("duration_mean_sec", round(sum(durs) / n, 1)),
            ("duration_total_hours", round(sum(durs) / 3600, 1)),
        ]
    return rows


def ingest_maec(root: Path, *, horizons=HORIZONS) -> MaecSummary:
    """Normalize MAEC onto the common schema + compute targets; write all artifacts."""
    dataset_dir = root / DATASET_REL
    if not dataset_dir.is_dir():
        raise ValueError(f"MAEC not found at {dataset_dir} — run `ecvol data fetch maec` first")

    records = build_records(dataset_dir)
    n = len(records)

    calls_path = root / SOURCE / "calls.parquet"
    write_calls_parquet(records, calls_path, id_type=pa.string())

    # Targets (price/target join), same machinery as FinCall.
    target_rows = compute_targets(records, root / "prices", horizons=horizons)
    targets_path = root / SOURCE / "targets.parquet"
    write_targets_parquet(target_rows, targets_path, id_type=pa.string())

    joined_ids = {r.call_id for r in target_rows if r.status == "ok"}
    parsed_calls = [r for r in records if r.parsed]
    joined = sum(1 for r in parsed_calls if r.call_id in joined_ids)
    join_rate = round(100 * joined / len(parsed_calls), 2) if parsed_calls else 0.0
    missing_tickers = {
        r.ticker for r in parsed_calls if not (root / "prices" / f"{r.ticker}.parquet").is_file()
    }

    reason_counts: dict[str, int] = {}
    for r in records:
        if r.reason:
            reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1
    target_reasons: dict[str, int] = {}
    for r in target_rows:
        if r.reason:
            target_reasons[r.reason] = target_reasons.get(r.reason, 0) + 1

    summary = MaecSummary(
        total_calls=n,
        parsed=len(parsed_calls),
        ok=sum(1 for r in records if r.status == "ok"),
        reason_counts=dict(sorted(reason_counts.items())),
        calls_with_features=sum(1 for r in records if r.n_turns > 0),
        total_sentences=sum(r.n_turns for r in parsed_calls),
        joined=joined,
        join_rate_pct=join_rate,
        missing_price_tickers=len(missing_tickers),
    )

    cov = root / "coverage"
    ingest_rows: list[tuple[str, object]] = [
        ("total_calls", n),
        ("parsed", summary.parsed),
        ("ok", summary.ok),
    ] + [(f"reason:{k}", v) for k, v in summary.reason_counts.items()]
    write_metric_csv(ingest_rows, cov / "maec_ingest_report.csv")
    write_metric_csv(_audio_feature_rows(records), cov / "maec_audio_features.csv")
    join_rows: list[tuple[str, object]] = [
        ("calls_parsed", len(parsed_calls)),
        ("calls_joined_to_targets", joined),
        ("join_rate_pct", join_rate),
        ("unique_tickers", len({r.ticker for r in parsed_calls})),
        ("missing_price_tickers", len(missing_tickers)),
    ] + [(f"target_reason:{k}", v) for k, v in sorted(target_reasons.items())]
    write_metric_csv(join_rows, cov / "maec_join_audit.csv")
    _write_missing_tickers(sorted(missing_tickers), cov / "maec_missing_tickers.csv")

    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest(
        [make_entry(calls_path, root, source_url=CALLS_SOURCE, license=CALLS_LICENSE)],
        root / "manifests" / "maec_calls.json",
    )
    write_manifest(
        [make_entry(targets_path, root, source_url=TARGETS_SOURCE, license=TARGETS_LICENSE)],
        root / "manifests" / "maec_targets.json",
    )
    return summary


def _write_missing_tickers(tickers: list[str], path: Path) -> None:
    """Committed list of MAEC tickers with no price data (reason-coded, no silent drops)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = ["ticker,reason\n"]
    buf += [f"{t},no_price_data\n" for t in tickers]
    path.write_text("".join(buf), encoding="utf-8")
