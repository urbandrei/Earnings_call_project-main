"""FinCall-Surprise ingestion onto the common call schema (T1.4).

Turns the raw release (`transcripts_{year}.json` + mirrored mp3s) plus the
reconstructed identity CSV (`fincall_identity.py`) into one normalized record
per call on the **common schema** shared with MAEC (T1.5):

    (call_id, ticker, utc_timestamp, transcript_json, audio_path,
     speaker_metadata, source)

Representation choices (honest, not fabricated):

- **utc_timestamp.** Call times-of-day are *not* available in this corpus — only
  3.4% of transcripts and 1.6% of slide PDFs mention any clock time in their
  head, and those are usually press-release times, not the call start. So no
  per-call time is extracted; the §10 risk #7 / DESIGN §5.3 documented fallback
  applies uniformly: **assume after-hours**. The record carries `call_date`
  (ISO) plus `time_known=False` / `assumed_after_hours=True`; downstream
  `targets.py` already consumes a missing timestamp as the after-hours case.
- **transcript_json.** The raw `input` field is one run of prose with coarse
  role markers ("Executives:", "Analysts:", "Operator:") glued inline. We split
  on those markers into role-tagged turns — the structural parse T3.1 refines
  (true per-speaker turns / sectioning are out of scope here).
- **speaker_metadata.** Per-role turn/char counts derived from the same parse.

Every call yields exactly one row with a `status`/`reason` — never a silent
drop (the T1.4 "100% parse or reason-coded" gate). The earnings cohort's join
rate to price/targets is audited against the ≥95% gate. Output: deterministic
`data/fincall/calls.parquet` (gitignored payload) + committed manifest and
coverage reports, written by `ecvol data ingest fincall`.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ecvol.data.calls import NAN, CallRecord, write_calls_parquet, write_metric_csv
from ecvol.data.manifests import make_entry, write_manifest

SOURCE = "fincall"
YEARS = (2019, 2020, 2021)
FINCALL_IDENTITY = "identity/fincall_identity.csv"
CALLS_LICENSE = "Derived artifact — normalized from FinCall-Surprise (Apache-2.0)"
CALLS_SOURCE = "computed: ecvol data ingest fincall (T1.4)"

# Coarse speaker-role markers the corpus uses (often glued to the prior word,
# e.g. "...North AmericaAnalysts:"). We split on the role word wherever it
# appears followed by a colon; mapped to the common-schema role vocabulary.
ROLE_MARKERS = {
    "Executives": "management",
    "Analysts": "analyst",
    "Operator": "operator",
    "Attendees": "attendee",
    "Shareholders": "shareholder",
}
ROLE_RE = re.compile(r"(" + "|".join(ROLE_MARKERS) + r"):\s*")
MIN_TRANSCRIPT_CHARS = 50  # below this the input is empty/boilerplate, not a call


# --- pure parsing ------------------------------------------------------------


def parse_transcript(text: str) -> list[dict]:
    """Split raw transcript prose into role-tagged turns.

    Text before the first role marker becomes an "unknown"-role turn (the
    operator/host preamble in calls that open without a marker). Empty turns are
    dropped; whitespace is collapsed at the edges only.
    """
    turns: list[dict] = []
    pos = 0
    role = "unknown"
    for m in ROLE_RE.finditer(text):
        chunk = text[pos : m.start()].strip()
        if chunk:
            turns.append({"role": role, "text": chunk})
        role = ROLE_MARKERS[m.group(1)]
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        turns.append({"role": role, "text": tail})
    return turns


def speaker_summary(turns: list[dict]) -> dict:
    """Per-role turn and character counts over a parsed transcript."""
    roles: dict[str, dict[str, int]] = {}
    for t in turns:
        r = roles.setdefault(t["role"], {"turns": 0, "chars": 0})
        r["turns"] += 1
        r["chars"] += len(t["text"])
    return {"n_turns": len(turns), "roles": roles}


def audio_relpath(year: int, mp3_id: str) -> str:
    """Path of a call's mp3 relative to the data root (posix, manifest-stable)."""
    return f"raw/{SOURCE}/mp3_{year}/{mp3_id}.mp3"


def probe_duration(path: Path) -> float | None:
    """Audio duration in seconds via ffprobe; None on missing file or decode error."""
    if not path.exists():
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        return float(proc.stdout.strip())
    except (ValueError, OSError):
        return None


# --- record assembly ---------------------------------------------------------


def _cohort_reason(parsed: bool, ticker: str, call_type: str, call_date: str) -> tuple[str, str]:
    """(status, reason) for a parsed call: ok iff resolved earnings call with a date.

    First failing reason wins (single-reason accounting, like targets.py). Audio
    availability is *not* a cohort gate — text-only models still use those calls;
    it is reported separately.
    """
    if not parsed:
        return "excluded", "empty_transcript"
    if not ticker:
        return "excluded", "unresolved_ticker"
    if call_type != "earnings":
        return "excluded", "non_earnings"
    if not call_date:
        return "excluded", "no_date"
    return "ok", ""


def build_records(
    raw: Path,
    identity: dict[str, dict],
    *,
    durations: dict[str, float] | None = None,
) -> list[CallRecord]:
    """One CallRecord per call across all year-files. `durations` maps call_id→sec."""
    durations = durations or {}
    records: list[CallRecord] = []
    for year in YEARS:
        data = json.loads((raw / f"transcripts_{year}.json").read_text(encoding="utf-8"))
        for call_id, rec in sorted(data.items()):
            ident = identity.get(call_id, {})
            ticker = str(ident.get("ticker") or "").strip()
            call_date = str(ident.get("date") or "").strip()
            call_type = str(ident.get("call_type") or "").strip()

            turns = parse_transcript(rec.get("input") or "")
            n_chars = sum(len(t["text"]) for t in turns)
            parsed = bool(turns) and n_chars >= MIN_TRANSCRIPT_CHARS

            mp3_id = str(rec.get("mp3_id") or "")
            mp3_abs = raw / f"mp3_{year}" / f"{mp3_id}.mp3"
            audio_exists = bool(mp3_id) and mp3_abs.exists()
            duration = durations.get(call_id, NAN)

            status, reason = _cohort_reason(parsed, ticker, call_type, call_date)
            try:
                label = int(rec.get("label"))
            except (TypeError, ValueError):
                label = -1

            records.append(
                CallRecord(
                    call_id=int(call_id),
                    source=SOURCE,
                    ticker=ticker,
                    call_date=call_date,
                    time_known=False,
                    assumed_after_hours=True,
                    call_type=call_type,
                    label=label,
                    n_turns=len(turns),
                    n_chars=n_chars,
                    transcript_json=json.dumps(turns, ensure_ascii=False),
                    speaker_metadata=json.dumps(speaker_summary(turns), ensure_ascii=False),
                    audio_path=audio_relpath(year, mp3_id) if audio_exists else "",
                    audio_exists=audio_exists,
                    audio_duration_sec=duration,
                    parsed=parsed,
                    status=status,
                    reason=reason,
                )
            )
    return records


# --- artifacts ---------------------------------------------------------------


@dataclass
class IngestSummary:
    total_calls: int
    parsed: int
    ok: int
    reason_counts: dict[str, int]
    audio_present: int
    audio_decoded: int
    earnings_calls: int
    earnings_resolved: int  # earnings + resolved ticker + date (cohort)
    earnings_joined: int  # of those, with >=1 ok target row
    join_rate_pct: float


def _is_decoded(r: CallRecord) -> bool:
    return r.audio_duration_sec == r.audio_duration_sec  # False only for NaN


def _duration_stats(records: list[CallRecord]) -> list[tuple[str, object]]:
    durs = sorted(r.audio_duration_sec for r in records if _is_decoded(r))
    if not durs:
        return [("audio_decoded", 0)]
    n = len(durs)

    def pct(p: float) -> float:
        return round(durs[min(n - 1, int(p * n))], 1)

    return [
        ("audio_decoded", n),
        ("duration_min_sec", round(durs[0], 1)),
        ("duration_p05_sec", pct(0.05)),
        ("duration_p25_sec", pct(0.25)),
        ("duration_median_sec", pct(0.50)),
        ("duration_p75_sec", pct(0.75)),
        ("duration_p95_sec", pct(0.95)),
        ("duration_max_sec", round(durs[-1], 1)),
        ("duration_mean_sec", round(sum(durs) / n, 1)),
        ("duration_total_hours", round(sum(durs) / 3600, 1)),
    ]


def _join_audit(
    records: list[CallRecord], targets_path: Path
) -> tuple[dict, list[tuple[str, object]]]:
    """Earnings-cohort join rate against targets.parquet (the ≥95% T1.4 gate).

    Cohort = earnings calls with a resolved ticker and a date (status == ok).
    Joined = cohort calls with at least one ok target row.
    """
    cohort = {r.call_id for r in records if r.status == "ok"}
    earnings = sum(1 for r in records if r.call_type == "earnings")
    joined_ids: set[int] = set()
    if targets_path.exists():
        tdf = pd.read_parquet(targets_path, columns=["call_id", "status"])
        joined_ids = set(tdf.loc[tdf["status"] == "ok", "call_id"].astype(int))
    joined = len(cohort & joined_ids)
    rate = round(100 * joined / len(cohort), 2) if cohort else 0.0
    stats = {
        "earnings_calls": earnings,
        "earnings_resolved": len(cohort),
        "earnings_joined": joined,
        "join_rate_pct": rate,
    }
    rows: list[tuple[str, object]] = [
        ("earnings_calls", earnings),
        ("earnings_resolved_cohort", len(cohort)),
        ("cohort_joined_to_targets", joined),
        ("join_rate_pct", rate),
        ("targets_present", int(targets_path.exists())),
    ]
    return stats, rows


def ingest_fincall(
    root: Path,
    *,
    probe_audio: bool = True,
) -> IngestSummary:
    """Normalize FinCall onto the common schema; write parquet + manifest + reports."""
    raw = root / "raw" / SOURCE
    ident_df = pd.read_csv(root / FINCALL_IDENTITY, dtype=str).fillna("")
    identity = {str(row["call_id"]): row for row in ident_df.to_dict("records")}

    durations: dict[str, float] = {}
    if probe_audio:
        durations = _probe_all_durations(raw, root)

    records = build_records(raw, identity, durations=durations)
    n = len(records)
    if n == 0:
        raise ValueError("no FinCall transcripts found — run `ecvol data fetch fincall` first")

    calls_path = root / SOURCE / "calls.parquet"
    write_calls_parquet(records, calls_path)

    reason_counts: dict[str, int] = {}
    for r in records:
        if r.reason:
            reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1
    reason_counts = dict(sorted(reason_counts.items()))

    join_stats, join_rows = _join_audit(records, root / "fincall" / "targets.parquet")

    summary = IngestSummary(
        total_calls=n,
        parsed=sum(1 for r in records if r.parsed),
        ok=sum(1 for r in records if r.status == "ok"),
        reason_counts=reason_counts,
        audio_present=sum(1 for r in records if r.audio_exists),
        audio_decoded=sum(1 for r in records if _is_decoded(r)),
        **join_stats,
    )

    # Committed coverage reports.
    cov = root / "coverage"
    ingest_rows: list[tuple[str, object]] = [
        ("total_calls", n),
        ("parsed", summary.parsed),
        ("ok", summary.ok),
        ("audio_present", summary.audio_present),
        ("audio_decoded", summary.audio_decoded),
    ] + [(f"reason:{k}", v) for k, v in reason_counts.items()]
    write_metric_csv(ingest_rows, cov / "fincall_ingest_report.csv")
    write_metric_csv(_duration_stats(records), cov / "fincall_audio_durations.csv")
    write_metric_csv(join_rows, cov / "fincall_join_audit.csv")

    # Manifest for the gitignored payload.
    entry = make_entry(calls_path, root, source_url=CALLS_SOURCE, license=CALLS_LICENSE)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest([entry], root / "manifests" / "fincall_calls.json")
    return summary


def _probe_all_durations(raw: Path, root: Path) -> dict[str, float]:
    """Probe every call's mp3 once, caching results (ffprobe dominates build time).

    Cache is keyed by call_id, persisted under data/raw/ref/, resumable across
    runs and idempotent (a present mp3 with an unchanged path is never re-probed).
    """
    cache_path = root / "raw" / "ref" / "fincall_audio_durations.json"
    cache: dict[str, float] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    out: dict[str, float] = {}
    dirty = False
    for year in YEARS:
        data = json.loads((raw / f"transcripts_{year}.json").read_text(encoding="utf-8"))
        for call_id, rec in data.items():
            mp3_id = str(rec.get("mp3_id") or "")
            mp3 = raw / f"mp3_{year}" / f"{mp3_id}.mp3"
            if call_id in cache:
                if cache[call_id] is not None:
                    out[call_id] = cache[call_id]
                continue
            dur = probe_duration(mp3)
            cache[call_id] = dur
            dirty = True
            if dur is not None:
                out[call_id] = dur
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    return out
