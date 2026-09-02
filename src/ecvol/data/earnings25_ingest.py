"""Earnings25 ingestion onto the common call schema (T7.1; DECISIONS 2026-08-23 §6, 2026-09-02).

Earnings25 (Zenodo 10.5281/zenodo.18762168, CC-BY-4.0) ships `testset-full/data.jsonl`
— one record per call with the transcript, CTC-aligned speaker-attributed
`speech_segments`, `audio_info`, and `extra_fields` (`Company`, `Country`,
`ReleaseDate`, `Industry`, `MarketCap`, `speaker_attributions`) — plus one MP3 per
call. It carries **no ticker** and its "S&P 500" sampling matched company *names*,
so ~5% of the calls are look-alikes (Grainger PLC, Paramount Group, Vertex Inc.,
…; `docs/earnings25_verification_2026-09.md` §4).

Universe rule (DECISIONS 2026-09-02 §2): a call is admitted iff its `Company`
resolves to a SEC ticker (the T1.4 matcher + a few curated overrides) **and** that
company (by CIK) was an S&P 500 constituent in the Wikipedia list as of the end
of the call's month — one cached page revision per month, fetched through the
MediaWiki API and manifested. Everything else is reason-coded: `lookalike`
(curated block list of name collisions that would otherwise resolve to the S&P
company's ticker), `unresolved`, `not_sp500`.

Call timing: `ReleaseDate` is a UTC datetime (verified against transcript clock
mentions, verification note §3). It is converted to America/New_York and passed
to the target machinery as a real timestamp, so this is the first corpus where
the after-hours rule runs on measured times (`time_known=True`,
`assumed_after_hours=False`). The per-call times are written to
`coverage/earnings25_call_times.csv` (provenance tier `earnings25_metadata`) for
T9.2.

Prices live in their own store (`prices_earnings25/`, 2025-06 → 2026-03) so the
2014–2022 archive the FinCall/MAEC targets were validated against is untouched.
Targets are computed under both horizon conventions (T9.1) with the same tested
machinery. `ecvol data ingest earnings25` is deterministic and idempotent; every
call yields exactly one row with a `status`/`reason` — zero silent drops.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import requests

from ecvol.collect.discovery import _cik_by_ticker, parse_constituents
from ecvol.data.calls import NAN, CallRecord, write_calls_parquet, write_metric_csv
from ecvol.data.fincall_identity import _lookup_name, load_sec_table
from ecvol.data.manifests import make_entry, write_manifest
from ecvol.data.prices import load_close_series
from ecvol.data.targets import (
    CONVENTIONS,
    HORIZONS,
    TARGET_FILES,
    compute_call_targets,
    write_targets_parquet,
)

SOURCE = "earnings25"
DATASET_REL = "raw/earnings25/earnings-25/testset-full"
REVISIONS_REL = "raw/live/sp500_revisions"
PRICES_REL = "prices_earnings25"
ET = ZoneInfo("America/New_York")
MIN_TRANSCRIPT_CHARS = 50

WIKI_TITLE = "List of S&P 500 companies"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_INDEX = "https://en.wikipedia.org/w/index.php"
WIKI_HEADERS = {"User-Agent": "ecvol-research/0.1 (andrei.roman.personal@gmail.com)"}
WIKI_LICENSE = "Wikipedia — CC-BY-SA-4.0 (cached page revision)"

CALLS_LICENSE = "Derived artifact — normalized from Earnings25 (CC-BY-4.0)"
CALLS_SOURCE = "computed: ecvol data ingest earnings25 (T7.1)"
TARGETS_LICENSE = "Derived artifact — computed from price data (DESIGN §5.3); no external source"
TARGETS_SOURCE = "computed: ecvol data ingest earnings25 / targets (DESIGN §5.3)"
RAW_LICENSE = "Earnings25 — CC-BY-4.0 (Zenodo 10.5281/zenodo.18762168)"
RAW_SOURCE = "https://zenodo.org/records/18762168"

# Company names the generic matcher cannot resolve (initials, "The", apostrophes),
# all verified S&P 500 members — and name collisions that *would* resolve to the
# S&P company's ticker but are a different listed company (reason `lookalike`).
TICKER_OVERRIDES = {
    "The Charles Schwab Corporation": "SCHW",
    "First Solar, Inc.": "FSLR",
    "T. Rowe Price Group, Inc.": "TROW",
    "A. O. Smith Corporation": "AOS",
    "O'Reilly Automotive, Inc.": "ORLY",
    "J.B. Hunt Transport Services, Inc.": "JBHT",
    "W. R. Berkley Corporation": "WRB",
    "Dayforce, Inc.": "DAY",
}
LOOKALIKES = frozenset(
    {
        "Grainger PLC",  # UK landlord, not W.W. Grainger
        "Paramount Group, Inc.",  # office REIT, not Paramount Skydance
        "Vertex, Inc.",  # tax software, not Vertex Pharmaceuticals
        "Domino's Pizza Group plc",  # UK franchisee, not Domino's Pizza Inc.
    }
)
_LEGAL_TAIL = re.compile(r"\b(Public Limited Company|S\.A\. de C\.V\.|N\.V\.)\s*$")


# --- pure parsing ------------------------------------------------------------


def read_records(dataset_dir: Path) -> list[dict]:
    """The `data.jsonl` records, in file order."""
    path = dataset_dir / "data.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def clean_company(name: str) -> str:
    return _LEGAL_TAIL.sub("", name.replace("`", "'")).strip(" ,")


def resolve_ticker(company: str, sec: dict) -> tuple[str, str]:
    """(ticker, reason): reason is "" when resolved, else `lookalike` / `unresolved`."""
    if company in LOOKALIKES:
        return "", "lookalike"
    if company in TICKER_OVERRIDES:
        return TICKER_OVERRIDES[company], ""
    hit = _lookup_name(clean_company(company), sec)
    return (hit[0], "") if hit else ("", "unresolved")


def release_to_et(release: str) -> datetime:
    """`ReleaseDate` (UTC, 'YYYY-MM-DDTHH:MM:SS') → naive America/New_York datetime."""
    utc = datetime.fromisoformat(release).replace(tzinfo=ZoneInfo("UTC"))
    return utc.astimezone(ET).replace(tzinfo=None)


def segments_to_turns(segments: list[dict], attributions: dict[str, str]) -> list[dict]:
    """Merge consecutive same-speaker segments into {role, speaker, text} turns.

    Roles: "operator" when the attributed name is Operator, else "unknown" —
    Earnings25 attributes names, not roles; T3.1's sectioning falls back to
    in-text cues for the management/analyst boundary exactly as it does for MAEC.
    """
    turns: list[dict] = []
    for seg in segments:
        ids = seg.get("speaker_ids") or []
        sid = str(ids[0]) if ids else ""
        name = attributions.get(sid, "")
        role = "operator" if name.strip().lower() == "operator" else "unknown"
        text = (seg.get("segment_transcript") or "").strip()
        if not text:
            continue
        if turns and turns[-1]["speaker"] == sid:
            turns[-1]["text"] += " " + text
        else:
            turns.append({"role": role, "speaker": sid, "text": text})
    return turns


def _speaker_summary(turns: list[dict], attributions: dict[str, str]) -> dict:
    roles: dict[str, dict[str, int]] = {}
    for t in turns:
        r = roles.setdefault(t["role"], {"turns": 0, "chars": 0})
        r["turns"] += 1
        r["chars"] += len(t["text"])
    return {"n_turns": len(turns), "roles": roles, "speakers": attributions}


# --- S&P 500 membership by month (Wikipedia page revisions) ------------------


def fetch_sp500_revision(root: Path, month: str) -> Path:
    """Cache the last revision of the S&P 500 list in `month` (YYYY-MM); idempotent."""
    out = root / REVISIONS_REL
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob(f"wiki_sp500_{month}_rev*.html"))
    if existing:
        return existing[0]
    y, m = (int(x) for x in month.split("-"))
    end = date(y + (m == 12), (m % 12) + 1, 1)  # first day of the next month
    q = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "revisions",
            "titles": WIKI_TITLE,
            "rvlimit": 1,
            "rvstart": f"{end.isoformat()}T00:00:00Z",
            "rvdir": "older",
            "rvprop": "ids|timestamp",
            "format": "json",
        },
        headers=WIKI_HEADERS,
        timeout=60,
    )
    q.raise_for_status()
    page = next(iter(q.json()["query"]["pages"].values()))
    revid = page["revisions"][0]["revid"]
    html = requests.get(
        WIKI_INDEX,
        params={"title": WIKI_TITLE.replace(" ", "_"), "oldid": revid},
        headers=WIKI_HEADERS,
        timeout=120,
    )
    html.raise_for_status()
    target = out / f"wiki_sp500_{month}_rev{revid}.html"
    tmp = target.with_suffix(".part")
    tmp.write_text(html.text, encoding="utf-8")
    tmp.replace(target)
    time.sleep(1)
    return target


def membership_by_month(root: Path, months: list[str]) -> tuple[dict[str, set[str]], list[Path]]:
    """{month → set of member CIKs} from the cached revisions; fetches missing months."""
    cik_by_ticker = _cik_by_ticker(root)
    members: dict[str, set[str]] = {}
    pages: list[Path] = []
    for month in months:
        page = fetch_sp500_revision(root, month)
        pages.append(page)
        rows = parse_constituents(page.read_text(encoding="utf-8"))
        ciks = {cik_by_ticker.get(r["ticker"], "") for r in rows}
        ciks.discard("")
        members[month] = ciks
    return members, pages


# --- record assembly ---------------------------------------------------------


@dataclass
class Earnings25Call:
    record: CallRecord
    release_utc: str
    call_datetime_et: str
    country: str
    industry: str
    market_cap: str
    sample_rate: int
    cik: str


def build_records(
    records: list[dict],
    sec: dict,
    members: dict[str, set[str]],
    cik_by_ticker: dict[str, str],
    dataset_dir: Path | None = None,
) -> list[Earnings25Call]:
    """One CallRecord per Earnings25 call, universe rule applied, zero silent drops.

    `audio_exists` is checked on disk when `dataset_dir` is given (else False).
    """
    out: list[Earnings25Call] = []
    for rec in records:
        audio_rel = str(rec.get("audio_file_path") or "")
        audio_exists = bool(dataset_dir and audio_rel and (dataset_dir / audio_rel).is_file())
        ef = rec["extra_fields"]
        company, release = ef["Company"], ef["ReleaseDate"]
        ticker, reason = resolve_ticker(company, sec)
        cik = cik_by_ticker.get(ticker, "") if ticker else ""
        et_dt = release_to_et(release)
        month = release[:7]
        if not reason and cik not in members.get(month, set()):
            reason = "not_sp500"

        attributions = json.loads(ef.get("speaker_attributions") or "{}")
        attributions = {str(k): str(v) for k, v in attributions.items()}
        turns = segments_to_turns(rec.get("speech_segments") or [], attributions)
        n_chars = sum(len(t["text"]) for t in turns)
        parsed = bool(turns) and n_chars >= MIN_TRANSCRIPT_CHARS
        if not reason and not parsed:
            reason = "empty_transcript"

        info = rec.get("audio_info") or {}
        call_id = str(rec["id"])
        out.append(
            Earnings25Call(
                record=CallRecord(
                    call_id=call_id,
                    source=SOURCE,
                    ticker=ticker if not reason or reason == "empty_transcript" else "",
                    call_date=et_dt.date().isoformat(),
                    time_known=True,
                    assumed_after_hours=False,
                    call_type="earnings",
                    label=-1,
                    n_turns=len(turns),
                    n_chars=n_chars,
                    transcript_json=json.dumps(turns, ensure_ascii=False),
                    speaker_metadata=json.dumps(
                        _speaker_summary(turns, attributions), ensure_ascii=False
                    ),
                    audio_path=f"{DATASET_REL}/{audio_rel}" if audio_rel else "",
                    audio_exists=audio_exists,
                    audio_duration_sec=float(info.get("duration_seconds") or NAN),
                    parsed=parsed,
                    status="ok" if not reason else "excluded",
                    reason=reason,
                ),
                release_utc=release,
                call_datetime_et=et_dt.isoformat(timespec="minutes"),
                country=str(ef.get("Country") or ""),
                industry=str(ef.get("Industry") or ""),
                market_cap=str(ef.get("MarketCap") or ""),
                sample_rate=int(info.get("sample_rate") or 0),
                cik=cik,
            )
        )
    return out


# --- audio probe (bitrate strata for T9.4) -----------------------------------


def probe_bitrate(path: Path) -> tuple[int, float] | None:
    """(sample_rate, kbps) via ffprobe; None when the file is missing or undecodable."""
    if not path.exists():
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate:format=bit_rate",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        vals = [v for line in proc.stdout.splitlines() for v in line.split(",") if v]
        return int(float(vals[0])), float(vals[1]) / 1000
    except (ValueError, OSError, IndexError):
        return None


def bitrate_stratum(kbps: float) -> str:
    """The T9.4 strata (DECISIONS 2026-09-02 §1): 64 kbps vs ≤24 kbps (else `other`)."""
    if kbps >= 56:
        return "64k"
    if kbps <= 24.5:
        return "le24k"
    return "other"


# --- targets -----------------------------------------------------------------


def compute_targets(calls: list[Earnings25Call], prices_dir: Path, *, horizons, convention):
    close_cache: dict[str, dict[str, float]] = {}
    rows = []
    for c in calls:
        r = c.record
        if r.status != "ok":
            continue
        if r.ticker not in close_cache:
            close_cache[r.ticker] = load_close_series(prices_dir, r.ticker)
        call = {
            "call_id": r.call_id,
            "ticker": r.ticker,
            "date": r.call_date,
            "call_type": r.call_type,
            "timestamp": datetime.fromisoformat(c.call_datetime_et),
        }
        rows.extend(
            compute_call_targets(
                call, close_cache[r.ticker], horizons=horizons, convention=convention
            )
        )
    return rows


# --- orchestration -----------------------------------------------------------


@dataclass
class Earnings25Summary:
    total_calls: int
    ok: int
    reason_counts: dict[str, int]
    months: list[str]
    joined: int
    join_rate_pct: float
    missing_price_tickers: int
    strata: dict[str, int]


def _write_call_times(calls: list[Earnings25Call], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["call_id", "ticker", "release_utc", "call_datetime_et", "tier"])
        for c in sorted(calls, key=lambda c: c.record.call_id):
            w.writerow(
                [
                    c.record.call_id,
                    c.record.ticker,
                    c.release_utc,
                    c.call_datetime_et,
                    "earnings25_metadata",
                ]
            )


def _write_inventory(calls: list[Earnings25Call], probes: dict[str, tuple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            [
                "call_id",
                "ticker",
                "cik",
                "country",
                "industry",
                "market_cap",
                "release_utc",
                "status",
                "reason",
                "sample_rate",
                "kbps",
                "stratum",
                "duration_sec",
            ]
        )
        for c in sorted(calls, key=lambda c: c.record.call_id):
            probe = probes.get(c.record.call_id)
            kbps = probe[1] if probe else NAN
            w.writerow(
                [
                    c.record.call_id,
                    c.record.ticker,
                    c.cik,
                    c.country,
                    c.industry,
                    c.market_cap,
                    c.release_utc,
                    c.record.status,
                    c.record.reason,
                    probe[0] if probe else c.sample_rate,
                    f"{kbps:.3f}" if kbps == kbps else "",
                    bitrate_stratum(kbps) if kbps == kbps else "",
                    f"{c.record.audio_duration_sec:.3f}",
                ]
            )


def ingest_earnings25(
    root: Path, *, horizons=HORIZONS, probe_audio: bool = True
) -> Earnings25Summary:
    """Normalize Earnings25 onto the common schema + targets; write all artifacts."""
    dataset_dir = root / DATASET_REL
    if not (dataset_dir / "data.jsonl").is_file():
        raise ValueError(
            f"Earnings25 not found at {dataset_dir} (see docs/earnings25_verification)"
        )

    records = read_records(dataset_dir)
    months = sorted({r["extra_fields"]["ReleaseDate"][:7] for r in records})
    members, pages = membership_by_month(root, months)
    sec, _ = load_sec_table(root)
    cik_by_ticker = _cik_by_ticker(root)
    calls = build_records(records, sec, members, cik_by_ticker, dataset_dir)

    out_dir = root / SOURCE
    write_calls_parquet([c.record for c in calls], out_dir / "calls.parquet", id_type=pa.string())

    prices_dir = root / PRICES_REL
    target_paths = {c: out_dir / TARGET_FILES[c] for c in CONVENTIONS}
    target_rows = {
        c: compute_targets(calls, prices_dir, horizons=horizons, convention=c) for c in CONVENTIONS
    }
    for c in CONVENTIONS:
        write_targets_parquet(target_rows[c], target_paths[c], id_type=pa.string())

    ok_calls = [c for c in calls if c.record.status == "ok"]
    joined_ids = {r.call_id for r in target_rows["trading"] if r.status == "ok"}
    joined = sum(1 for c in ok_calls if c.record.call_id in joined_ids)
    join_rate = round(100 * joined / len(ok_calls), 2) if ok_calls else 0.0
    missing = {
        c.record.ticker
        for c in ok_calls
        if not (prices_dir / f"{c.record.ticker}.parquet").is_file()
    }

    probes: dict[str, tuple] = {}
    if probe_audio:
        for c in calls:
            if not c.record.audio_exists:
                continue
            p = probe_bitrate(root / c.record.audio_path)
            if p is not None:
                probes[c.record.call_id] = p
    strata: dict[str, int] = {}
    for _sr, kbps in probes.values():
        strata[bitrate_stratum(kbps)] = strata.get(bitrate_stratum(kbps), 0) + 1

    reason_counts: dict[str, int] = {}
    for c in calls:
        if c.record.reason:
            reason_counts[c.record.reason] = reason_counts.get(c.record.reason, 0) + 1
    target_reasons: dict[str, int] = {}
    for r in target_rows["trading"]:
        if r.reason:
            target_reasons[r.reason] = target_reasons.get(r.reason, 0) + 1

    summary = Earnings25Summary(
        total_calls=len(calls),
        ok=len(ok_calls),
        reason_counts=dict(sorted(reason_counts.items())),
        months=months,
        joined=joined,
        join_rate_pct=join_rate,
        missing_price_tickers=len(missing),
        strata=dict(sorted(strata.items())),
    )

    cov = root / "coverage"
    write_metric_csv(
        [
            ("total_calls", summary.total_calls),
            ("ok_calls", summary.ok),
            ("months", " ".join(months)),
            ("joined_calls", joined),
            ("join_rate_pct", join_rate),
            ("missing_price_tickers", len(missing)),
        ]
        + [(f"reason:{k}", v) for k, v in summary.reason_counts.items()]
        + [(f"target_reason:{k}", v) for k, v in sorted(target_reasons.items())]
        + [(f"stratum:{k}", v) for k, v in summary.strata.items()],
        cov / "earnings25_ingest_report.csv",
    )
    _write_call_times(calls, cov / "earnings25_call_times.csv")
    _write_inventory(calls, probes, cov / "earnings25_inventory.csv")
    write_metric_csv(
        [(t, "no_price_data") for t in sorted(missing)], cov / "earnings25_missing_tickers.csv"
    )

    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest(
        [make_entry(dataset_dir / "data.jsonl", root, source_url=RAW_SOURCE, license=RAW_LICENSE)]
        + [make_entry(p, root, source_url=WIKI_INDEX, license=WIKI_LICENSE) for p in pages],
        root / "manifests" / "earnings25_raw.json",
    )
    write_manifest(
        [
            make_entry(
                out_dir / "calls.parquet", root, source_url=CALLS_SOURCE, license=CALLS_LICENSE
            )
        ],
        root / "manifests" / "earnings25_calls.json",
    )
    write_manifest(
        [
            make_entry(target_paths[c], root, source_url=TARGETS_SOURCE, license=TARGETS_LICENSE)
            for c in CONVENTIONS
        ],
        root / "manifests" / "earnings25_targets.json",
    )
    return summary
