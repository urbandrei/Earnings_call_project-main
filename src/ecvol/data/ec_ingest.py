"""EC (Qin & Yang 2019 EarningsCall) ingestion onto the common call schema (T6R.2).

The reproduction study runs each published model on *its own* dataset (DECISIONS
2026-08-23 §5). HTML (Yang et al. 2020) and the VolTAGE/KeFVP lineage use the EC
release: 572 folders `CompanyName_YYYYMMDD` under `raw/ec/extracted/ACL19_Release/`,
each with `TextSequence.txt` (one executive sentence per line, no speaker roles)
and per-sentence CEO audio clips (`CEO/*.mp3`). Identity comes from the lineage's
own `full_stock_data.csv` (VolTAGE; `text_file_name` = folder name → ticker), so
no reconstruction is needed; the 14 folders absent from that table are
reason-coded `unresolved`.

Records mirror MAEC's shape (sentences as `unknown`-role turns, no full-call
audio file → `audio_exists=False`, sentence-clip count in `speaker_metadata`),
so the sectioning/text pipeline and the split builder work unchanged. Targets
are computed under both horizon conventions from the shared price archive
(`prices/`; EC tickers missing there are pulled by `ecvol prices pull --dataset
ec`). The lineage's published split (VolTAGE `{train,val,test}_split3.csv`) is
written as `splits/ec_published.csv` in our split-CSV format so the reproduction
can run on the published split and on our embargoed/ticker-disjoint splits with
one code path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from ecvol.data.calls import NAN, CallRecord, write_calls_parquet, write_metric_csv
from ecvol.data.manifests import make_entry, write_manifest
from ecvol.data.prices import load_close_series
from ecvol.data.targets import (
    CONVENTIONS,
    HORIZONS,
    TARGET_FILES,
    compute_call_targets,
    write_targets_parquet,
)

SOURCE = "ec"
DATASET_REL = "raw/ec/extracted/ACL19_Release"
LINEAGE_DATA = "raw/ref/lineage/VolTAGE/semi-supervised-gcn/gcn/data"
FOLDER_RE = re.compile(r"^(.+)_(\d{4})(\d{2})(\d{2})$")
MIN_TRANSCRIPT_CHARS = 50

CALLS_LICENSE = "Derived artifact — normalized from the EC release (Qin & Yang 2019; research use)"
CALLS_SOURCE = "computed: ecvol data ingest ec (T6R.2)"
TARGETS_LICENSE = "Derived artifact — computed from price data (DESIGN §5.3); no external source"
TARGETS_SOURCE = "computed: ecvol data ingest ec / targets (DESIGN §5.3)"


# --- pure parsing ------------------------------------------------------------


def parse_folder_name(name: str) -> tuple[str, str] | None:
    """`CompanyName_YYYYMMDD` → (company, ISO date); None if the suffix is not a date."""
    m = FOLDER_RE.match(name)
    if not m:
        return None
    return m.group(1), f"{m.group(2)}-{m.group(3)}-{m.group(4)}"


def read_sentences(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip()
    ]


def lineage_tickers(root: Path) -> dict[str, str]:
    """folder name → ticker from VolTAGE's `full_stock_data.csv` (upper-cased)."""
    d = pd.read_csv(root / LINEAGE_DATA / "full_stock_data.csv", dtype=str)
    return {r.text_file_name: r.ticker.strip().upper() for r in d.itertuples()}


def published_split(root: Path) -> pd.DataFrame:
    """VolTAGE's train/val/test files → rows (call_id, ticker, split)."""
    frames = []
    for s in ("train", "val", "test"):
        d = pd.read_csv(root / LINEAGE_DATA / f"{s}_split3.csv", dtype=str)
        frames.append(
            pd.DataFrame(
                {"call_id": d["text_file_name"], "ticker": d["ticker"].str.upper(), "split": s}
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- record assembly ---------------------------------------------------------


def build_records(dataset_dir: Path, tickers: dict[str, str]) -> list[CallRecord]:
    records: list[CallRecord] = []
    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            records.append(_excluded(folder.name, "invalid_folder_name"))
            continue
        _, call_date = parsed
        ticker = tickers.get(folder.name, "")
        sentences = read_sentences(folder / "TextSequence.txt")
        turns = [{"role": "unknown", "text": s} for s in sentences]
        n_chars = sum(len(s) for s in sentences)
        n_clips = sum(1 for _ in (folder / "CEO").glob("*.mp3")) if (folder / "CEO").is_dir() else 0
        ok_text = bool(sentences) and n_chars >= MIN_TRANSCRIPT_CHARS
        if not ticker:
            status, reason = "excluded", "unresolved"
        elif not ok_text:
            status, reason = "excluded", "empty_transcript"
        else:
            status, reason = "ok", ""
        meta = {
            "n_turns": len(turns),
            "roles": {"unknown": {"turns": len(turns), "chars": n_chars}},
            "speakers": "unavailable",
            "sentence_audio_clips": n_clips,  # per-sentence CEO mp3s; no full-call file
        }
        records.append(
            CallRecord(
                call_id=folder.name,
                source=SOURCE,
                ticker=ticker,
                call_date=call_date,
                time_known=False,
                assumed_after_hours=True,
                call_type="earnings",
                label=-1,
                n_turns=len(turns),
                n_chars=n_chars,
                transcript_json=json.dumps(turns, ensure_ascii=False),
                speaker_metadata=json.dumps(meta, ensure_ascii=False),
                audio_path="",
                audio_exists=False,
                audio_duration_sec=NAN,
                parsed=ok_text,
                status=status,
                reason=reason,
            )
        )
    return records


def _excluded(call_id: str, reason: str) -> CallRecord:
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


def compute_targets(records: list[CallRecord], prices_dir: Path, *, horizons, convention):
    close_cache: dict[str, dict[str, float]] = {}
    rows = []
    for r in records:
        if r.status != "ok":
            continue
        if r.ticker not in close_cache:
            close_cache[r.ticker] = load_close_series(prices_dir, r.ticker)
        call = {
            "call_id": r.call_id,
            "ticker": r.ticker,
            "date": r.call_date,
            "call_type": r.call_type,
        }
        rows.extend(
            compute_call_targets(
                call, close_cache[r.ticker], horizons=horizons, convention=convention
            )
        )
    return rows


# --- orchestration -----------------------------------------------------------


@dataclass
class EcSummary:
    total_calls: int
    ok: int
    reason_counts: dict[str, int]
    joined: int
    join_rate_pct: float
    missing_price_tickers: int
    published_split_calls: int


# EC tickers (2017) that Yahoo now serves only under a successor symbol; the parquet
# keeps the historical ticker so the lineage's ids join unchanged.
TICKER_ALIASES = {
    "ABC": "COR",  # AmerisourceBergen → Cencora
    "ANTM": "ELV",  # Anthem → Elevance
    "BLL": "BALL",  # Ball Corp
    "CTL": "LUMN",  # CenturyLink → Lumen
    "FB": "META",  # Facebook → Meta
    "FBHS": "FBIN",  # Fortune Brands
    "GPS": "GAP",  # Gap
    "PKI": "RVTY",  # PerkinElmer → Revvity
    "CBS": "PARA",  # CBS → ViacomCBS → Paramount
    "WRK": "SW",  # WestRock → Smurfit Westrock
    "RHI.F": "RHI",  # Robert Half (lineage typo)
    "ADS": "BFH",  # Alliance Data → Bread Financial
}


def ec_tickers(root: Path) -> list[str]:
    """Resolved EC tickers (for the price pull)."""
    return sorted(set(lineage_tickers(root).values()))


def ingest_ec(root: Path, *, horizons=HORIZONS) -> EcSummary:
    dataset_dir = root / DATASET_REL
    if not dataset_dir.is_dir():
        raise ValueError(f"EC not found at {dataset_dir} — see JOURNAL 2026-09-03")
    tickers = lineage_tickers(root)
    records = build_records(dataset_dir, tickers)
    out_dir = root / SOURCE
    write_calls_parquet(records, out_dir / "calls.parquet", id_type=pa.string())

    prices_dir = root / "prices"
    target_paths = {c: out_dir / TARGET_FILES[c] for c in CONVENTIONS}
    target_rows = {
        c: compute_targets(records, prices_dir, horizons=horizons, convention=c)
        for c in CONVENTIONS
    }
    for c in CONVENTIONS:
        write_targets_parquet(target_rows[c], target_paths[c], id_type=pa.string())

    ok = [r for r in records if r.status == "ok"]
    joined_ids = {r.call_id for r in target_rows["trading"] if r.status == "ok"}
    joined = sum(1 for r in ok if r.call_id in joined_ids)
    missing = {r.ticker for r in ok if not (prices_dir / f"{r.ticker}.parquet").is_file()}

    # published split in our CSV format (as_of = day0 from the trading targets)
    as_of = {r.call_id: r.as_of for r in target_rows["trading"] if r.horizon == horizons[0]}
    pub = published_split(root)
    pub["as_of"] = pub["call_id"].map(as_of).fillna("")
    pub = pub[pub["call_id"].isin({r.call_id for r in ok})]
    (root / "splits").mkdir(parents=True, exist_ok=True)
    pub.sort_values("call_id")[["call_id", "ticker", "as_of", "split"]].to_csv(
        root / "splits" / "ec_published.csv", index=False, lineterminator="\n"
    )

    reason_counts: dict[str, int] = {}
    for r in records:
        if r.reason:
            reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1
    target_reasons: dict[str, int] = {}
    for r in target_rows["trading"]:
        if r.reason:
            target_reasons[r.reason] = target_reasons.get(r.reason, 0) + 1
    summary = EcSummary(
        total_calls=len(records),
        ok=len(ok),
        reason_counts=dict(sorted(reason_counts.items())),
        joined=joined,
        join_rate_pct=round(100 * joined / len(ok), 2) if ok else 0.0,
        missing_price_tickers=len(missing),
        published_split_calls=int(len(pub)),
    )
    cov = root / "coverage"
    write_metric_csv(
        [
            ("total_calls", summary.total_calls),
            ("ok_calls", summary.ok),
            ("joined_calls", joined),
            ("join_rate_pct", summary.join_rate_pct),
            ("missing_price_tickers", len(missing)),
            ("published_split_calls", summary.published_split_calls),
        ]
        + [(f"reason:{k}", v) for k, v in summary.reason_counts.items()]
        + [(f"target_reason:{k}", v) for k, v in sorted(target_reasons.items())],
        cov / "ec_ingest_report.csv",
    )
    write_metric_csv(
        [(t, "no_price_data") for t in sorted(missing)], cov / "ec_missing_tickers.csv"
    )
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest(
        [
            make_entry(
                out_dir / "calls.parquet", root, source_url=CALLS_SOURCE, license=CALLS_LICENSE
            )
        ],
        root / "manifests" / "ec_calls.json",
    )
    write_manifest(
        [
            make_entry(target_paths[c], root, source_url=TARGETS_SOURCE, license=TARGETS_LICENSE)
            for c in CONVENTIONS
        ],
        root / "manifests" / "ec_targets.json",
    )
    return summary
