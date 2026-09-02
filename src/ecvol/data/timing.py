"""Call-timestamp retrofit (T9.2; DECISIONS 2026-08-23 §8, 2026-09-03).

DESIGN §5.3 defines day 0 as the last session *before the call's information is
public*. Until now no corpus carried a time of day, so every target used the
documented assume-after-hours fallback. This module measures the information
boundary per call and records where it came from:

| tier | source | what it gives |
|---|---|---|
| `edgar_8k` | SEC submissions API: earliest 8-K / 8-K/A with Item 2.02 accepted in
  [call date − 3, call date + 1] | the earnings-release timestamp (ET), the boundary itself |
| `earnings25_metadata` | Earnings25 `ReleaseDate` (UTC → ET) | the call start time |
| `dec_flag` | SCSS DEC `beforeAfterMarket` on (ticker, date) | before/after only |
| `assumed_after_hours` | none | the fallback |

The release time outranks the call time because releases routinely precede the
call by hours or a day (a 16:17 ET filing before a next-morning call); the DEC
flag is a coarse check, never an anchor. Output per dataset:
`coverage/{dataset}_timing.csv` (call_id, ticker, call_date, tier, anchor_et,
session_rule, source_ref) — committed, deterministic, one row per resolved call —
which the target builders consume through `load_call_times`. (Earnings25's
ingestion-time `earnings25_call_times.csv` is the raw call-start table this reads.)

Network: `data.sec.gov` fair use (UA header, ≤10 req/s); every response is cached
under `raw/ref/edgar_submissions/` and never re-fetched, so re-runs are offline.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ecvol.collect.discovery import _cik_by_ticker

SEC_SUBMISSIONS = "https://data.sec.gov/submissions/"
SEC_HEADERS = {"User-Agent": "ecvol-research/0.1 (andrei.roman.personal@gmail.com)"}
CACHE_REL = "raw/ref/edgar_submissions"
DEC_REL = "raw/ref/scss/DEC.csv"
ET = ZoneInfo("America/New_York")
WINDOW_BEFORE = 3  # days before the call date an Item 2.02 8-K may be accepted
WINDOW_AFTER = 1
TIERS = ("edgar_8k", "earnings25_metadata", "dec_flag", "assumed_after_hours")


# --- EDGAR ---------------------------------------------------------------------


def fetch_submissions(cik: str, cache_dir: Path, *, sleep: float = 0.12) -> pd.DataFrame:
    """All filings for a CIK (recent + paginated history) as a frame; cached forever."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    main = cache_dir / f"CIK{cik}.json"
    if not main.exists():
        _download(SEC_SUBMISSIONS + f"CIK{cik}.json", main, sleep)
    doc = json.loads(main.read_text(encoding="utf-8"))
    frames = [_filings_frame(doc["filings"]["recent"])]
    for extra in doc["filings"].get("files", []):
        page = cache_dir / extra["name"]
        if not page.exists():
            _download(SEC_SUBMISSIONS + extra["name"], page, sleep)
        frames.append(_filings_frame(json.loads(page.read_text(encoding="utf-8"))))
    return pd.concat(frames, ignore_index=True)


def _download(url: str, target: Path, sleep: float) -> None:
    resp = requests.get(url, headers=SEC_HEADERS, timeout=60)
    resp.raise_for_status()
    tmp = target.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(target)
    time.sleep(sleep)


def _filings_frame(block: dict) -> pd.DataFrame:
    cols = ("form", "filingDate", "acceptanceDateTime", "items", "accessionNumber")
    return pd.DataFrame({k: block.get(k, []) for k in cols})


def release_filing(filings: pd.DataFrame, call_date: date) -> dict | None:
    """Earliest 8-K (or 8-K/A) with Item 2.02 accepted in the window around `call_date`."""
    if filings.empty:
        return None
    f = filings[filings["form"].isin(["8-K", "8-K/A"])].copy()
    f = f[f["items"].fillna("").str.contains("2.02", regex=False)]
    if f.empty:
        return None
    f["fd"] = pd.to_datetime(f["filingDate"]).dt.date
    lo, hi = call_date - timedelta(days=WINDOW_BEFORE), call_date + timedelta(days=WINDOW_AFTER)
    f = f[(f["fd"] >= lo) & (f["fd"] <= hi)].sort_values("acceptanceDateTime")
    if f.empty:
        return None
    row = f.iloc[0]
    accepted = datetime.fromisoformat(str(row["acceptanceDateTime"]).replace("Z", "+00:00"))
    return {
        "anchor_et": accepted.astimezone(ET).replace(tzinfo=None),
        "accession": str(row["accessionNumber"]),
        "form": str(row["form"]),
    }


# --- the day-0 rule on a measured timestamp -----------------------------------


def session_rule(anchor_et: datetime, call_date: date) -> str:
    """How DESIGN §5.3 reads a measured ET timestamp relative to the call date.

    `before_open`: public before 09:30 on the call date (or on an earlier day) →
    the call date is day +1. `after_close`: at/after 16:00 on the call date → the
    call date is day 0. `intraday`: 09:30–16:00 on the call date → treated as
    after-hours and flagged (DESIGN §5.3), so the reaction session stays in the
    pre window. `after_call_date`: the anchor falls after the call date (a late
    8-K) → treated as after-hours on the call date.
    """
    if anchor_et.date() < call_date:
        return "before_open"
    if anchor_et.date() > call_date:
        return "after_call_date"
    hm = (anchor_et.hour, anchor_et.minute)
    if hm < (9, 30):
        return "before_open"
    if hm >= (16, 0):
        return "after_close"
    return "intraday"


def anchor_timestamp(anchor_et: datetime, call_date: date) -> datetime:
    """The timestamp handed to `targets.anchor_day0`: keep the measured time on the
    call date; collapse earlier-day anchors to 00:00 on the call date (→ before-open
    branch) and later-day anchors to 16:00 on the call date (→ after-hours branch)."""
    if anchor_et.date() < call_date:
        return datetime.combine(call_date, datetime.min.time())
    if anchor_et.date() > call_date:
        return datetime.combine(call_date, datetime.min.time().replace(hour=16))
    return anchor_et


# --- per-dataset tables ----------------------------------------------------------


@dataclass
class TimingRow:
    call_id: str
    ticker: str
    call_date: str
    tier: str
    anchor_et: str  # ISO minutes; "" for dec_flag / fallback
    session_rule: (
        str  # before_open | intraday | after_close | after_call_date | assumed_after_hours
    )
    source_ref: str  # accession number / Earnings25 id / DEC id / ""


def _dec_flags(root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    path = root / DEC_REL
    if not path.is_file():
        return {}
    d = pd.read_csv(path, usecols=["ticker", "day_earnings", "beforeAfterMarket", "id"], dtype=str)
    d["date"] = pd.to_datetime(d["day_earnings"]).dt.date.astype(str)
    return {(r.ticker, r.date): (r.beforeAfterMarket, r.id) for r in d.itertuples()}


def _earnings25_times(root: Path) -> dict[str, str]:
    path = root / "coverage" / "earnings25_call_times.csv"
    if not path.is_file():
        return {}
    t = pd.read_csv(path, dtype=str).fillna("")
    return {r.call_id: r.call_datetime_et for r in t.itertuples() if r.ticker}


def build_call_times(
    root: Path, dataset: str, *, fetch: bool = True, limit: int | None = None
) -> tuple[list[TimingRow], dict[str, int]]:
    """One TimingRow per resolved call of `dataset`; returns (rows, tier counts)."""
    calls = pd.read_parquet(
        root / dataset / "calls.parquet", columns=["call_id", "ticker", "call_date", "status"]
    )
    calls = calls[(calls["ticker"] != "") & (calls["call_date"] != "")].copy()
    calls["call_id"] = calls["call_id"].astype(str)
    calls = calls.sort_values("call_id").reset_index(drop=True)
    if limit is not None:
        calls = calls.head(limit)
    cik_by_ticker = _cik_by_ticker(root)
    cache_dir = root / CACHE_REL
    dec = _dec_flags(root)
    e25 = _earnings25_times(root) if dataset == "earnings25" else {}

    filings_cache: dict[str, pd.DataFrame] = {}
    rows: list[TimingRow] = []
    for r in calls.itertuples(index=False):
        cdate = date.fromisoformat(r.call_date)
        cik = cik_by_ticker.get(r.ticker, "")
        hit = None
        if cik:
            if cik not in filings_cache:
                cached = (cache_dir / f"CIK{cik}.json").exists()
                filings_cache[cik] = (
                    fetch_submissions(cik, cache_dir) if (fetch or cached) else pd.DataFrame()
                )
            hit = release_filing(filings_cache[cik], cdate)
        if hit is not None:
            a = hit["anchor_et"]
            rows.append(
                TimingRow(
                    r.call_id,
                    r.ticker,
                    r.call_date,
                    "edgar_8k",
                    a.isoformat(timespec="minutes"),
                    session_rule(a, cdate),
                    hit["accession"],
                )
            )
            continue
        if r.call_id in e25:
            a = datetime.fromisoformat(e25[r.call_id])
            rows.append(
                TimingRow(
                    r.call_id,
                    r.ticker,
                    r.call_date,
                    "earnings25_metadata",
                    a.isoformat(timespec="minutes"),
                    session_rule(a, cdate),
                    r.call_id,
                )
            )
            continue
        flag = dec.get((r.ticker, r.call_date))
        if flag is not None:
            rule = "before_open" if flag[0] == "BeforeMarket" else "after_close"
            rows.append(TimingRow(r.call_id, r.ticker, r.call_date, "dec_flag", "", rule, flag[1]))
            continue
        rows.append(
            TimingRow(
                r.call_id,
                r.ticker,
                r.call_date,
                "assumed_after_hours",
                "",
                "assumed_after_hours",
                "",
            )
        )
    counts = {t: sum(1 for x in rows if x.tier == t) for t in TIERS}
    return rows, counts


def write_call_times(rows: list[TimingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            ["call_id", "ticker", "call_date", "tier", "anchor_et", "session_rule", "source_ref"]
        )
        for r in sorted(rows, key=lambda r: r.call_id):
            w.writerow(
                [
                    r.call_id,
                    r.ticker,
                    r.call_date,
                    r.tier,
                    r.anchor_et,
                    r.session_rule,
                    r.source_ref,
                ]
            )


def load_call_times(path: Path) -> dict[str, datetime]:
    """{call_id: timestamp for `targets.anchor_day0`} for the measured tiers.

    `dec_flag` rows become 00:00 (before) / 16:00 (after) on the call date; the
    fallback tier is absent so callers keep assuming after-hours for it.
    """
    if not path.is_file():
        return {}
    out: dict[str, datetime] = {}
    t = pd.read_csv(path, dtype=str).fillna("")
    for r in t.itertuples():
        cdate = date.fromisoformat(r.call_date)
        if r.anchor_et:
            out[r.call_id] = anchor_timestamp(datetime.fromisoformat(r.anchor_et), cdate)
        elif r.tier == "dec_flag":
            hour = 0 if r.session_rule == "before_open" else 16
            out[r.call_id] = datetime.combine(cdate, datetime.min.time().replace(hour=hour))
    return out


# --- measured-anchor targets (variant) + sensitivity ---------------------------

MEASURED_FILES = {
    "trading": "targets_measured.parquet",
    "calendar": "targets_measured_calendar.parquet",
}
PRICE_DIRS = {"fincall": "prices", "maec": "prices", "earnings25": "prices_earnings25"}


def build_measured_targets(root: Path, dataset: str, *, horizons=None) -> dict[str, int]:
    """Targets for `dataset` with day 0 anchored on the measured timestamp where one
    exists (`coverage/{dataset}_timing.csv`), the fallback elsewhere — written beside
    the primary files as `targets_measured{,_calendar}.parquet` + manifest.

    Returns {"calls": n, "measured": n_with_timestamp, "ok_rows": n}.
    """
    import pyarrow as pa

    from ecvol.data.manifests import make_entry, write_manifest
    from ecvol.data.prices import load_close_series
    from ecvol.data.targets import (
        CONVENTIONS,
        HORIZONS,
        TARGETS_LICENSE,
        compute_call_targets,
        write_targets_parquet,
    )

    horizons = horizons or HORIZONS
    stamps = load_call_times(root / "coverage" / f"{dataset}_timing.csv")
    calls = pd.read_parquet(
        root / dataset / "calls.parquet", columns=["call_id", "ticker", "call_date", "call_type"]
    )
    calls = calls[(calls["ticker"] != "") & (calls["call_date"] != "")]
    id_type = pa.int64() if dataset == "fincall" else pa.string()
    prices_dir = root / PRICE_DIRS[dataset]
    close_cache: dict[str, dict[str, float]] = {}
    entries = []
    ok_rows = 0
    for convention in CONVENTIONS:
        rows = []
        for r in calls.itertuples(index=False):
            if r.ticker not in close_cache:
                close_cache[r.ticker] = load_close_series(prices_dir, r.ticker)
            call = {
                "call_id": r.call_id,
                "ticker": r.ticker,
                "date": r.call_date,
                "call_type": r.call_type,
                "timestamp": stamps.get(str(r.call_id)),
            }
            rows.extend(
                compute_call_targets(
                    call, close_cache[r.ticker], horizons=horizons, convention=convention
                )
            )
        path = root / dataset / MEASURED_FILES[convention]
        write_targets_parquet(rows, path, id_type=id_type)
        if convention == "trading":
            ok_rows = sum(1 for x in rows if x.status == "ok")
        entries.append(
            make_entry(
                path,
                root,
                source_url=f"computed: ecvol timing targets {dataset} (DECISIONS 2026-09-03)",
                license=TARGETS_LICENSE,
            )
        )
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest(entries, root / "manifests" / f"{dataset}_targets_measured.json")
    measured = sum(1 for c in calls["call_id"].astype(str) if c in stamps)
    return {"calls": int(len(calls)), "measured": int(measured), "ok_rows": int(ok_rows)}


def sensitivity(root: Path, datasets=("fincall", "maec", "earnings25")) -> pd.DataFrame:
    """Fallback vs measured anchors: how many day-0 anchors move, and what the Stage-0
    baselines (persistence, train-fit HAR) do on the test segment under each anchor.

    Rows: (dataset, anchor, split, target, horizon, model) with n / mse / r2_oos, plus
    (dataset, anchor="shift") rows giving the share of calls whose `as_of` moved.
    Uses the committed splits (defined on the fallback cohort) for both anchors.
    """
    from ecvol.eval import evaluate as E
    from ecvol.eval import metrics as M
    from ecvol.models import baselines as B

    rows = []
    for dataset in datasets:
        primary = root / dataset / "targets.parquet"
        measured = root / dataset / MEASURED_FILES["trading"]
        if not (primary.is_file() and measured.is_file()):
            continue
        frames = {}
        for anchor, path in (("fallback", primary), ("measured", measured)):
            t = pd.read_parquet(path)
            t = t[t["status"] == "ok"].copy()
            t["call_id"] = t["call_id"].astype(str)
            frames[anchor] = t
        a, b = frames["fallback"], frames["measured"]
        j = a[["call_id", "horizon", "as_of"]].merge(
            b[["call_id", "horizon", "as_of"]], on=["call_id", "horizon"], suffixes=("_f", "_m")
        )
        j = j[j["horizon"] == j["horizon"].max()]
        rows.append(
            {
                "dataset": dataset,
                "anchor": "shift",
                "split": "",
                "target": "",
                "horizon": int(j["horizon"].max()) if len(j) else 0,
                "model": "as_of_moved_share",
                "n": int(len(j)),
                "mse": float("nan"),
                "r2_oos": float((j["as_of_f"] != j["as_of_m"]).mean()) if len(j) else float("nan"),
            }
        )
        for scheme in ("temporal", "ticker_disjoint"):
            split_csv = root / "splits" / f"{dataset}_{scheme}.csv"
            if not split_csv.is_file():
                continue
            assign = pd.read_csv(split_csv, dtype={"call_id": str}).set_index("call_id")["split"]
            for anchor, t in frames.items():
                for tau in E.HORIZONS:
                    at = t[t["horizon"] == tau].copy()
                    at["split"] = at["call_id"].map(assign).fillna("excluded")
                    tr = at[at["split"] == "train"]
                    te = at[at["split"] == "test"]
                    if len(tr) < 10 or len(te) == 0:
                        continue
                    coef = B.har_fit(
                        B.har_design(tr["rv_daily"], tr["rv_weekly"], tr["rv_monthly"]),
                        tr["v_post"].to_numpy(),
                    )
                    har = B.har_predict(
                        B.har_design(te["rv_daily"], te["rv_weekly"], te["rv_monthly"]), coef
                    )
                    for target in ("v", "dv"):
                        y = E.target_truth(te, target, har)
                        base = E.persistence_pred(te, target)
                        preds = {
                            "persistence": base,
                            "har": E.vpost_to_target(har, te, target, har),
                        }
                        for model, yp in preds.items():
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "anchor": anchor,
                                    "split": scheme,
                                    "target": target,
                                    "horizon": int(tau),
                                    "model": model,
                                    "n": int(len(te)),
                                    "mse": float(M.mse(y, yp)),
                                    "r2_oos": float(M.r2_oos(y, yp, base)),
                                }
                            )
    table = pd.DataFrame(rows)
    out = root / "results" / "timing_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")
    return table
