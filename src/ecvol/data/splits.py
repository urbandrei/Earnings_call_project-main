"""Leakage-proof split construction — DESIGN §5.4 (T1.6).

Three split schemes, built per dataset over its **modeling cohort** (calls with
≥1 ok target row in `{dataset}/targets.parquet`):

- **temporal** — train < val < test in calendar order by the call's `as_of`
  (day-0) session, with a 30-trading-day **embargo** before each boundary so no
  call's target window (up to the longest horizon, 30 sessions) crosses it.
  Calls in the embargo zone are dropped (`split="embargo"`), never silently.
- **ticker_disjoint** — a seeded partition of *tickers* into train/val/test
  (≈70/10/20 by call count), so a company in val/test never appears in train.
  The headline identity-control split. (Sector stratification, DESIGN §5.4.2's
  "where metadata allows", is skipped — neither corpus carries sector metadata.)
- **combined** — temporal × ticker_disjoint: a call keeps a split only when both
  schemes agree, else it's `excluded`. The hardest condition.

Splits are **committed CSVs** (`data/splits/{dataset}_{scheme}.csv`), generated
once by `ecvol splits build` and stable across reruns; the leakage assertions in
`tests/test_splits.py` run in CI forever after (DESIGN §5.4.6).
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from ecvol.data import calendar as cal
from ecvol.data.calls import write_metric_csv

DATASETS = ("fincall", "maec")
DEFAULT_RATIOS = (0.70, 0.10, 0.20)  # train / val / test (DESIGN §5.4)
DEFAULT_EMBARGO = 30  # trading days; ≥ longest horizon (DESIGN §5.4.1)
DEFAULT_SEED = 0
SPLITS = ("train", "val", "test")


@dataclass
class CallInfo:
    call_id: int | str
    ticker: str
    as_of: str  # ISO date of day0
    idx: int  # NYSE session index of as_of


# --- cohort + session indexing -----------------------------------------------


def load_cohort(targets_path: Path) -> tuple[list[CallInfo], int]:
    """Cohort (one CallInfo per call with ≥1 ok target) + the longest ok horizon.

    `as_of` is identical across a call's horizon rows; `idx` is filled later.
    """
    table = pq.read_table(targets_path, columns=["call_id", "ticker", "as_of", "horizon", "status"])
    call_id = table.column("call_id").to_pylist()
    ticker = table.column("ticker").to_pylist()
    as_of = table.column("as_of").to_pylist()
    horizon = table.column("horizon").to_pylist()
    status = table.column("status").to_pylist()

    best: dict[object, tuple[str, str]] = {}
    max_h = 0
    for cid, tk, ao, h, st in zip(call_id, ticker, as_of, horizon, status, strict=True):
        if st != "ok":
            continue
        best.setdefault(cid, (tk, ao))
        max_h = max(max_h, h)
    cohort = [CallInfo(cid, tk, ao, -1) for cid, (tk, ao) in best.items()]
    return cohort, max_h


def assign_session_indices(cohort: list[CallInfo]) -> None:
    """Fill `idx` with each call's NYSE session index (mutates in place)."""
    if not cohort:
        return
    days = [date.fromisoformat(c.as_of) for c in cohort]
    sessions = cal.sessions_in_range(min(days), max(days))
    index = {d: i for i, d in enumerate(sessions)}
    for c, d in zip(cohort, days, strict=True):
        if d not in index:  # as_of is always a session (targets.py), so this is a hard error
            raise ValueError(f"as_of {c.as_of} for call {c.call_id} is not a NYSE session")
        c.idx = index[d]


# --- the three schemes -------------------------------------------------------


def _boundaries(cohort: list[CallInfo], ratios: tuple[float, float, float]) -> tuple[int, int]:
    """Session-index boundaries b1, b2 at the train / train+val call-count quantiles."""
    idxs = sorted(c.idx for c in cohort)
    n = len(idxs)
    b1 = idxs[min(n - 1, int(ratios[0] * n))]
    b2 = idxs[min(n - 1, int((ratios[0] + ratios[1]) * n))]
    return b1, b2


def temporal_split(
    cohort: list[CallInfo],
    *,
    embargo: int,
    horizon: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[object, str]:
    """Assign train/val/test/embargo by `as_of` session, embargoing each boundary.

    A call is dropped to `embargo` when it sits within `gap` sessions on the train
    side of a boundary — `gap = max(embargo, horizon)` guarantees no target window
    (length `horizon`) crosses, and a ≥`gap`-session hole between segments.
    """
    if not cohort:
        return {}
    b1, b2 = _boundaries(cohort, ratios)
    gap = max(embargo, horizon)
    out: dict[object, str] = {}
    for c in cohort:
        if c.idx <= b1 - gap:
            out[c.call_id] = "train"
        elif b1 < c.idx <= b2 - gap:
            out[c.call_id] = "val"
        elif c.idx > b2:
            out[c.call_id] = "test"
        else:
            out[c.call_id] = "embargo"
    return out


def ticker_disjoint_split(
    cohort: list[CallInfo],
    *,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[object, str]:
    """Partition tickers (seeded) into train/val/test ≈ ratios by call count."""
    if not cohort:
        return {}
    counts: dict[str, int] = {}
    for c in cohort:
        counts[c.ticker] = counts.get(c.ticker, 0) + 1
    tickers = sorted(counts)  # deterministic order before the seeded shuffle
    random.Random(seed).shuffle(tickers)

    total = len(cohort)
    train_cap = ratios[0] * total
    val_cap = (ratios[0] + ratios[1]) * total
    ticker_split: dict[str, str] = {}
    cum = 0
    for t in tickers:
        if cum < train_cap:
            ticker_split[t] = "train"
        elif cum < val_cap:
            ticker_split[t] = "val"
        else:
            ticker_split[t] = "test"
        cum += counts[t]
    return {c.call_id: ticker_split[c.ticker] for c in cohort}


def combined_split(
    temporal: dict[object, str], ticker_disjoint: dict[object, str]
) -> dict[object, str]:
    """Keep a call's split only where both schemes agree, else `excluded`."""
    out: dict[object, str] = {}
    for cid, t in temporal.items():
        d = ticker_disjoint.get(cid)
        out[cid] = t if (t == d and t in SPLITS) else "excluded"
    return out


# --- artifacts ---------------------------------------------------------------


def _write_split_csv(cohort: list[CallInfo], assignment: dict[object, str], path: Path) -> None:
    """Committed split CSV (call_id, ticker, as_of, split), sorted by call_id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        ((c.call_id, c.ticker, c.as_of, assignment[c.call_id]) for c in cohort),
        key=lambda r: str(r[0]),
    )
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["call_id", "ticker", "as_of", "split"])
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _counts(assignment: dict[object, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in assignment.values():
        out[v] = out.get(v, 0) + 1
    return out


@dataclass
class SplitSummary:
    dataset: str
    cohort: int
    horizon: int
    embargo: int
    seed: int
    scheme_counts: dict[str, dict[str, int]]  # scheme -> split -> count


def build_dataset_splits(
    root: Path,
    dataset: str,
    *,
    embargo: int = DEFAULT_EMBARGO,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> SplitSummary:
    """Build all three split CSVs for one dataset from its targets.parquet."""
    targets_path = root / dataset / "targets.parquet"
    if not targets_path.is_file():
        raise ValueError(f"no targets at {targets_path} — run `ecvol data ingest {dataset}` first")
    cohort, horizon = load_cohort(targets_path)
    if not cohort:
        raise ValueError(f"empty cohort for {dataset} (no ok target rows)")
    assign_session_indices(cohort)

    temporal = temporal_split(cohort, embargo=embargo, horizon=horizon, ratios=ratios)
    ticker = ticker_disjoint_split(cohort, seed=seed, ratios=ratios)
    combined = combined_split(temporal, ticker)
    schemes = {"temporal": temporal, "ticker_disjoint": ticker, "combined": combined}

    splits_dir = root / "splits"
    for scheme, assignment in schemes.items():
        _write_split_csv(cohort, assignment, splits_dir / f"{dataset}_{scheme}.csv")

    scheme_counts = {scheme: _counts(a) for scheme, a in schemes.items()}
    report_rows: list[tuple[str, object]] = [
        ("cohort", len(cohort)),
        ("horizon", horizon),
        ("embargo", embargo),
        ("seed", seed),
    ]
    for scheme in schemes:
        for split in (*SPLITS, "embargo", "excluded"):
            if split in scheme_counts[scheme]:
                report_rows.append((f"{scheme}:{split}", scheme_counts[scheme][split]))
    write_metric_csv(report_rows, root / "coverage" / f"{dataset}_splits_report.csv")

    return SplitSummary(dataset, len(cohort), horizon, embargo, seed, scheme_counts)


def build_splits(
    root: Path,
    *,
    embargo: int = DEFAULT_EMBARGO,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> list[SplitSummary]:
    """Build splits for every dataset whose targets.parquet exists."""
    summaries = []
    for dataset in DATASETS:
        if (root / dataset / "targets.parquet").is_file():
            summaries.append(
                build_dataset_splits(root, dataset, embargo=embargo, seed=seed, ratios=ratios)
            )
    if not summaries:
        raise ValueError("no dataset targets found — run `ecvol data ingest <dataset>` first")
    return summaries
