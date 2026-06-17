"""Price ingestion: adjusted daily OHLCV for the call universe (T1.2).

Primary source is **yfinance** (DESIGN §5.2 named Stooq, but Stooq closed free
programmatic access in 2026 — see DECISIONS.md 2026-06-15; yfinance's known
drift risk is mitigated by the coverage report + manifests here and the Tiingo
cross-check in `tiingo.py`). Tiingo remains the cross-check source.

The universe is the union of resolved FinCall tickers (T1.4 identity CSV) and
MAEC tickers (folder names `YYYYMMDD_TICKER`). Every excluded ticker lands in the
coverage report with a reason code — **never silently dropped** (DESIGN §5.2).
Per-ticker parquets are gitignored payloads; the coverage report + manifest are
committed so the provenance and the documented shortfall travel with the repo.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

from ecvol.data import calendar as cal
from ecvol.data.manifests import make_entry, sha256_file, write_manifest

# Fetch window: generous padding around both corpora (MAEC 2015–2018, FinCall
# 2019–2021) so the longest pre-call (30-session) and post-call (30-session)
# target windows are always covered. Fixed → deterministic re-pulls.
START = date(2014, 6, 1)
END = date(2022, 6, 30)

PRICE_LICENSE = "Yahoo Finance — research use; not redistributed (manifest records provenance only)"
PRICE_COLUMNS = ("date", "open", "high", "low", "close", "volume")
COMPLETENESS_FLAG = 0.95  # below this fraction of expected sessions → quality flag

FINCALL_IDENTITY = "identity/fincall_identity.csv"
MAEC_DATASET_DIR = "raw/maec/repo/MAEC_Dataset"


# --- universe ----------------------------------------------------------------


@dataclass
class UniverseEntry:
    ticker: str
    in_fincall: bool = False
    fincall_calls: int = 0
    in_maec: bool = False
    maec_calls: int = 0


def to_yahoo_symbol(ticker: str) -> str:
    """Map a universe ticker to its Yahoo symbol (share classes use '-', not '.')."""
    return ticker.strip().upper().replace(".", "-").replace("/", "-")


def _fincall_tickers(root: Path) -> dict[str, int]:
    """Resolved FinCall tickers → call count (rows with a non-empty ticker)."""
    path = root / FINCALL_IDENTITY
    if not path.is_file():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    resolved = df[df["ticker"].str.strip() != ""]
    return resolved["ticker"].str.strip().str.upper().value_counts().to_dict()


def _maec_tickers(root: Path) -> dict[str, int]:
    """MAEC tickers parsed from `YYYYMMDD_TICKER` folder names → call count."""
    base = root / MAEC_DATASET_DIR
    if not base.is_dir():
        return {}
    counts: dict[str, int] = {}
    for child in base.iterdir():
        if not child.is_dir() or "_" not in child.name:
            continue
        stamp, _, ticker = child.name.rpartition("_")
        if len(stamp) == 8 and stamp.isdigit() and ticker:
            key = ticker.strip().upper()
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_universe(root: Path) -> dict[str, UniverseEntry]:
    """Combined FinCall + MAEC ticker universe with per-dataset call counts."""
    universe: dict[str, UniverseEntry] = {}
    for ticker, n in _fincall_tickers(root).items():
        universe.setdefault(ticker, UniverseEntry(ticker)).in_fincall = True
        universe[ticker].fincall_calls = n
    for ticker, n in _maec_tickers(root).items():
        entry = universe.setdefault(ticker, UniverseEntry(ticker))
        entry.in_maec = True
        entry.maec_calls = n
    return dict(sorted(universe.items()))


# --- fetch + cache -----------------------------------------------------------


def _frame_to_rows(df: pd.DataFrame) -> list[dict]:
    """yfinance OHLCV frame → sorted, deterministic row dicts (NaN rows dropped)."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    rows = []
    for ts, r in df.sort_index().iterrows():
        vol = r.get("Volume")
        rows.append(
            {
                "date": pd.Timestamp(ts).date().isoformat(),
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": int(vol) if pd.notna(vol) else 0,
            }
        )
    return rows


def write_price_parquet(rows: list[dict], path: Path) -> None:
    """Write OHLCV rows as deterministic parquet (same rows → same bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "date": pa.array([r["date"] for r in rows], pa.string()),
        "open": pa.array([r["open"] for r in rows], pa.float64()),
        "high": pa.array([r["high"] for r in rows], pa.float64()),
        "low": pa.array([r["low"] for r in rows], pa.float64()),
        "close": pa.array([r["close"] for r in rows], pa.float64()),
        "volume": pa.array([r["volume"] for r in rows], pa.int64()),
    }
    table = pa.table(arrays)
    pq.write_table(table, path, compression="none", store_schema=True)


def load_close_series(prices_dir: Path, ticker: str) -> dict[str, float]:
    """Adjusted close by ISO date for a cached ticker (`{}` if no parquet).

    The price I/O counterpart used by `targets.py`; mirrors `tiingo._parquet_close`
    but keyed by `(prices_dir, ticker)` so callers don't construct paths themselves.
    """
    path = prices_dir / f"{ticker}.parquet"
    if not path.is_file():
        return {}
    table = pq.read_table(path, columns=["date", "close"])
    dates = table.column("date").to_pylist()
    closes = table.column("close").to_pylist()
    return dict(zip(dates, closes, strict=True))


def fetch_batch(symbols: list[str], start: date, end: date) -> dict[str, list[dict]]:
    """Download a batch of Yahoo symbols → {symbol: rows}. Empty list = no data."""
    if not symbols:
        return {}
    data = yf.download(
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
        actions=False,
    )
    out: dict[str, list[dict]] = {}
    for sym in symbols:
        try:
            sub = data[sym] if len(symbols) > 1 else data
        except (KeyError, TypeError):
            out[sym] = []
            continue
        out[sym] = _frame_to_rows(sub) if sub is not None and not sub.empty else []
    return out


# --- coverage report ---------------------------------------------------------


@dataclass
class CoverageRow:
    ticker: str
    yahoo_symbol: str
    status: str  # "ok" | "missing"
    reason: str
    n_rows: int
    first_date: str
    last_date: str
    expected_sessions: int
    completeness: float
    in_fincall: bool
    fincall_calls: int
    in_maec: bool
    maec_calls: int


@dataclass
class CoverageSummary:
    universe_total: int = 0
    fincall_total: int = 0
    maec_total: int = 0
    covered_total: int = 0
    covered_fincall: int = 0
    covered_maec: int = 0
    fincall_coverage_pct: float = 0.0
    maec_coverage_pct: float = 0.0
    combined_coverage_pct: float = 0.0
    missing_tickers: list[str] = field(default_factory=list)
    low_completeness: list[str] = field(default_factory=list)


def _coverage_row(entry: UniverseEntry, prices_dir: Path) -> CoverageRow:
    path = prices_dir / f"{entry.ticker}.parquet"
    yahoo = to_yahoo_symbol(entry.ticker)
    base = dict(
        ticker=entry.ticker,
        yahoo_symbol=yahoo,
        in_fincall=entry.in_fincall,
        fincall_calls=entry.fincall_calls,
        in_maec=entry.in_maec,
        maec_calls=entry.maec_calls,
    )
    if not path.is_file():
        return CoverageRow(
            status="missing",
            reason="no_data:delisted_renamed_or_unlisted_on_yahoo",
            n_rows=0,
            first_date="",
            last_date="",
            expected_sessions=0,
            completeness=0.0,
            **base,
        )
    dates = pq.read_table(path, columns=["date"]).column("date").to_pylist()
    first, last = dates[0], dates[-1]
    expected = cal.session_count(date.fromisoformat(first), date.fromisoformat(last))
    completeness = round(len(dates) / expected, 4) if expected else 0.0
    return CoverageRow(
        status="ok",
        reason="gappy" if completeness < COMPLETENESS_FLAG else "",
        n_rows=len(dates),
        first_date=first,
        last_date=last,
        expected_sessions=expected,
        completeness=completeness,
        **base,
    )


def _pct(num: int, denom: int) -> float:
    return round(100 * num / denom, 2) if denom else 0.0


def build_coverage(
    universe: dict[str, UniverseEntry], prices_dir: Path
) -> tuple[list[CoverageRow], CoverageSummary]:
    rows = [_coverage_row(e, prices_dir) for e in universe.values()]
    s = CoverageSummary(universe_total=len(rows))
    s.fincall_total = sum(1 for r in rows if r.in_fincall)
    s.maec_total = sum(1 for r in rows if r.in_maec)
    ok = [r for r in rows if r.status == "ok"]
    s.covered_total = len(ok)
    s.covered_fincall = sum(1 for r in ok if r.in_fincall)
    s.covered_maec = sum(1 for r in ok if r.in_maec)
    s.fincall_coverage_pct = _pct(s.covered_fincall, s.fincall_total)
    s.maec_coverage_pct = _pct(s.covered_maec, s.maec_total)
    s.combined_coverage_pct = _pct(s.covered_total, s.universe_total)
    s.missing_tickers = sorted(r.ticker for r in rows if r.status == "missing")
    s.low_completeness = sorted(r.ticker for r in ok if r.reason == "gappy")
    return rows, s


def write_coverage(rows: list[CoverageRow], path: Path) -> None:
    """Write the coverage report as a deterministic, sorted CSV (committed artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("ticker")
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    path.write_text(buf.getvalue(), encoding="utf-8")


# --- orchestration -----------------------------------------------------------


def pull_prices(
    root: Path,
    *,
    refresh: bool = False,
    batch_size: int = 100,
    start: date = START,
    end: date = END,
) -> CoverageSummary:
    """Pull adjusted daily OHLCV for the whole universe; cache, manifest, report.

    Idempotent/resumable: a ticker whose parquet already exists is skipped unless
    `refresh`. Missing data is recorded in the coverage report, never dropped.
    """
    universe = build_universe(root)
    if not universe:
        raise FileNotFoundError(
            f"empty universe — expected {FINCALL_IDENTITY} and/or {MAEC_DATASET_DIR} under {root}"
        )
    prices_dir = root / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(root)
    pending = [t for t in universe if refresh or not (prices_dir / f"{t}.parquet").exists()]

    # Primary pass — yfinance (batched).
    sym_to_ticker = {to_yahoo_symbol(t): t for t in pending}
    symbols = list(sym_to_ticker)
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        for sym, rows in fetch_batch(batch, start, end).items():
            if rows:
                ticker = sym_to_ticker[sym]
                write_price_parquet(rows, prices_dir / f"{ticker}.parquet")
                sources[ticker] = "yahoo"

    # Fallback pass — Tiingo recovers tickers Yahoo has purged (delisted/acquired),
    # but only if a key is available. Inert (and the shortfall is documented) otherwise.
    # Scoped to the FinCall universe: the ≥98% gate binds on FinCall only (DECISIONS
    # 2026-06-15) and the free Tiingo tier is 50 req/hr · 500 symbols/month — attempting
    # all ~310 misses (mostly MAEC) would rate-limit and silently drop. MAEC Tiingo
    # recovery is deferred to T1.5 (DECISIONS 2026-06-17).
    still_missing = [
        t
        for t, entry in universe.items()
        if entry.in_fincall and not (prices_dir / f"{t}.parquet").exists()
    ]
    if still_missing:
        _tiingo_fallback(still_missing, prices_dir, root, sources, start, end)

    _write_sources(root, sources)
    rows, summary = build_coverage(universe, prices_dir)
    write_coverage(rows, root / "coverage" / "prices_coverage.csv")
    _write_manifest(prices_dir, root, sources)
    return summary


def _tiingo_fallback(
    tickers: list[str],
    prices_dir: Path,
    root: Path,
    sources: dict[str, str],
    start: date,
    end: date,
) -> None:
    from ecvol.data.tiingo import fetch_tiingo_ohlcv, load_api_key  # avoid import cycle

    key = load_api_key()
    if not key:
        return
    for ticker in tickers:
        try:
            rows = fetch_tiingo_ohlcv(ticker, key, start, end)
        except Exception:  # noqa: BLE001 — one bad ticker must not abort the pull
            continue
        if rows:
            write_price_parquet(rows, prices_dir / f"{ticker}.parquet")
            sources[ticker] = "tiingo"


def _source_url(ticker: str, source: str) -> str:
    sym = to_yahoo_symbol(ticker)
    if source == "tiingo":
        return f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
    return f"https://finance.yahoo.com/quote/{sym}/history"


def _sources_path(root: Path) -> Path:
    return root / "coverage" / "prices_sources.json"


def _load_sources(root: Path) -> dict[str, str]:
    path = _sources_path(root)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_sources(root: Path, sources: dict[str, str]) -> None:
    path = _sources_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(sources.items())), indent=2) + "\n", encoding="utf-8")


def _write_manifest(prices_dir: Path, root: Path, sources: dict[str, str]) -> None:
    entries = []
    for parquet in sorted(prices_dir.glob("*.parquet")):
        ticker = parquet.stem
        entries.append(
            make_entry(
                parquet,
                root,
                source_url=_source_url(ticker, sources.get(ticker, "yahoo")),
                license=PRICE_LICENSE,
            )
        )
    if entries:
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        write_manifest(entries, root / "manifests" / "prices.json")


# `sha256_file` re-exported for callers that verify a single price file.
__all__ = [
    "build_universe",
    "build_coverage",
    "pull_prices",
    "to_yahoo_symbol",
    "write_price_parquet",
    "load_close_series",
    "sha256_file",
]
