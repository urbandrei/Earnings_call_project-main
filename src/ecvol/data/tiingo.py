"""Tiingo cross-check of the primary price pull (T1.2 acceptance gate).

DESIGN §5.2: cross-check adjusted daily returns against Tiingo on a 5% random
ticker sample; **gate: return correlation > 0.999** on the overlap, any ticker
below 0.99 investigated and documented. Needs a free Tiingo API key in
`TIINGO_API_KEY` (env or `.env`); without one the CLI errors actionably.

The correlation math (`return_correlation`) is pure and unit-tested; only
`fetch_tiingo` touches the network.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import requests

from ecvol.data.prices import END, START, to_yahoo_symbol

TIINGO_PRICES_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


def load_api_key(root: Path | None = None) -> str | None:
    """Tiingo key from `TIINGO_API_KEY` env var, falling back to `<root>/.env`.

    `root` defaults to the project root (`.`), where CLAUDE.md keeps secrets.
    """
    key = os.environ.get("TIINGO_API_KEY")
    if key:
        return key.strip()
    env_path = (root or Path(".")) / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "TIINGO_API_KEY":
                return value.strip().strip("\"'")
    return None


def sample_tickers(tickers: list[str], fraction: float = 0.05, seed: int = 0) -> list[str]:
    """Deterministic random sample (≥1 ticker) of the covered universe."""
    pool = sorted(tickers)
    if not pool:
        return []
    k = max(1, round(len(pool) * fraction))
    return sorted(random.Random(seed).sample(pool, min(k, len(pool))))


def _returns(close_by_date: dict[str, float]) -> dict[str, float]:
    """Simple daily returns keyed by the *later* date, over a date→close map."""
    dates = sorted(close_by_date)
    out: dict[str, float] = {}
    for prev, cur in zip(dates, dates[1:], strict=False):
        p = close_by_date[prev]
        if p:
            out[cur] = close_by_date[cur] / p - 1.0
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def return_correlation(a_close: dict[str, float], b_close: dict[str, float]) -> tuple[float, int]:
    """Pearson correlation of daily returns over the two sources' common dates.

    Returns (correlation, n_overlapping_returns).
    """
    ra, rb = _returns(a_close), _returns(b_close)
    common = sorted(set(ra) & set(rb))
    xs = [ra[d] for d in common]
    ys = [rb[d] for d in common]
    return _pearson(xs, ys), len(common)


def _parquet_close(path: Path) -> dict[str, float]:
    table = pq.read_table(path, columns=["date", "close"])
    dates = table.column("date").to_pylist()
    closes = table.column("close").to_pylist()
    return dict(zip(dates, closes, strict=True))


def fetch_tiingo_ohlcv(ticker: str, key: str, start: date = START, end: date = END) -> list[dict]:
    """Tiingo split/dividend-*adjusted* daily OHLCV rows for `ticker`.

    Same row schema as the yfinance path (`prices.write_price_parquet`), so Tiingo
    can serve as a fallback source for tickers Yahoo has purged (delisted/acquired).
    """
    url = TIINGO_PRICES_URL.format(ticker=to_yahoo_symbol(ticker))
    resp = requests.get(
        url,
        params={
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "format": "csv",
            "token": key,
        },
        timeout=60,
    )
    resp.raise_for_status()
    rows: list[dict] = []
    for r in csv.DictReader(io.StringIO(resp.text)):
        d = r.get("date")
        adj_close = r.get("adjClose") or r.get("close")
        if not d or not adj_close:
            continue
        rows.append(
            {
                "date": d[:10],
                "open": float(r.get("adjOpen") or r.get("open") or adj_close),
                "high": float(r.get("adjHigh") or r.get("high") or adj_close),
                "low": float(r.get("adjLow") or r.get("low") or adj_close),
                "close": float(adj_close),
                "volume": int(float(r.get("adjVolume") or r.get("volume") or 0)),
            }
        )
    rows.sort(key=lambda x: x["date"])
    return rows


def fetch_tiingo(ticker: str, key: str, start: date = START, end: date = END) -> dict[str, float]:
    """Tiingo adjusted-close by date for `ticker` (used by the cross-check)."""
    return {r["date"]: r["close"] for r in fetch_tiingo_ohlcv(ticker, key, start, end)}


@dataclass
class CrossCheckRow:
    ticker: str
    correlation: float
    n_overlap: int
    status: str  # "pass" | "warn" | "investigate" | "no_overlap" | "tiingo_missing"


@dataclass
class CrossCheckResult:
    rows: list[CrossCheckRow]
    n_sampled: int
    n_passed: int
    min_correlation: float
    gate_passed: bool
    undocumented: list[str]  # sub-0.99 tickers without a committed reason code


def load_documented_exceptions(path: Path) -> dict[str, str]:
    """Read committed `ticker,reason` cross-check exceptions (missing file → empty).

    DESIGN §5.2 allows a sub-0.99 ticker through the gate only when it is
    *investigated and documented*; the reason code lives in a committed CSV so
    the gate stays reproducible and CI-checkable (no silent acceptances).
    """
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        ticker = (row.get("ticker") or "").strip()
        if ticker:
            out[ticker] = (row.get("reason") or "").strip()
    return out


def cross_check(
    prices_dir: Path,
    tickers: list[str],
    key: str,
    *,
    start: date = START,
    end: date = END,
    documented: dict[str, str] | None = None,
) -> CrossCheckResult:
    """Cross-check sampled tickers' returns against Tiingo (DESIGN §5.2 gate).

    Gate passes iff every scored ticker is `pass` (>0.999) or `warn` (0.99–0.999,
    tolerated drift) or a `documented` `investigate` (<0.99 with a committed reason).
    Undocumented sub-0.99 tickers fail the gate and are surfaced for investigation.
    """
    documented = documented or {}
    rows: list[CrossCheckRow] = []
    for ticker in tickers:
        parquet = prices_dir / f"{ticker}.parquet"
        if not parquet.is_file():
            rows.append(CrossCheckRow(ticker, float("nan"), 0, "tiingo_missing"))
            continue
        ours = _parquet_close(parquet)
        theirs = fetch_tiingo(ticker, key, start, end)
        if not theirs:
            rows.append(CrossCheckRow(ticker, float("nan"), 0, "tiingo_missing"))
            continue
        corr, n = return_correlation(ours, theirs)
        if n < 2:
            status = "no_overlap"
        elif corr > 0.999:
            status = "pass"
        elif corr >= 0.99:
            status = "warn"
        else:
            status = "investigate"
        rows.append(CrossCheckRow(ticker, round(corr, 6), n, status))
    scored = [r for r in rows if r.status in ("pass", "warn", "investigate")]
    n_passed = sum(1 for r in scored if r.status == "pass")
    min_corr = min((r.correlation for r in scored), default=float("nan"))
    undocumented = [
        r.ticker for r in scored if r.status == "investigate" and r.ticker not in documented
    ]
    gate = bool(scored) and not undocumented
    return CrossCheckResult(rows, len(tickers), n_passed, min_corr, gate, undocumented)


def write_crosscheck_report(
    result: CrossCheckResult,
    documented: dict[str, str],
    out_dir: Path,
    *,
    fraction: float,
    seed: int,
) -> Path:
    """Persist the cross-check run as `crosscheck_report.csv` + `crosscheck_summary.json`.

    Stdout-only results aren't reproducible; these committed artifacts are the
    auditable record of the gate (one row per sampled ticker + a summary).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "crosscheck_report.csv"
    with report.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "correlation", "n_overlap", "status", "reason"])
        for r in result.rows:
            writer.writerow(
                [r.ticker, r.correlation, r.n_overlap, r.status, documented.get(r.ticker, "")]
            )
    summary = out_dir / "crosscheck_summary.json"
    summary.write_text(
        json.dumps(
            {
                "n_sampled": result.n_sampled,
                "n_passed": result.n_passed,
                "n_warn": sum(1 for r in result.rows if r.status == "warn"),
                "n_investigate": sum(1 for r in result.rows if r.status == "investigate"),
                "min_correlation": result.min_correlation,
                "gate_passed": result.gate_passed,
                "undocumented": result.undocumented,
                "fraction": fraction,
                "seed": seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
