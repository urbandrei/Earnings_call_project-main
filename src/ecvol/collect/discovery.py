"""ecvol-live discovery: universe snapshot × earnings calendar (TX4 pilot).

Stage 1 of the forward-collection design (docs/fincall_methodology_and_successor.md
§4.1): a dated S&P 500 + S&P 400 universe snapshot (Wikipedia constituent lists,
raw HTML cached + manifested; CIK joined from the SEC ticker table T1.4 already
caches) crossed with the Nasdaq daily earnings calendar (one cached JSON per
day, so re-runs are offline) to yield one candidate row per (ticker, call date)
with a BMO/AMC timing bucket. Exact call datetimes and replay URLs are located
per-call at capture time, not here.

Everything lands under ``data_root/live/`` and is reproducible from the caches;
the pilot sample is a seeded stratified draw so the 35/15 split is auditable.
"""

import json
import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ecvol.data.manifests import make_entry, write_manifest

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
WIKI_LICENSE = "CC-BY-SA-4.0"  # Wikipedia text license; we store facts (tickers), not prose
NASDAQ_CALENDAR_URL = "https://api.nasdaq.com/api/calendar/earnings"
NASDAQ_LICENSE = "nasdaq.com public calendar (facts; not redistributed)"
# Wikipedia wants a descriptive UA; api.nasdaq.com hangs (no response) unless the
# request carries full browser-like headers — verified 2026-08-13.
HEADERS = {
    "User-Agent": "Mozilla/5.0 ecvol research project (andrei.roman.personal@gmail.com)",
    "Accept": "application/json, text/html, */*",
}
NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
TIMING_BUCKETS = {
    "time-pre-market": "bmo",
    "time-after-hours": "amc",
    "time-not-supplied": "unknown",
}


def norm_ticker(ticker: str) -> str:
    """Share-class separators vary by source (BRK.B / BRK-B / BRK/B) — normalize to '-'."""
    return ticker.strip().upper().replace(".", "-").replace("/", "-")


def _fetch_cached(
    url: str, cache: Path, *, params: dict | None = None, headers: dict | None = None
) -> bytes:
    """Download `url` to `cache` once (atomic); later calls read the cache."""
    if not cache.exists():
        resp = requests.get(url, params=params, headers=headers or HEADERS, timeout=60)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(cache.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(cache)
    return cache.read_bytes()


def parse_constituents(html: str) -> list[dict[str, str]]:
    """Parse a Wikipedia constituent list (table id="constituents") → ticker/company/sector."""
    table = BeautifulSoup(html, "html.parser").find("table", id="constituents")
    if table is None:
        raise ValueError("no <table id='constituents'> in page")
    # header cells can contain line breaks ("GICS<br>Sector") — join with spaces
    header_cells = table.find("tr").find_all("th")
    header = [" ".join(th.get_text(" ", strip=True).split()) for th in header_cells]
    idx = {name: header.index(name) for name in ("Symbol", "Security", "GICS Sector")}
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) < len(header):
            continue
        rows.append(
            {
                "ticker": norm_ticker(cells[idx["Symbol"]]),
                "company": cells[idx["Security"]],
                "gics_sector": cells[idx["GICS Sector"]],
            }
        )
    if not rows:
        raise ValueError("constituents table parsed to zero rows")
    return rows


def _cik_by_ticker(data_root: Path) -> dict[str, str]:
    """ticker → 10-digit CIK from the SEC table the identity work already caches."""
    from ecvol.data.fincall_identity import SEC_HEADERS, SEC_TICKERS_URL

    cache = data_root / "raw" / "ref" / "company_tickers.json"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    table = json.loads(cache.read_text(encoding="utf-8"))
    return {norm_ticker(row["ticker"]): f"{row['cik_str']:010d}" for row in table.values()}


def build_universe(data_root: Path) -> pd.DataFrame:
    """Snapshot S&P 500 + S&P 400 constituents; cache raw HTML; write universe.csv + manifest."""
    live = data_root / "live"
    frames = []
    entries = []
    for index_name, url in (("sp500", WIKI_SP500_URL), ("sp400", WIKI_SP400_URL)):
        cache = data_root / "raw" / "live" / f"wiki_{index_name}.html"
        html = _fetch_cached(url, cache).decode("utf-8")
        entries.append(make_entry(cache, data_root, source_url=url, license=WIKI_LICENSE))
        frame = pd.DataFrame(parse_constituents(html))
        frame["index_membership"] = index_name
        frames.append(frame)
    universe = pd.concat(frames, ignore_index=True)
    dupes = universe[universe.ticker.duplicated(keep=False)]
    if not dupes.empty:  # a ticker in both lists would double-count a call
        raise ValueError(f"tickers in both indices: {sorted(dupes.ticker.unique())}")
    cik = _cik_by_ticker(data_root)
    universe["cik"] = universe.ticker.map(cik)
    universe = universe.sort_values("ticker", ignore_index=True)
    live.mkdir(parents=True, exist_ok=True)
    universe.to_csv(live / "universe.csv", index=False)
    write_manifest(entries, data_root / "manifests" / "live_universe.json")
    return universe


def parse_nasdaq_day(payload: dict, day: date) -> list[dict[str, str]]:
    """One cached Nasdaq calendar JSON → candidate rows (empty list on no-earnings days)."""
    rows = (payload.get("data") or {}).get("rows") or []
    return [
        {
            "ticker": norm_ticker(row["symbol"]),
            "calendar_name": row.get("name", ""),
            "call_date": day.isoformat(),
            "timing": TIMING_BUCKETS.get(row.get("time", ""), "unknown"),
            "fiscal_quarter": row.get("fiscalQuarterEnding", ""),
        }
        for row in rows
        if row.get("symbol")
    ]


def build_discovery(data_root: Path, start: date, end: date) -> pd.DataFrame:
    """Sweep the calendar over [start, end], intersect with the universe → discovery.csv."""
    universe = pd.read_csv(data_root / "live" / "universe.csv", dtype={"cik": str})
    candidates = []
    day = start
    while day <= end:
        cache = data_root / "raw" / "live" / "calendar" / f"nasdaq_{day.isoformat()}.json"
        params = {"date": day.isoformat()}
        raw = _fetch_cached(NASDAQ_CALENDAR_URL, cache, params=params, headers=NASDAQ_HEADERS)
        payload = json.loads(raw)
        candidates.extend(parse_nasdaq_day(payload, day))
        day += timedelta(days=1)
    columns = ["ticker", "calendar_name", "call_date", "timing", "fiscal_quarter"]
    calendar = pd.DataFrame(candidates, columns=columns)
    discovery = calendar.merge(universe, on="ticker", how="inner")
    discovery["discovered_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    discovery = discovery.sort_values(["call_date", "ticker"], ignore_index=True)
    discovery.to_csv(data_root / "live" / "discovery.csv", index=False)
    return discovery


def sample_pilot(discovery: pd.DataFrame, *, n_sp500: int, n_sp400: int, seed: int) -> pd.DataFrame:
    """Seeded stratified draw, one call per ticker (a ticker can appear on two calendar days)."""
    one_per_ticker = discovery.drop_duplicates("ticker")
    parts = []
    for index_name, n in (("sp500", n_sp500), ("sp400", n_sp400)):
        stratum = one_per_ticker[one_per_ticker.index_membership == index_name]
        if len(stratum) < n:
            raise ValueError(f"{index_name}: need {n}, discovery has {len(stratum)}")
        picks = sorted(random.Random(f"{seed}:{index_name}").sample(sorted(stratum.ticker), n))
        parts.append(stratum[stratum.ticker.isin(picks)])
    return pd.concat(parts).sort_values(["call_date", "ticker"], ignore_index=True)
