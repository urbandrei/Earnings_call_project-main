"""T1.2 price ingestion: universe assembly, parquet determinism, coverage report."""

from pathlib import Path

import pyarrow.parquet as pq

from ecvol.data import calendar as cal
from ecvol.data.prices import (
    build_coverage,
    build_universe,
    to_yahoo_symbol,
    write_price_parquet,
)


def test_to_yahoo_symbol_normalizes_share_classes():
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol("bf.b") == "BF-B"
    assert to_yahoo_symbol(" aapl ") == "AAPL"
    assert to_yahoo_symbol("AAPL") == "AAPL"


def _make_identity(root: Path, rows: list[tuple[str, str]]) -> None:
    (root / "identity").mkdir(parents=True, exist_ok=True)
    lines = ["call_id,ticker,call_type"]
    for i, (ticker, ctype) in enumerate(rows):
        lines.append(f"{i},{ticker},{ctype}")
    csv_text = "\n".join(lines) + "\n"
    (root / "identity" / "fincall_identity.csv").write_text(csv_text, encoding="utf-8")


def _make_maec(root: Path, folders: list[str]) -> None:
    base = root / "raw" / "maec" / "repo" / "MAEC_Dataset"
    base.mkdir(parents=True, exist_ok=True)
    for name in folders:
        (base / name).mkdir()


def test_build_universe_unions_fincall_and_maec(tmp_path):
    _make_identity(tmp_path, [("AAPL", "earnings"), ("AAPL", "earnings"), ("MSFT", "earnings")])
    _make_maec(tmp_path, ["20150226_AAPL", "20160101_AMZN", "not_a_call", "20160202_AMZN"])
    universe = build_universe(tmp_path)

    assert set(universe) == {"AAPL", "AMZN", "MSFT"}
    assert universe["AAPL"].in_fincall and universe["AAPL"].fincall_calls == 2
    assert universe["AAPL"].in_maec and universe["AAPL"].maec_calls == 1
    assert universe["AMZN"].in_maec and universe["AMZN"].maec_calls == 2
    assert universe["AMZN"].fincall_calls == 0
    assert universe["MSFT"].in_fincall and not universe["MSFT"].in_maec
    # deterministic ordering
    assert list(universe) == sorted(universe)


def test_write_price_parquet_is_deterministic(tmp_path):
    rows = [
        {"date": "2020-01-02", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100},
        {"date": "2020-01-03", "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 200},
    ]
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_price_parquet(rows, a)
    write_price_parquet(rows, b)
    assert a.read_bytes() == b.read_bytes()
    table = pq.read_table(a)
    assert table.column("date").to_pylist() == ["2020-01-02", "2020-01-03"]
    assert table.column("close").to_pylist() == [1.5, 2.0]


def _contiguous_rows(start, end):
    rows = []
    for d in cal.sessions_in_range(start, end):
        rows.append(
            {"date": d.isoformat(), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
        )
    return rows


def test_build_coverage_status_reasons_and_summary(tmp_path):
    from datetime import date

    _make_identity(tmp_path, [("AAA", "earnings"), ("BBB", "earnings"), ("CCC", "earnings")])
    _make_maec(tmp_path, ["20150226_AAA", "20150226_DDD"])
    universe = build_universe(tmp_path)
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()

    # AAA fully covered (every session present); BBB gappy; CCC + DDD missing.
    full = _contiguous_rows(date(2020, 1, 2), date(2020, 3, 31))
    write_price_parquet(full, prices_dir / "AAA.parquet")
    write_price_parquet(full[::3], prices_dir / "BBB.parquet")  # keep ~1/3 of sessions

    rows, summary = build_coverage(universe, prices_dir)
    by_ticker = {r.ticker: r for r in rows}

    assert by_ticker["AAA"].status == "ok" and by_ticker["AAA"].reason == ""
    assert by_ticker["AAA"].completeness >= 0.99
    assert by_ticker["BBB"].status == "ok" and by_ticker["BBB"].reason == "gappy"
    assert by_ticker["CCC"].status == "missing"
    assert by_ticker["CCC"].reason.startswith("no_data")
    assert by_ticker["DDD"].status == "missing"

    # FinCall universe = AAA,BBB,CCC → 2/3 covered; MAEC = AAA,DDD → 1/2 covered.
    assert summary.fincall_total == 3 and summary.covered_fincall == 2
    assert summary.fincall_coverage_pct == round(200 / 3, 2)
    assert summary.maec_total == 2 and summary.covered_maec == 1
    assert summary.missing_tickers == ["CCC", "DDD"]
    assert "BBB" in summary.low_completeness


def test_build_universe_empty_when_no_sources(tmp_path):
    assert build_universe(tmp_path) == {}


def test_pull_prices_tiingo_fallback_recovers_yahoo_misses(tmp_path, monkeypatch):
    import json

    from ecvol.data import prices as P

    _make_identity(tmp_path, [("AAA", "earnings"), ("BBB", "earnings")])

    # yfinance serves AAA, returns nothing for BBB (Yahoo purged the delisted symbol).
    def fake_fetch_batch(symbols, start, end):
        row = {
            "date": "2020-01-02",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        }
        return {s: ([row] if s == "AAA" else []) for s in symbols}

    # Tiingo retains BBB.
    def fake_tiingo(ticker, key, start, end):
        return [
            {"date": "2020-01-02", "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 9}
        ]

    monkeypatch.setattr(P, "fetch_batch", fake_fetch_batch)
    monkeypatch.setattr("ecvol.data.tiingo.load_api_key", lambda root=None: "key")
    monkeypatch.setattr("ecvol.data.tiingo.fetch_tiingo_ohlcv", fake_tiingo)

    summary = P.pull_prices(tmp_path)

    assert summary.fincall_coverage_pct == 100.0
    assert (tmp_path / "prices" / "AAA.parquet").is_file()
    assert (tmp_path / "prices" / "BBB.parquet").is_file()
    sources = json.loads((tmp_path / "coverage" / "prices_sources.json").read_text())
    assert sources == {"AAA": "yahoo", "BBB": "tiingo"}
    manifest = json.loads((tmp_path / "manifests" / "prices.json").read_text())
    by_path = {e["path"]: e["source_url"] for e in manifest}
    assert "tiingo.com" in by_path["prices/BBB.parquet"]
    assert "yahoo.com" in by_path["prices/AAA.parquet"]


def test_pull_prices_no_key_leaves_documented_shortfall(tmp_path, monkeypatch):
    from ecvol.data import prices as P

    _make_identity(tmp_path, [("AAA", "earnings"), ("BBB", "earnings")])

    def fake_fetch_batch(symbols, start, end):
        row = {
            "date": "2020-01-02",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        }
        return {s: ([row] if s == "AAA" else []) for s in symbols}

    monkeypatch.setattr(P, "fetch_batch", fake_fetch_batch)
    monkeypatch.setattr("ecvol.data.tiingo.load_api_key", lambda root=None: None)  # no key

    summary = P.pull_prices(tmp_path)
    assert summary.fincall_coverage_pct == 50.0
    assert summary.missing_tickers == ["BBB"]
    assert not (tmp_path / "prices" / "BBB.parquet").exists()
