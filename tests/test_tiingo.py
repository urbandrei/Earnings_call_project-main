"""T1.2 Tiingo cross-check: return-correlation math, sampling, key loading (no network)."""

import math
from datetime import date

from ecvol.data.prices import write_price_parquet
from ecvol.data.tiingo import (
    cross_check,
    fetch_tiingo,
    fetch_tiingo_ohlcv,
    load_api_key,
    return_correlation,
    sample_tickers,
)

_TIINGO_CSV = (
    "date,close,high,low,open,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume,divCash,splitFactor\n"
    "2020-01-02T00:00:00.000Z,100,101,99,100,1000,50.0,50.5,49.5,50.0,2000,0,2.0\n"
    "2020-01-03T00:00:00.000Z,102,103,101,100,1100,51.0,51.5,50.5,50.0,2200,0,2.0\n"
)


class _FakeResp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_fetch_tiingo_ohlcv_uses_adjusted_columns(monkeypatch):
    monkeypatch.setattr("ecvol.data.tiingo.requests.get", lambda *a, **k: _FakeResp(_TIINGO_CSV))
    rows = fetch_tiingo_ohlcv("AAA", key="k")
    assert [r["date"] for r in rows] == ["2020-01-02", "2020-01-03"]
    assert rows[0]["close"] == 50.0 and rows[0]["open"] == 50.0  # adjClose/adjOpen, not raw
    assert rows[0]["volume"] == 2000  # adjVolume
    # close-map helper derives from the same parse
    monkeypatch.setattr("ecvol.data.tiingo.requests.get", lambda *a, **k: _FakeResp(_TIINGO_CSV))
    assert fetch_tiingo("AAA", key="k") == {"2020-01-02": 50.0, "2020-01-03": 51.0}


def test_return_correlation_identical_series_is_one():
    a = {"2020-01-02": 10.0, "2020-01-03": 11.0, "2020-01-06": 10.5, "2020-01-07": 12.0}
    corr, n = return_correlation(a, dict(a))
    assert n == 3
    assert math.isclose(corr, 1.0, abs_tol=1e-9)


def test_return_correlation_uses_only_common_dates():
    a = {"2020-01-02": 10.0, "2020-01-03": 11.0, "2020-01-06": 10.5}
    b = {"2020-01-03": 11.0, "2020-01-06": 10.5, "2020-01-07": 9.0}  # offset by one date
    corr, n = return_correlation(a, b)
    # only the 01-03→01-06 return is shared
    assert n == 1
    assert math.isnan(corr)  # <2 points → undefined


def test_return_correlation_scaled_prices_still_correlate():
    a = {"2020-01-02": 10.0, "2020-01-03": 11.0, "2020-01-06": 9.9, "2020-01-07": 12.1}
    b = {d: v * 7.5 for d, v in a.items()}  # different split-adjustment scale, same returns
    corr, n = return_correlation(a, b)
    assert n == 3
    assert math.isclose(corr, 1.0, abs_tol=1e-9)


def test_sample_tickers_is_deterministic_and_sized():
    pool = [f"T{i:03d}" for i in range(100)]
    s1 = sample_tickers(pool, fraction=0.05, seed=0)
    s2 = sample_tickers(pool, fraction=0.05, seed=0)
    assert s1 == s2 == sorted(s1)
    assert len(s1) == 5
    assert sample_tickers(pool, fraction=0.05, seed=1) != s1
    assert len(sample_tickers([], 0.05)) == 0
    assert len(sample_tickers(["ONE"], 0.05)) == 1  # always at least one


def test_load_api_key_env_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    assert load_api_key(tmp_path) is None
    (tmp_path / ".env").write_text(
        "# comment\nTIINGO_API_KEY = 'abc123'\nOTHER=x\n", encoding="utf-8"
    )
    assert load_api_key(tmp_path) == "abc123"
    monkeypatch.setenv("TIINGO_API_KEY", "fromenv")
    assert load_api_key(tmp_path) == "fromenv"  # env wins


def test_cross_check_gate_with_mocked_fetch(tmp_path, monkeypatch):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    rows = [
        {"date": "2020-01-02", "open": 1, "high": 1, "low": 1, "close": 10.0, "volume": 1},
        {"date": "2020-01-03", "open": 1, "high": 1, "low": 1, "close": 11.0, "volume": 1},
        {"date": "2020-01-06", "open": 1, "high": 1, "low": 1, "close": 9.9, "volume": 1},
        {"date": "2020-01-07", "open": 1, "high": 1, "low": 1, "close": 12.1, "volume": 1},
    ]
    write_price_parquet(rows, prices_dir / "AAA.parquet")

    # Tiingo returns the same adjusted closes (scaled) → corr 1.0 → gate passes.
    def fake_fetch(ticker, key, start=date(2014, 6, 1), end=date(2022, 6, 30)):
        return {r["date"]: r["close"] * 4.0 for r in rows}

    monkeypatch.setattr("ecvol.data.tiingo.fetch_tiingo", fake_fetch)
    result = cross_check(prices_dir, ["AAA"], key="k")
    assert result.gate_passed
    assert result.rows[0].status == "pass"
    assert math.isclose(result.rows[0].correlation, 1.0, abs_tol=1e-6)


def test_cross_check_flags_divergent_ticker(tmp_path, monkeypatch):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    rows = [
        {"date": "2020-01-02", "open": 1, "high": 1, "low": 1, "close": 10.0, "volume": 1},
        {"date": "2020-01-03", "open": 1, "high": 1, "low": 1, "close": 11.0, "volume": 1},
        {"date": "2020-01-06", "open": 1, "high": 1, "low": 1, "close": 9.9, "volume": 1},
        {"date": "2020-01-07", "open": 1, "high": 1, "low": 1, "close": 12.1, "volume": 1},
    ]
    write_price_parquet(rows, prices_dir / "AAA.parquet")

    def fake_fetch(ticker, key, start=date(2014, 6, 1), end=date(2022, 6, 30)):
        # uncorrelated/opposite movements
        return {"2020-01-02": 5.0, "2020-01-03": 4.0, "2020-01-06": 6.0, "2020-01-07": 3.0}

    monkeypatch.setattr("ecvol.data.tiingo.fetch_tiingo", fake_fetch)
    result = cross_check(prices_dir, ["AAA"], key="k")
    assert not result.gate_passed
    assert result.rows[0].status in ("warn", "investigate")
