"""Discovery-stage logic for the TX4 pilot: parsing and sampling (no network)."""

from datetime import date

import pandas as pd
import pytest

from ecvol.collect.discovery import (
    norm_ticker,
    parse_constituents,
    parse_nasdaq_day,
    sample_pilot,
)

WIKITABLE = """
<table id="constituents" class="wikitable">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
<tr><td><a href="x">MMM</a></td><td>3M</td><td>Industrials</td><td>Conglomerates</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector</td></tr>
</table>
"""


def test_norm_ticker_share_classes():
    assert norm_ticker("brk.b") == "BRK-B"
    assert norm_ticker("BRK/B ") == "BRK-B"


def test_parse_constituents():
    rows = parse_constituents(WIKITABLE)
    assert rows == [
        {"ticker": "MMM", "company": "3M", "gics_sector": "Industrials"},
        {"ticker": "BRK-B", "company": "Berkshire Hathaway", "gics_sector": "Financials"},
    ]


def test_parse_constituents_rejects_missing_table():
    with pytest.raises(ValueError, match="constituents"):
        parse_constituents("<table id='other'></table>")


def test_parse_nasdaq_day_buckets_and_empty():
    payload = {
        "data": {
            "rows": [
                {"symbol": "AAPL", "name": "Apple", "time": "time-after-hours"},
                {"symbol": "JPM", "name": "JPMorgan", "time": "time-pre-market"},
                {"symbol": "XYZ", "name": "Mystery", "time": "time-not-supplied"},
                {"symbol": "", "name": "blank symbol dropped"},
            ]
        }
    }
    rows = parse_nasdaq_day(payload, date(2026, 7, 20))
    assert [r["timing"] for r in rows] == ["amc", "bmo", "unknown"]
    assert all(r["call_date"] == "2026-07-20" for r in rows)
    assert parse_nasdaq_day({"data": None}, date(2026, 7, 4)) == []


def _discovery_frame() -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows.append({"ticker": f"BIG{i}", "index_membership": "sp500", "call_date": "2026-07-20"})
    for i in range(6):
        rows.append({"ticker": f"MID{i}", "index_membership": "sp400", "call_date": "2026-07-21"})
    rows.append({"ticker": "BIG0", "index_membership": "sp500", "call_date": "2026-07-22"})
    return pd.DataFrame(rows)


def test_sample_pilot_stratified_deterministic_one_per_ticker():
    frame = _discovery_frame()
    a = sample_pilot(frame, n_sp500=4, n_sp400=3, seed=7)
    b = sample_pilot(frame, n_sp500=4, n_sp400=3, seed=7)
    assert a.equals(b)
    assert a.index_membership.value_counts().to_dict() == {"sp500": 4, "sp400": 3}
    assert a.ticker.is_unique  # BIG0's duplicate calendar row can't be drawn twice


def test_sample_pilot_insufficient_stratum():
    with pytest.raises(ValueError, match="sp400"):
        sample_pilot(_discovery_frame(), n_sp500=2, n_sp400=7, seed=7)
