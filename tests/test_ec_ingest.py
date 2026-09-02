"""T6R.2 EC ingestion: folder parsing, lineage-based identity, published split, targets."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ecvol.data import ec_ingest as E
from ecvol.data.calendar import sessions_in_range
from ecvol.data.prices import write_price_parquet

DAY0 = date(2017, 4, 25)


def test_parse_folder_name():
    assert E.parse_folder_name("3M Company_20170425") == ("3M Company", "2017-04-25")
    assert E.parse_folder_name("Amazon.com, Inc._20170202") == ("Amazon.com, Inc.", "2017-02-02")
    assert E.parse_folder_name("no_date_here") is None


def _seed(root: Path) -> None:
    ds = root / E.DATASET_REL
    for name, n in (("3M Company_20170425", 3), ("Ghost Corp_20170501", 2)):
        d = ds / name / "CEO"
        d.mkdir(parents=True)
        (ds / name / "TextSequence.txt").write_text(
            "\n".join(f"Sentence number {i} about revenue and margins." for i in range(n)) + "\n",
            encoding="utf-8",
        )
        (d / "Some One_1_1.mp3").write_bytes(b"")
    lin = root / E.LINEAGE_DATA
    lin.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": ["mmm"],
            "name": ["3M"],
            "year": [2017],
            "month": [4],
            "day": [25],
            "text_file_name": ["3M Company_20170425"],
        }
    ).to_csv(lin / "full_stock_data.csv", index=False)
    for s, ids in (("train", ["3M Company_20170425"]), ("val", []), ("test", [])):
        pd.DataFrame({"ticker": ["MMM"] * len(ids), "text_file_name": ids}).to_csv(
            lin / f"{s}_split3.csv", index=False
        )
    sessions = sessions_in_range(DAY0 - timedelta(days=90), DAY0 + timedelta(days=90))
    rows = [
        {"date": d.isoformat(), "open": p, "high": p, "low": p, "close": p, "volume": 1}
        for d, p in zip(
            sessions, (100.0 + (i % 5) * 0.3 for i in range(len(sessions))), strict=True
        )
    ]
    write_price_parquet(rows, root / "prices" / "MMM.parquet")


def test_ingest_ec_identity_split_targets(tmp_path: Path):
    root = tmp_path / "data"
    _seed(root)
    s = E.ingest_ec(root, horizons=(3,))
    assert s.total_calls == 2 and s.ok == 1 and s.reason_counts == {"unresolved": 1}
    assert s.joined == 1 and s.join_rate_pct == 100.0 and s.published_split_calls == 1
    calls = pq.read_table(root / "ec" / "calls.parquet").to_pandas().set_index("call_id")
    assert calls.loc["3M Company_20170425", "ticker"] == "MMM"  # upper-cased from the lineage
    assert calls.loc["Ghost Corp_20170501", "reason"] == "unresolved"
    assert not calls["audio_exists"].any()  # sentence clips only, no full-call file
    pub = pd.read_csv(root / "splits" / "ec_published.csv")
    assert list(pub.columns) == ["call_id", "ticker", "as_of", "split"]
    assert pub.iloc[0].tolist() == ["3M Company_20170425", "MMM", DAY0.isoformat(), "train"]
    assert (root / "ec" / "targets_calendar.parquet").is_file()
    assert (root / "manifests" / "ec_targets.json").is_file()
    assert (root / "coverage" / "ec_missing_tickers.csv").is_file()


def test_ingest_ec_deterministic(tmp_path: Path):
    root = tmp_path / "data"
    _seed(root)
    E.ingest_ec(root, horizons=(3,))
    b = {n: (root / "ec" / n).read_bytes() for n in ("calls.parquet", "targets.parquet")}
    E.ingest_ec(root, horizons=(3,))
    for n, x in b.items():
        assert (root / "ec" / n).read_bytes() == x
