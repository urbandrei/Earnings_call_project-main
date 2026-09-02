"""T6R.1 substrate audit: split-integrity metrics, sentinel-zero and binary-column detection."""

from datetime import date
from pathlib import Path

import pandas as pd

from ecvol.eval import substrate as S


def _frame(rows):
    return pd.DataFrame(rows, columns=["ticker", "date"])


def test_split_integrity_overlap_embargo_duplicates():
    frames = {
        "train": _frame([("AAA", date(2017, 1, 10)), ("BBB", date(2017, 8, 3))]),
        "val": _frame([("AAA", date(2017, 8, 3)), ("CCC", date(2017, 9, 1))]),
        "test": _frame([("BBB", date(2017, 10, 24)), ("DDD", date(2017, 11, 1))]),
    }
    rows = {r["metric"]: r for r in S.split_integrity(frames, "X", "src")}
    assert rows["calls_train"]["value"] == 2
    assert rows["test_ticker_in_train_share"]["value"] == 0.5  # BBB in train, DDD not
    assert rows["test_ticker_in_train_share"]["numerator"] == 1
    assert rows["embargo_days_train_to_val"]["value"] == 0  # 2017-08-03 on both sides
    assert rows["embargo_days_val_to_test"]["value"] == 53
    assert rows["duplicate_calls_train_val"]["value"] == 0  # same date, different ticker
    # non-temporal schemes carry no embargo rows
    keys = {r["metric"] for r in S.split_integrity(frames, "X", "src", temporal=False)}
    assert not any(k.startswith("embargo") for k in keys)


def test_sentinel_zeros_and_binary_columns(tmp_path: Path):
    lineage = tmp_path
    for s, rel in S.EC_SINGLE.items():
        p = lineage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"ticker": ["A", "B"], "future_Single_1": [-4.0, 0.0], "future_Single_2": [0.0, -3.0]}
        ).to_csv(p, index=False)
        pd.DataFrame({"ticker": ["A"], "future_2": [-4.1]}).to_csv(
            lineage / f"KeFVP/price_data/{s}_split_Avg_Series_WITH_LOG.csv", index=False
        )
    rows = {r["metric"]: r for r in S.ec_sentinel_zeros(lineage)}
    assert (
        rows["single_day_exact_zeros_all"]["value"] == 6
        and rows["single_day_exact_zeros_all"]["denominator"] == 12
    )
    assert rows["avg_series_exact_zeros_train"]["value"] == 0

    for s in ("train", "dev", "test"):
        p = lineage / S.MAEC_PRICE[("15", s)]
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "time": ["2015-02-25", "2015-03-01"],
                "future_label_1": [7.1, 7.3],
                "future_label_3": [0, 1],
                "future_label_7": [1, 0],
                "future_label_15": [0, 0],
                "future_label_30": [1, 1],
                "future_label_2": [7.2, 7.4],
            }
        ).to_csv(p, index=False)
    rows = {r["metric"]: r for r in S.maec_interleaved_binary(lineage, "15")}
    assert rows["binary_label_columns_test"]["value"] == "3,7,15,30"
    assert rows["binary_label_columns_test"]["numerator"] == 4
    assert rows["binary_columns_are_tau_test"]["value"] == 1
