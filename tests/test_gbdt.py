"""T2.2 Stage-1 LightGBM baseline with ticker fixed effect."""

import numpy as np
import pandas as pd

from ecvol.models.gbdt import NUMERIC_FEATURES, train_predict_gbdt


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in NUMERIC_FEATURES})
    df["ticker"] = rng.choice(["AAA", "BBB", "CCC"], size=n)
    # target depends on a feature + ticker effect so the model has signal to fit
    df["y"] = df["v_pre"] + (df["ticker"] == "AAA") * 0.5 + rng.normal(0, 0.1, n)
    return df


def test_train_predict_runs_and_aligns():
    train, predict = _frame(seed=0), _frame(seed=1)
    pred = train_predict_gbdt(train, predict, "y", seed=0)
    assert pred.shape == (len(predict),)
    assert np.isfinite(pred).all()


def test_deterministic_per_seed():
    train, predict = _frame(seed=0), _frame(seed=1)
    a = train_predict_gbdt(train, predict, "y", seed=0)
    b = train_predict_gbdt(train, predict, "y", seed=0)
    assert np.array_equal(a, b)


def test_nan_target_rows_dropped_in_training():
    train = _frame(seed=0)
    train.loc[: len(train) // 2, "y"] = np.nan  # half the targets missing
    pred = train_predict_gbdt(train, _frame(seed=2), "y", seed=0)
    assert np.isfinite(pred).all()  # still trains on the surviving rows


def test_unseen_ticker_in_predict_is_handled():
    train = _frame(seed=0)
    predict = _frame(seed=1)
    predict.loc[0, "ticker"] = "ZZZ"  # ticker never seen in train
    pred = train_predict_gbdt(train, predict, "y", seed=0)  # shared category coding
    assert np.isfinite(pred).all()
