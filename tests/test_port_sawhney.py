"""T6R.3 Sawhney port: their label convention, split rule, financial features."""

import numpy as np
import pandas as pd

from ecvol.eval import port_sawhney as P


def _closes(n=100, seed=0):
    dates = [f"2017-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
    rng = np.random.default_rng(seed)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return dict(zip(dates, p, strict=True)), dates, p


def test_their_labels_use_sessions_after_and_before_the_call():
    closes, dates, p = _closes()
    r = p[1:] / p[:-1] - 1  # r[i] = return on session i+1
    lab = P.their_labels(closes, dates[50])
    assert np.isclose(lab["future_3"], P.their_volatility(r[50:53]))  # sessions 51..53
    assert np.isclose(lab["past_3"], P.their_volatility(r[46:49]))  # sessions 47..49
    assert P.their_labels(closes, dates[5]) is None  # window not available


def test_their_volatility_is_mean_subtracted():
    r = np.array([0.01, 0.01, 0.01])
    assert P.their_volatility(r) == -np.inf or np.isinf(P.their_volatility(r))
    r = np.array([0.02, -0.02])
    assert np.isclose(P.their_volatility(r), np.log(0.02))


def test_split_uses_month_day_key_and_60_20_20():
    df = pd.DataFrame({"call_date": [f"2017-{m:02d}-15" for m in (3, 1, 2, 4, 5, 6, 7, 8, 9, 10)]})
    s = P.their_split(df)
    assert s.value_counts().to_dict() == {"train": 6, "val": 2, "test": 2}
    assert s[df["call_date"] == "2017-01-15"].iloc[0] == "train"
    assert s[df["call_date"] == "2017-10-15"].iloc[0] == "test"


def test_past_vol_vector_first_entry_is_session_minus_one():
    closes, dates, p = _closes()
    r = p[1:] / p[:-1] - 1
    x = P.past_vol_vector(closes, dates[50], 3)
    assert x.shape == (30,) and np.isclose(x[0], P.their_volatility(r[45:48]))  # sessions 46..48


def test_sentence_matrix_pads_and_means():
    vec = {"growth": np.ones(300, np.float32), "strong": np.zeros(300, np.float32)}
    m = P.sentence_matrix(["Strong growth ahead", "nothing known"], vec, set())
    assert m.shape == (2, 300) and np.isclose(m[0, 0], 0.5) and m[1].sum() == 0


def test_ensemble_picks_best_alpha():
    y = np.array([1.0, 2.0, 3.0])
    assert P.ensemble(y, np.zeros(3), y) == 1.0 and P.ensemble(np.zeros(3), y, y) == 0.0


def test_condition_splits_map_committed_splits_and_exclude_the_rest(tmp_path):
    (tmp_path / "splits").mkdir()
    pd.DataFrame(
        {"call_id": ["a", "b", "c"], "ticker": ["A", "B", "C"], "as_of": ["x"] * 3,
         "split": ["train", "embargo", "test"]}
    ).to_csv(tmp_path / "splits" / "ec_temporal.csv", index=False)  # fmt: skip
    df = pd.DataFrame({"call_id": ["a", "b", "c", "d"], "split": ["train", "val", "test", "test"]})
    out = P.condition_splits(tmp_path, df, ("their", "embargoed"))
    assert out["their"].tolist() == ["train", "val", "test", "test"]
    assert out["embargoed"].tolist() == ["train", "embargo", "test", "excluded"]
