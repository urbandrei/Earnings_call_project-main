"""T3.4 identity-control suite: ticker-only, transcript shuffle, identity probe, orchestration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.eval import controls as C
from ecvol.models.ticker_only import ticker_mean_fit_predict

# --- ticker-only -------------------------------------------------------------


def test_ticker_mean_and_unseen_fallback():
    tr_t = np.array(["A", "A", "B"])
    tr_y = np.array([1.0, 3.0, 10.0])
    pred = ticker_mean_fit_predict(tr_t, tr_y, np.array(["A", "B", "C"]))
    assert pred[0] == 2.0  # mean of A
    assert pred[1] == 10.0  # B
    assert np.isclose(pred[2], np.mean(tr_y))  # unseen C → global mean


def test_ticker_mean_ignores_nan_targets():
    pred = ticker_mean_fit_predict(np.array(["A", "A"]), np.array([np.nan, 4.0]), np.array(["A"]))
    assert pred[0] == 4.0


# --- shuffle -----------------------------------------------------------------


def test_within_shuffle_keeps_singletons_global_permutes():
    rng = np.random.default_rng(0)
    Xte = np.arange(12.0).reshape(6, 2)
    tickers = np.array(["A", "A", "A", "B", "B", "C"])  # C is a singleton
    out = C._shuffle_text(Xte, tickers, [0, 1], "within_shuffle", rng)
    assert np.array_equal(out[5], Xte[5])  # singleton C unchanged
    assert set(map(tuple, out[:3])) == set(map(tuple, Xte[:3]))  # A rows are a permutation
    nop = C._shuffle_text(Xte, tickers, [], "global_shuffle", rng)
    assert np.array_equal(nop, Xte)  # no text cols → no-op


# --- identity probe ----------------------------------------------------------


def test_identity_probe_separable(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fincall").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, calls = [], []
    for t, center in (("AAA", 0.0), ("BBB", 10.0)):
        for c in range(8):
            cid = f"{t}{c}"
            vec = list(rng.normal(center, 0.1, size=16))
            rows.append(
                {"call_id": cid, "source": "fincall", "scope": "full", "n_chunks": 1, "vector": vec}
            )
            calls.append({"call_id": cid, "ticker": t})
    pq.write_table(pa.Table.from_pylist(rows), root / "fincall" / "text_embeddings.parquet")
    pq.write_table(pa.Table.from_pylist(calls), root / "fincall" / "calls.parquet")
    res = C.identity_probe(root, "fincall", seed=0)
    assert res["n_tickers"] == 2 and res["chance"] == 0.5
    assert res["probe_accuracy"] > 0.8  # well-separated clusters → easily identified


# --- orchestration smoke -----------------------------------------------------


def _eval_frame(n=80):
    rng = np.random.default_rng(3)
    rows = []
    q = ["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"]
    for c in range(n):
        v = float(rng.normal(-4, 0.5))
        for h in (3, 7, 15, 30):
            rows.append(
                {
                    "call_id": str(c),
                    "ticker": f"T{c % 10}",
                    "horizon": h,
                    "as_of": q[c % 4],
                    "v_pre": v,
                    "v_post": v + float(rng.normal(0, 0.3)),
                    "delta_v": float(rng.normal(0, 0.3)),
                    "rv_daily": abs(v),
                    "rv_weekly": abs(v),
                    "rv_monthly": abs(v),
                    "n_turns": 20,
                    "n_chars": 5000,
                }
            )
    return pd.DataFrame(rows)


def test_controls_dataset_smoke(tmp_path: Path):
    splits = tmp_path / "splits"
    splits.mkdir()
    pd.DataFrame(
        [
            {"call_id": str(c), "split": "train" if c < 56 else ("val" if c < 68 else "test")}
            for c in range(80)
        ]
    ).to_csv(splits / "fincall_temporal.csv", index=False)
    df = _eval_frame()
    rng = np.random.default_rng(4)
    emb_cols, other_cols = ["e0", "e1", "e2"], ["fb_x", "sf_y"]
    tdf = pd.DataFrame(
        {
            "call_id": [str(c) for c in range(80)],
            **{c: rng.normal(size=80) for c in emb_cols + other_cols},
        }
    )
    rows = C._controls_dataset(df, tdf, emb_cols, other_cols, "fincall", splits, seeds=(0,))
    t = pd.DataFrame(rows)
    assert {"ticker_only", "ridge_text", "mlp_text_pastvol"} <= set(t["model"])
    assert set(t["condition"]) == {"real", "within_shuffle", "global_shuffle"}
    # pastvol-only head has no text cols → shuffle is a no-op (real == within == global)
    pv = t[(t.model == "ridge_pastvol") & (t.target == "v") & (t.horizon == 3)]
    assert pv["r2_oos"].nunique() == 1
