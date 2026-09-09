"""T6R.2 HTML reproduction: label tables, chunk-sequence assembly, and a CPU smoke fit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecvol.eval import reproduce as R
from ecvol.features.text._common import content_hash
from ecvol.features.text.embeddings import EMB_MODEL


def test_our_labels_wide_table(tmp_path: Path):
    root = tmp_path
    (root / "x").mkdir()
    rows = []
    for cid in ("a", "b"):
        for h in (3, 7):
            rows.append(
                {
                    "call_id": cid,
                    "ticker": "T",
                    "as_of": "2017-01-05",
                    "horizon": h,
                    "v_post": -4.0 - h / 10,
                    "v_pre": -4.5,
                    "status": "ok",
                }
            )
    rows.append(
        {
            "call_id": "c",
            "ticker": "T",
            "as_of": "2017-01-05",
            "horizon": 3,
            "v_post": 0.0,
            "v_pre": 0.0,
            "status": "excluded",
        }
    )
    pd.DataFrame(rows).to_parquet(root / "x" / "targets.parquet")
    (root / "prices").mkdir()
    dates = [f"2017-01-{d:02d}" for d in range(2, 32)]  # 30 sessions; as_of = index 3
    close = [100.0 * (1.02**k) for k in range(len(dates))]  # +2% every session
    pd.DataFrame({"date": dates, "close": close}).to_parquet(root / "prices" / "T.parquet")
    lab = R.our_labels(root, "x")
    assert list(lab.index) == ["a", "b"]  # excluded rows dropped
    assert lab.loc["a", "future_7"] == -4.7 and lab.loc["a", "past_3"] == -4.5
    # auxiliary = ln|return on session +τ| (the KeFVP `future_Single_τ` definition)
    assert np.isclose(lab.loc["a", "future_Single_3"], np.log(0.02))
    assert np.isclose(lab.loc["a", "future_Single_15"], np.log(0.02))
    assert np.isnan(lab.loc["a", "future_Single_30"])  # beyond the price series


def test_chunk_sequences_follow_transcript_order(tmp_path: Path):
    root = tmp_path
    (root / "x" / "cache").mkdir(parents=True)
    chunks = pd.DataFrame(
        {
            "call_id": ["k", "k", "k"],
            "turn_idx": [1, 0, 0],
            "chunk_in_turn": [0, 1, 0],
            "text": ["third", "second", "first"],
        }
    )
    chunks.to_parquet(root / "x" / "chunks.parquet")
    pd.DataFrame(
        {
            "hash": [content_hash(EMB_MODEL, t) for t in ("first", "second", "third")],
            "vector": [np.full(4, i, dtype=np.float32) for i in (1, 2, 3)],
        }
    ).to_parquet(root / "x" / "cache" / "text_emb_bge_m3.parquet")
    seqs = R.chunk_sequences(root, "x")
    assert seqs["k"].shape == (3, 4)
    assert seqs["k"][:, 0].tolist() == [1.0, 2.0, 3.0]  # (turn 0, chunk 0), (0, 1), (1, 0)


def test_html_head_smoke_fit_cpu():
    pytest.importorskip("torch")
    from ecvol.models import html_head as H

    rng = np.random.default_rng(0)
    seqs = [rng.normal(size=(rng.integers(3, 9), 8)).astype(np.float32) for _ in range(12)]
    y = rng.normal(size=12)
    ya = rng.normal(size=12)
    tr, te = np.arange(8), np.arange(8, 12)
    pred = H.fit_predict(seqs, y, ya, tr, te, alpha=0.5, seed=0, epochs=1, device="cpu")
    assert pred.shape == (4,) and np.all(np.isfinite(pred))
    again = H.fit_predict(seqs, y, ya, tr, te, alpha=0.5, seed=0, epochs=1, device="cpu")
    assert np.allclose(pred, again)  # seeded ⇒ repeatable on CPU
