"""T3.2 frozen text features: pooling, aggregation, surface stats, content-hash cache.

All torch-free — the model-loading paths import torch/sentence-transformers/transformers lazily,
so these run in CI (which installs no GPU group). The real model output is validated by the
50-call benchmark, not here; here we lock the pure logic + the bit-identical cache guarantee.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from ecvol.features.text import _common, embeddings, finbert, surface
from ecvol.features.text.sections import write_chunks_parquet


def _chunk(call_id, section, role, turn_idx, text, cit=0):
    return {
        "call_id": call_id,
        "source": "fincall",
        "section": section,
        "role": role,
        "turn_idx": turn_idx,
        "chunk_in_turn": cit,
        "n_words": len(text.split()),
        "n_chars": len(text),
        "oversize": False,
        "text": text,
    }


def _chunks_df():
    return pd.DataFrame(
        [
            _chunk(1, "prepared_remarks", "management", 0, "Revenue was $1.2 billion up 20%."),
            _chunk(1, "qa", "analyst", 1, "How are margins trending?"),
            _chunk(1, "qa", "management", 2, "Margins expanded nicely."),
        ]
    )


# --- _common -----------------------------------------------------------------


def test_content_hash_stable_and_model_sensitive():
    a = _common.content_hash("m1", "hello")
    assert a == _common.content_hash("m1", "hello")
    assert a != _common.content_hash("m2", "hello")
    assert a != _common.content_hash("m1", "world")


def test_pooled_mean_and_weighted():
    v = np.array([[0.0, 0.0], [2.0, 4.0]])
    assert np.allclose(_common.pooled(v), [1.0, 2.0])
    assert np.allclose(_common.pooled(v, np.array([3.0, 1.0])), [0.5, 1.0])
    assert np.isnan(_common.pooled(np.zeros((0, 2)))).all()


def test_encode_with_cache_dedupes_and_is_bit_identical(tmp_path: Path):
    calls = {"n": 0}

    def fake_encode(texts):
        calls["n"] += len(texts)
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    cache = tmp_path / "cache.parquet"
    texts = ["aa", "bbb", "aa"]  # one duplicate
    out1, new1 = _common.encode_with_cache(texts, "m", cache, fake_encode)
    assert new1 == 2 and calls["n"] == 2  # duplicate encoded once
    assert np.allclose(out1, [[2, 1], [3, 1], [2, 1]])
    out2, new2 = _common.encode_with_cache(texts, "m", cache, fake_encode)
    assert new2 == 0 and calls["n"] == 2  # all cache hits, nothing re-encoded
    assert np.array_equal(out1, out2)  # bit-identical re-run


# --- embeddings pooling ------------------------------------------------------


def test_pool_embeddings_by_scope():
    chunks = _chunks_df()
    vecs = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 4.0]])  # prepared, qa, qa
    df = embeddings.pool_embeddings(chunks, vecs)
    by = {r.scope: r.vector for r in df.itertuples()}
    assert np.allclose(by["prepared_remarks"], [1.0, 0.0])
    assert np.allclose(by["qa"], [0.0, 3.0])  # mean of the two qa rows
    assert np.allclose(by["full"], [1 / 3, 2.0])  # mean of all three


# --- finbert aggregation -----------------------------------------------------


def test_aggregate_by_scope_and_role():
    chunks = _chunks_df()
    labels = ["positive", "negative", "neutral"]
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.6, 0.2, 0.2]])
    df = finbert.aggregate(chunks, probs, labels)
    prep = df[(df.scope == "prepared_remarks") & (df.role == "all")].iloc[0]
    assert np.isclose(prep.p_positive, 0.8) and np.isclose(prep.net, 0.7)
    qa_analyst = df[(df.scope == "qa") & (df.role == "analyst")].iloc[0]
    assert np.isclose(qa_analyst.p_negative, 0.7) and np.isclose(qa_analyst.net, -0.5)
    qa_all = df[(df.scope == "qa") & (df.role == "all")].iloc[0]
    assert np.isclose(qa_all.p_positive, 0.4)  # mean(0.2, 0.6)


# --- surface -----------------------------------------------------------------


def test_surface_numeric_density_and_questions():
    df = surface.compute(_chunks_df())
    prep = df[df.scope == "prepared_remarks"].iloc[0]
    # "Revenue was $1.2 billion up 20%." → $1.2 and 20% are numeric (2 of 6 tokens)
    assert np.isclose(prep.numeric_density, 2 / 6)
    qa = df[df.scope == "qa"].iloc[0]
    assert qa.question_marks == 1 and qa.n_turns == 2


def test_surface_build_deterministic(tmp_path: Path):
    root = tmp_path / "data"
    rows = [_chunk(1, "prepared_remarks", "management", 0, "Numbers like 5 and 6 percent.")]
    write_chunks_parquet(rows, root / "fincall" / "chunks.parquet", id_type=pa.int64())
    surface.build(root, "fincall")
    b1 = (root / "fincall" / "text_surface.parquet").read_bytes()
    surface.build(root, "fincall")
    assert (root / "fincall" / "text_surface.parquet").read_bytes() == b1
