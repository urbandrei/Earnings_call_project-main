"""Frozen BGE-M3 section-pooled embeddings (T3.2, GPU).

Each T3.1 chunk is embedded once (content-hash cached, so re-runs are bit-identical and
incremental), then pooled per call scope (prepared_remarks / qa / full) by mean or
n_words-weighted mean. Model BAAI/bge-m3 (MIT, 1024-d, 8192-token context → no truncation of
our <=320-word chunks); fp32 + deterministic kernels (DECISIONS 2026-06-19).
Output: `data/{dataset}/text_embeddings.parquet` (+ manifest); cache under `data/{dataset}/cache/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.data.manifests import make_entry, write_manifest
from ecvol.features.text._common import (
    SCOPE_FULL,
    SCOPES,
    encode_with_cache,
    id_type_for,
    load_chunks,
    pooled,
    set_deterministic,
    write_feature_parquet,
)

EMB_MODEL = "BAAI/bge-m3"
EMB_DIM = 1024
EMB_SOURCE = "derived: ecvol featurize text (BGE-M3 embeddings, T3.2)"
EMB_LICENSE = "derived; model BAAI/bge-m3 (MIT)"


def load_model(device: str = "cuda"):
    from sentence_transformers import SentenceTransformer

    set_deterministic()
    model = SentenceTransformer(EMB_MODEL, device=device)
    model.eval()
    return model


def _encoder(model, batch_size: int):
    import torch

    def encode(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            return model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    return encode


def pool_embeddings(
    chunks: pd.DataFrame, vecs: np.ndarray, *, weighted: bool = False
) -> pd.DataFrame:
    """Per (call_id, scope) pooled vector over the chunk embeddings."""
    call_ids = chunks["call_id"].to_numpy()
    sections = chunks["section"].to_numpy()
    weights = chunks["n_words"].to_numpy()
    sources = chunks["source"].to_numpy()
    rows: list[dict] = []
    for call_id in sorted(pd.unique(call_ids)):
        in_call = call_ids == call_id
        source = sources[in_call][0]
        for scope in SCOPES:
            m = in_call if scope == SCOPE_FULL else (in_call & (sections == scope))
            if not m.any():
                continue
            pv = pooled(vecs[m], weights[m] if weighted else None)
            rows.append(
                {
                    "call_id": call_id,
                    "source": source,
                    "scope": scope,
                    "n_chunks": int(m.sum()),
                    "vector": pv.tolist(),
                }
            )
    return pd.DataFrame(rows)


def build(
    root: Path,
    dataset: str,
    *,
    limit: int | None = None,
    device: str = "cuda",
    batch_size: int = 32,
    weighted: bool = False,
) -> tuple[int, int]:
    """Embed + pool one dataset; returns (feature_rows, newly_encoded_chunks)."""
    chunks = load_chunks(root, dataset, limit=limit)
    model = load_model(device)
    cache_path = root / dataset / "cache" / "text_emb_bge_m3.parquet"
    vecs, n_new = encode_with_cache(
        chunks["text"].tolist(), EMB_MODEL, cache_path, _encoder(model, batch_size)
    )
    df = pool_embeddings(chunks, vecs, weighted=weighted)
    out = root / dataset / "text_embeddings.parquet"
    write_feature_parquet(df, out, id_type=id_type_for(dataset), sort_cols=["call_id", "scope"])
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    entry = make_entry(out, root, source_url=EMB_SOURCE, license=EMB_LICENSE)
    write_manifest([entry], root / "manifests" / f"{dataset}_text_embeddings.json")
    return len(df), n_new
