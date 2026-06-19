"""Shared helpers for the Stage-2 frozen text features (T3.2).

Loading the T3.1 chunk table, content-hash cache keys, deterministic torch setup,
section/scope grouping, pooling, and deterministic parquet writing — used by
`embeddings.py`, `finbert.py`, `surface.py`. Kept torch-free except for the lazily
imported `set_deterministic` so the CPU-only `surface` path and CI need no GPU stack.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.features.text.sections import SECTION_PREPARED, SECTION_QA

# Per-call feature "scopes": the two sections plus the whole call.
SCOPE_FULL = "full"
SCOPES = (SECTION_PREPARED, SECTION_QA, SCOPE_FULL)


def id_type_for(dataset: str) -> pa.DataType:
    return pa.int64() if dataset == "fincall" else pa.string()


def load_chunks(root: Path, dataset: str, *, limit: int | None = None) -> pd.DataFrame:
    """Load `data/{dataset}/chunks.parquet`; optionally the first `limit` calls (sorted)."""
    df = pd.read_parquet(root / dataset / "chunks.parquet")
    if limit is not None:
        keep = sorted(df["call_id"].unique())[:limit]
        df = df[df["call_id"].isin(keep)].reset_index(drop=True)
    return df


def content_hash(model_id: str, text: str) -> str:
    """Stable cache key for one text under one model (config-invariant inputs only)."""
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def set_deterministic(seed: int = 0) -> None:
    """Make frozen inference bit-identical across runs (fp32 + deterministic kernels)."""
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def scope_mask(sections: pd.Series, scope: str) -> pd.Series:
    return sections.notna() if scope == SCOPE_FULL else (sections == scope)


def pooled(vectors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Mean (weights=None) or weighted-mean pool of row vectors → one vector (float64)."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape[0] == 0:
        return np.full(v.shape[1], np.nan, dtype=np.float64)
    if weights is None:
        return v.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    tot = w.sum()
    if tot <= 0:
        return v.mean(axis=0)
    return (v * w[:, None]).sum(axis=0) / tot


def load_vector_cache(path: Path) -> dict[str, np.ndarray]:
    """hash -> float32 vector (the content-hash feature cache); {} if absent."""
    if not path.is_file():
        return {}
    t = pq.read_table(path)
    return {
        h: np.asarray(v, dtype=np.float32)
        for h, v in zip(t.column("hash").to_pylist(), t.column("vector").to_pylist(), strict=True)
    }


def save_vector_cache(cache: dict[str, np.ndarray], path: Path) -> None:
    """Deterministic cache parquet (sorted by hash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(cache.items())
    table = pa.table(
        {
            "hash": pa.array([h for h, _ in items], pa.string()),
            "vector": pa.array(
                [np.asarray(v, dtype=np.float32).tolist() for _, v in items], pa.list_(pa.float32())
            ),
        }
    )
    pq.write_table(table, path, compression="none", store_schema=True)


def encode_with_cache(texts, model_id, cache_path, encode_fn):
    """Return (aligned float32 matrix, n_newly_encoded). Dedupes by content hash; persists.

    `encode_fn(list[str]) -> ndarray` does its own internal batching. Only texts whose hash is
    not already cached are encoded, so re-runs are cache hits (bit-identical) and incremental.
    """
    texts = list(texts)
    cache = load_vector_cache(cache_path)
    hashes = [content_hash(model_id, t) for t in texts]
    missing = {h: t for h, t in zip(hashes, texts, strict=True) if h not in cache}
    if missing:
        mh = list(missing)
        embs = np.asarray(encode_fn([missing[h] for h in mh]), dtype=np.float32)
        for h, e in zip(mh, embs, strict=True):
            cache[h] = e
        save_vector_cache(cache, cache_path)
    out = np.vstack([cache[h] for h in hashes]) if hashes else np.zeros((0, 0), np.float32)
    return out.astype(np.float32), len(missing)


def write_feature_parquet(
    df: pd.DataFrame, path: Path, *, id_type: pa.DataType, sort_cols: list[str]
) -> None:
    """Deterministic parquet (sorted, compression none) — the T0.3 convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(sort_cols).reset_index(drop=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.set_column(
        table.schema.get_field_index("call_id"),
        "call_id",
        table.column("call_id").cast(id_type),
    )
    pq.write_table(table, path, compression="none", store_schema=True)
