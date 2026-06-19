"""Assemble the per-call Stage-2 text design matrix from the T3.2 feature parquets (T3.3).

Concatenates, per call: BGE-M3 embeddings for prepared_remarks + qa (2048-d), FinBERT
aggregates (scope x role x {pos,neg,neu,net}), and surface stats (scope x stat). Missing
scopes/roles (e.g. a call with no detected Q&A, or MAEC's absent speaker roles) are zero-filled.
Returns (matrix_df, embedding_cols, other_cols) so the caller can PCA-reduce the embedding block
for the MLP while ridge consumes the raw columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMB_DIM = 1024
EMB_SCOPES = ("prepared_remarks", "qa")
FB_SCOPES = ("prepared_remarks", "qa", "full")
FB_ROLES = ("all", "management", "analyst", "operator")
FB_METRICS = ("p_positive", "p_negative", "p_neutral", "net")
SF_SCOPES = ("prepared_remarks", "qa", "full")
SF_STATS = (
    "n_turns",
    "n_chunks",
    "n_words",
    "n_chars",
    "numeric_density",
    "question_marks",
    "words_per_turn",
)


def build_text_matrix(root: Path, dataset: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Per-call feature matrix; returns (df indexed by call_id col, emb_cols, other_cols)."""
    emb = pd.read_parquet(root / dataset / "text_embeddings.parquet")
    fb = pd.read_parquet(root / dataset / "text_finbert.parquet")
    sf = pd.read_parquet(root / dataset / "text_surface.parquet")
    for d in (emb, fb, sf):
        d["call_id"] = d["call_id"].astype(str)

    calls = sorted(set(emb["call_id"]))
    idx = {c: i for i, c in enumerate(calls)}
    n = len(calls)

    emb_cols = [f"e{j}" for j in range(EMB_DIM * len(EMB_SCOPES))]
    emat = np.zeros((n, len(emb_cols)), dtype=np.float64)
    for r in emb.itertuples(index=False):
        if r.scope in EMB_SCOPES:
            s = EMB_SCOPES.index(r.scope)
            emat[idx[r.call_id], s * EMB_DIM : (s + 1) * EMB_DIM] = np.asarray(r.vector, np.float64)

    fb_cols = [f"fb_{sc}_{ro}_{m}" for sc in FB_SCOPES for ro in FB_ROLES for m in FB_METRICS]
    fbpos = {c: i for i, c in enumerate(fb_cols)}
    fmat = np.zeros((n, len(fb_cols)), dtype=np.float64)
    for r in fb.itertuples(index=False):
        if r.scope in FB_SCOPES and r.role in FB_ROLES:
            for m in FB_METRICS:
                fmat[idx[r.call_id], fbpos[f"fb_{r.scope}_{r.role}_{m}"]] = getattr(r, m)

    sf_cols = [f"sf_{sc}_{st}" for sc in SF_SCOPES for st in SF_STATS]
    sfpos = {c: i for i, c in enumerate(sf_cols)}
    smat = np.zeros((n, len(sf_cols)), dtype=np.float64)
    for r in sf.itertuples(index=False):
        if r.scope in SF_SCOPES:
            for st in SF_STATS:
                smat[idx[r.call_id], sfpos[f"sf_{r.scope}_{st}"]] = getattr(r, st)

    other_cols = fb_cols + sf_cols
    df = pd.DataFrame(np.hstack([emat, fmat, smat]), columns=emb_cols + other_cols)
    df.insert(0, "call_id", calls)
    return df, emb_cols, other_cols
