"""Assemble per-call audio feature matrices for the Stage-3 heads (T4.4, FinCall-only).

Loads the three T4.x audio parquets — eGeMAPS (88-d functionals), WavLM-Large (1024-d),
emotion2vec+ (1024-d) — into per-call column blocks the Stage-3 evaluator composes into feature
sets (eGeMAPS / WavLM / emotion2vec+ / WavLM+eGeMAPS / +text fusion). eGeMAPS is a dense block;
WavLM and emotion2vec+ are embedding blocks (PCA-reduced for the MLP). Missing calls are absent
rows (the evaluator inner-joins on the split cohort).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EGEMAPS_PARQUET = "fincall/audio_egemaps.parquet"
WAVLM_PARQUET = "fincall/audio_wavlm.parquet"
E2V_PARQUET = "fincall/audio_emotion2vec.parquet"


def _vec_block(path: Path, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    """Explode a `vector` list-column parquet into prefixed scalar columns + call_id (str)."""
    d = pd.read_parquet(path, columns=["call_id", "vector"])
    d["call_id"] = d["call_id"].astype(str)
    mat = np.vstack(d["vector"].map(np.asarray).to_numpy())
    cols = [f"{prefix}{i}" for i in range(mat.shape[1])]
    out = pd.DataFrame(mat, columns=cols)
    out.insert(0, "call_id", d["call_id"].to_numpy())
    return out, cols


def load_audio_blocks(root: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Per-call audio matrix (outer-joined on call_id) + a {block_name: columns} map.

    Blocks: `egemaps` (dense, 88), `wavlm` (emb, 1024), `emotion2vec` (emb, 1024).
    """
    eg = pd.read_parquet(root / EGEMAPS_PARQUET)
    eg["call_id"] = eg["call_id"].astype(str)
    eg_cols = [c for c in eg.columns if c != "call_id"]

    wav, wav_cols = _vec_block(root / WAVLM_PARQUET, "wav")
    e2v, e2v_cols = _vec_block(root / E2V_PARQUET, "e2v")

    df = eg.merge(wav, on="call_id", how="outer").merge(e2v, on="call_id", how="outer")
    blocks = {"egemaps": eg_cols, "wavlm": wav_cols, "emotion2vec": e2v_cols}
    return df, blocks
