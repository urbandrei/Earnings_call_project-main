"""Surface text statistics per call section (T3.2, CPU).

Cheap, interpretable features over the T1.3/T3.1 chunks — length, turn counts, numeric
density, question marks — computed per scope (prepared_remarks, qa, full). No model, no GPU;
deterministic. Output: `data/{dataset}/text_surface.parquet` (+ manifest).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ecvol.data.manifests import make_entry, write_manifest
from ecvol.features.text._common import (
    SCOPES,
    id_type_for,
    load_chunks,
    scope_mask,
    write_feature_parquet,
)

SURFACE_SOURCE = "derived: ecvol featurize text (surface, T3.2)"
SURFACE_LICENSE = "derived"

# A token counts as "numeric" if it carries a digit (covers 12, 3.4, 12%, $1.2, 2020, -5).
_NUMERIC = re.compile(r"\d")
_TOKEN = re.compile(r"\S+")


def _numeric_density(text: str) -> tuple[int, int]:
    toks = _TOKEN.findall(text or "")
    if not toks:
        return 0, 0
    return sum(1 for t in toks if _NUMERIC.search(t)), len(toks)


def compute(chunks: pd.DataFrame) -> pd.DataFrame:
    """Per (call_id, scope) surface stats; one row per scope present for the call."""
    rows: list[dict] = []
    num, tot = zip(*chunks["text"].map(_numeric_density), strict=True) if len(chunks) else ([], [])
    chunks = chunks.assign(_num=list(num), _tot=list(tot), _q=chunks["text"].str.count(r"\?"))
    for call_id, g in chunks.groupby("call_id", sort=True):
        source = g["source"].iloc[0]
        for scope in SCOPES:
            sg = g[scope_mask(g["section"], scope)]
            if len(sg) == 0:
                continue
            n_turns = sg["turn_idx"].nunique()
            n_words = int(sg["n_words"].sum())
            tot_tokens = int(sg["_tot"].sum())
            rows.append(
                {
                    "call_id": call_id,
                    "source": source,
                    "scope": scope,
                    "n_turns": int(n_turns),
                    "n_chunks": int(len(sg)),
                    "n_words": n_words,
                    "n_chars": int(sg["n_chars"].sum()),
                    "numeric_density": (int(sg["_num"].sum()) / tot_tokens) if tot_tokens else 0.0,
                    "question_marks": int(sg["_q"].sum()),
                    "words_per_turn": (n_words / n_turns) if n_turns else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build(root: Path, dataset: str, *, limit: int | None = None) -> int:
    """Compute + write surface features for one dataset; returns the row count."""
    chunks = load_chunks(root, dataset, limit=limit)
    df = compute(chunks)
    out = root / dataset / "text_surface.parquet"
    write_feature_parquet(df, out, id_type=id_type_for(dataset), sort_cols=["call_id", "scope"])
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    entry = make_entry(out, root, source_url=SURFACE_SOURCE, license=SURFACE_LICENSE)
    write_manifest([entry], root / "manifests" / f"{dataset}_text_surface.json")
    return len(df)
