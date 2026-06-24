"""κ-audit: the T6.2 content gate for LLM features.

Constrained decoding guarantees schema-*valid* output; this scores whether it is schema-
*correct* by comparing model output to human rubric labels on the 50-call train-only audit
sample (`ecvol featurize llm-audit-sample`). The gate (DESIGN §6 / TASKS T6.2): **κ > 0.6**
on the categorical/ordinal fields before any corpus-scale run is trusted.

Agreement metric per field group (schema.py): Cohen's κ for the categorical
(`guidance_direction`), linearly-weighted κ for the ordinals (the 0–4 scales), and κ on the
present/absent binarization of the `surprise_mentions` count. Q&A-only fields are scored only
on Q&A rows (prepared-remarks cells are the N/A floor — `applicable_fields`).

`kappa_matrix` cross-validates models against each other (the exploration question: do
stronger models agree more / produce more signal?).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .schema import (
    CATEGORICAL_FIELDS,
    COUNT_FIELDS,
    LABEL_FIELDS,
    ORDINAL_FIELDS,
    applicable_fields,
)

KAPPA_THRESHOLD = 0.6


def _load_labels(sheet_csv: str | Path) -> pd.DataFrame:
    # keep_default_na=False so the "NA" not-applicable marker survives (pandas would coerce it)
    return pd.read_csv(sheet_csv, keep_default_na=False, dtype={"call_id": str})


def _load_features(features_parquet: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(features_parquet)
    df["call_id"] = df["call_id"].astype(str)
    return df


def _field_kappa(field: str, a: list, b: list) -> float:
    """κ for one field, using the right variant for its type."""
    if field in CATEGORICAL_FIELDS:
        return cohen_kappa_score([str(x) for x in a], [str(x) for x in b])
    if field in ORDINAL_FIELDS:
        return cohen_kappa_score([int(x) for x in a], [int(x) for x in b], weights="linear")
    if field in COUNT_FIELDS:  # binarize present/absent
        return cohen_kappa_score([int(int(x) > 0) for x in a], [int(int(x) > 0) for x in b])
    raise ValueError(f"unknown field {field!r}")


def _aligned(merged: pd.DataFrame, field: str, a_col: str, b_col: str) -> tuple[list, list]:
    """Rows where the field applies to the section and both sides have a usable value."""
    a_vals, b_vals = [], []
    for _, r in merged.iterrows():
        if field not in applicable_fields(r["section"]):
            continue
        av, bv = str(r[a_col]).strip(), str(r[b_col]).strip()
        if av in ("", "NA") or bv in ("", "NA"):
            continue
        a_vals.append(av)
        b_vals.append(bv)
    return a_vals, b_vals


def compute_kappa(sheet_csv: str | Path, features_parquet: str | Path) -> dict[str, dict]:
    """Per-field κ of model vs. human. Returns ``{field: {"kappa": float|None, "n": int}}``."""
    labels = _load_labels(sheet_csv)
    model = _load_features(features_parquet)
    merged = labels.merge(model, on=["call_id", "section"], suffixes=("_h", "_m"))
    out: dict[str, dict] = {}
    for field in LABEL_FIELDS:
        a, b = _aligned(merged, field, f"{field}_h", f"{field}_m")
        if len(a) < 2:
            out[field] = {"kappa": None, "n": len(a)}
        else:
            out[field] = {"kappa": float(_field_kappa(field, a, b)), "n": len(a)}
    return out


def passes_gate(kappas: dict[str, dict], thr: float = KAPPA_THRESHOLD) -> bool:
    """True iff every scored field clears ``thr`` (unscored fields — too few labels — ignored)."""
    vals = [v["kappa"] for v in kappas.values() if v["kappa"] is not None]
    return bool(vals) and all(k >= thr for k in vals)


def kappa_matrix(
    features_parquets: dict[str, str | Path],
) -> dict[tuple[str, str], dict[str, dict]]:
    """Pairwise model-vs-model per-field κ over shared ``(call, section)`` — the cross-val table."""
    frames = {name: _load_features(p) for name, p in features_parquets.items()}
    names = sorted(frames)
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for i, na in enumerate(names):
        for nb in names[i + 1 :]:
            merged = frames[na].merge(frames[nb], on=["call_id", "section"], suffixes=("_h", "_m"))
            pair: dict[str, dict] = {}
            for field in LABEL_FIELDS:
                a, b = _aligned(merged, field, f"{field}_h", f"{field}_m")
                pair[field] = (
                    {"kappa": float(_field_kappa(field, a, b)), "n": len(a)}
                    if len(a) >= 2
                    else {"kappa": None, "n": len(a)}
                )
            out[(na, nb)] = pair
    return out
