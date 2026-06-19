"""Frozen FinBERT sentiment aggregates (T3.2, GPU).

Each T3.1 chunk is classified once (content-hash cached) into ProsusAI/finbert's
positive/negative/neutral probabilities, then averaged per call scope (prepared_remarks / qa /
full) and speaker role (management / analyst / operator / all), with a net = p_pos - p_neg.
fp32 + deterministic (DECISIONS 2026-06-19). Output: `data/{dataset}/text_finbert.parquet`.
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
    set_deterministic,
    write_feature_parquet,
)

FINBERT_MODEL = "ProsusAI/finbert"
ROLES = ("management", "analyst", "operator")
FINBERT_SOURCE = "derived: ecvol featurize text (ProsusAI/finbert, T3.2)"
FINBERT_LICENSE = "derived; model ProsusAI/finbert"
MAX_LENGTH = 512  # FinBERT (BERT) context; our <=320-word chunks mostly fit, tail truncated


def load_model(device: str = "cuda"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_deterministic()
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(device).eval()
    return tok, model


def labels_of(model) -> list[str]:
    """Class labels in logit-index order (so cached prob vectors map consistently)."""
    id2label = model.config.id2label
    return [id2label[i].lower() for i in range(len(id2label))]


def _classifier(tok, model, device: str, batch_size: int):
    import torch

    def encode(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)
            out.append(probs.cpu().numpy())
        return np.vstack(out).astype(np.float32)

    return encode


def aggregate(chunks: pd.DataFrame, probs: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Mean class probabilities per (call_id, scope, role) + net = p_positive - p_negative."""
    call_ids = chunks["call_id"].to_numpy()
    sections = chunks["section"].to_numpy()
    roles = chunks["role"].to_numpy()
    sources = chunks["source"].to_numpy()
    li = {label: i for i, label in enumerate(labels)}
    pos, neg = li.get("positive"), li.get("negative")
    rows: list[dict] = []
    for call_id in sorted(pd.unique(call_ids)):
        in_call = call_ids == call_id
        source = sources[in_call][0]
        for scope in SCOPES:
            sm = in_call if scope == SCOPE_FULL else (in_call & (sections == scope))
            if not sm.any():
                continue
            for role in ("all", *ROLES):
                rm = sm if role == "all" else (sm & (roles == role))
                if not rm.any():
                    continue
                mean = probs[rm].mean(axis=0)
                row = {
                    "call_id": call_id,
                    "source": source,
                    "scope": scope,
                    "role": role,
                    "n_chunks": int(rm.sum()),
                }
                for label in labels:
                    row[f"p_{label}"] = float(mean[li[label]])
                if pos is not None and neg is not None:
                    row["net"] = float(mean[pos] - mean[neg])
                rows.append(row)
    return pd.DataFrame(rows)


def build(
    root: Path,
    dataset: str,
    *,
    limit: int | None = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> tuple[int, int]:
    """Classify + aggregate one dataset; returns (feature_rows, newly_classified_chunks)."""
    chunks = load_chunks(root, dataset, limit=limit)
    tok, model = load_model(device)
    cache_path = root / dataset / "cache" / "text_finbert.parquet"
    probs, n_new = encode_with_cache(
        chunks["text"].tolist(),
        FINBERT_MODEL,
        cache_path,
        _classifier(tok, model, device, batch_size),
    )
    df = aggregate(chunks, probs, labels_of(model))
    out = root / dataset / "text_finbert.parquet"
    write_feature_parquet(
        df, out, id_type=id_type_for(dataset), sort_cols=["call_id", "scope", "role"]
    )
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    entry = make_entry(out, root, source_url=FINBERT_SOURCE, license=FINBERT_LICENSE)
    write_manifest([entry], root / "manifests" / f"{dataset}_text_finbert.json")
    return len(df), n_new
