"""T6R.2 — reproduce a published model on its own dataset, then re-run it under our controls.

First model: **HTML** (Yang et al. 2020) on **EC** (Qin & Yang 2019). Three split
conditions × two label sources:

- `published` split (VolTAGE `*_split3.csv`, 385/54/109 of the priced calls) with the
  **published labels** (`future_τ` / auxiliary `future_Single_τ` from the shipped
  files) — the reproduction step; the paper reports text-only MSE 1.175 / 0.372 /
  0.153 / 0.133 at τ = 3 / 7 / 15 / 30 on 576 calls (Table 2), our copy of the split
  holds 560.
- `published` split with **our labels** (`targets.parquet`, trading-day v_post;
  auxiliary v_post(3)) — isolates the label effect.
- `temporal` (30-session embargo) and `ticker_disjoint` splits with our labels —
  the controlled condition (RQ3).

Persistence (`past_τ` for the published labels, `v_pre` for ours) is reported on
every cell as the floor. Sequences are the call's chunk embeddings in transcript
order, rebuilt from the BGE-M3 content-hash cache. Output: long-format
`results/result_table_6r.csv` (+ `model=html` rows carry the seed std).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import metrics as M
from ecvol.features.text._common import content_hash
from ecvol.features.text.embeddings import EMB_MODEL
from ecvol.models import html_head as H

LINEAGE_DATA = "raw/ref/lineage/VolTAGE/semi-supervised-gcn/gcn/data"
SINGLE = "raw/ref/lineage/KeFVP/price_data"
TAUS = (3, 7, 15, 30)
PUBLISHED_HTML_TEXT_MSE = {3: 1.175, 7: 0.372, 15: 0.153, 30: 0.133}


def chunk_sequences(root: Path, dataset: str) -> dict[str, np.ndarray]:
    """{call_id: [n_chunks, 1024] in transcript order} from chunks.parquet + the cache."""
    chunks = pd.read_parquet(
        root / dataset / "chunks.parquet", columns=["call_id", "turn_idx", "chunk_in_turn", "text"]
    )
    cache = pd.read_parquet(root / dataset / "cache" / "text_emb_bge_m3.parquet")
    vec = dict(zip(cache["hash"], cache["vector"], strict=True))
    chunks = chunks.sort_values(["call_id", "turn_idx", "chunk_in_turn"])
    out: dict[str, np.ndarray] = {}
    for cid, g in chunks.groupby("call_id", sort=False):
        rows = [vec.get(content_hash(EMB_MODEL, t)) for t in g["text"]]
        rows = [np.asarray(r, dtype=np.float32) for r in rows if r is not None]
        if rows:
            out[str(cid)] = np.vstack(rows)
    return out


def published_labels(root: Path) -> pd.DataFrame:
    """call_id → future_τ / past_τ (VolTAGE split files) + future_Single_τ (KeFVP)."""
    frames = []
    for s in ("train", "val", "test"):
        d = pd.read_csv(root / LINEAGE_DATA / f"{s}_split3.csv")
        single = pd.read_csv(root / SINGLE / f"{s}_split_SeriesSingleDayVol3.csv")
        d = d.merge(
            single[["text_file_name", *[f"future_Single_{t}" for t in TAUS]]],
            on="text_file_name",
            how="left",
        )
        frames.append(d)
    lab = pd.concat(frames, ignore_index=True).drop_duplicates("text_file_name")
    return lab.rename(columns={"text_file_name": "call_id"}).set_index("call_id")


def our_labels(root: Path, dataset: str) -> pd.DataFrame:
    t = pd.read_parquet(root / dataset / "targets.parquet")
    t = t[t["status"] == "ok"]
    t["call_id"] = t["call_id"].astype(str)
    wide = t.pivot(index="call_id", columns="horizon", values="v_post")
    pre = t.pivot(index="call_id", columns="horizon", values="v_pre")
    wide.columns = [f"future_{c}" for c in wide.columns]
    pre.columns = [f"past_{c}" for c in pre.columns]
    out = wide.join(pre)
    for tau in TAUS:
        out[f"future_Single_{tau}"] = wide["future_3"]  # auxiliary: the shortest horizon
    return out


def _cells(split_df: pd.DataFrame, labels: pd.DataFrame, seqs: dict, tau: int):
    d = split_df[split_df["call_id"].isin(labels.index) & split_df["call_id"].isin(seqs)].copy()
    d = d.dropna(subset=[]).reset_index(drop=True)
    y = labels.loc[d["call_id"], f"future_{tau}"].to_numpy(dtype=float)
    ya = labels.loc[d["call_id"], f"future_Single_{tau}"].to_numpy(dtype=float)
    base = labels.loc[d["call_id"], f"past_{tau}"].to_numpy(dtype=float)
    keep = np.isfinite(y) & np.isfinite(ya) & np.isfinite(base)
    d, y, ya, base = d[keep].reset_index(drop=True), y[keep], ya[keep], base[keep]
    S = [seqs[c] for c in d["call_id"]]
    idx = {s: np.where(d["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}
    return d, y, ya, base, S, idx


def run_html_reproduction(
    root: Path, *, dataset: str = "ec", seeds=(0, 1, 2), epochs: int = H.EPOCHS
) -> pd.DataFrame:
    seqs = chunk_sequences(root, dataset)
    label_sets = {"published": published_labels(root), "ours": our_labels(root, dataset)}
    rows = []
    for scheme, label_names in (
        ("published", ("published", "ours")),
        ("temporal", ("ours",)),
        ("ticker_disjoint", ("ours",)),
    ):
        split_df = pd.read_csv(root / "splits" / f"{dataset}_{scheme}.csv", dtype={"call_id": str})
        for label_name in label_names:
            labels = label_sets[label_name]
            for tau in TAUS:
                d, y, ya, base, S, idx = _cells(split_df, labels, seqs, tau)
                tr, va, te = idx["train"], idx["val"], idx["test"]
                if len(tr) < 10 or len(te) == 0:
                    continue
                # α tuned on validation (seed 0), then retrain on train+val, test with all seeds
                best_alpha, best = None, np.inf
                for a in H.ALPHAS if len(va) else (0.5,):
                    pv = H.fit_predict(S, y, ya, tr, va, alpha=a, seed=seeds[0], epochs=epochs)
                    v = float(M.mse(y[va], pv)) if len(va) else 0.0
                    if v < best:
                        best_alpha, best = a, v
                trva = np.concatenate([tr, va])
                preds = [
                    H.fit_predict(S, y, ya, trva, te, alpha=best_alpha, seed=s, epochs=epochs)
                    for s in seeds
                ]
                mses = [float(M.mse(y[te], p)) for p in preds]
                mean_pred = np.mean(preds, axis=0)
                common = {
                    "dataset": dataset,
                    "split": scheme,
                    "labels": label_name,
                    "horizon": tau,
                    "n_train": int(len(trva)),
                    "n_test": int(len(te)),
                }
                rows.append(
                    {
                        **common,
                        "model": "html_text",
                        "alpha": best_alpha,
                        "mse": float(np.mean(mses)),
                        "mse_seed_std": float(np.std(mses)),
                        "r2_oos_vs_persistence": float(M.r2_oos(y[te], mean_pred, base[te])),
                        "published_mse": PUBLISHED_HTML_TEXT_MSE[tau]
                        if (scheme, label_name) == ("published", "published")
                        else np.nan,
                    }
                )
                rows.append(
                    {
                        **common,
                        "model": "persistence",
                        "alpha": np.nan,
                        "mse": float(M.mse(y[te], base[te])),
                        "mse_seed_std": 0.0,
                        "r2_oos_vs_persistence": 0.0,
                        "published_mse": np.nan,
                    }
                )
    table = pd.DataFrame(rows)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_6r.csv", index=False, lineterminator="\n")
    return table
