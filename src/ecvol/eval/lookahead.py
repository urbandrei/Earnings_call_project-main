"""T7.2 — frozen-pipeline post-cutoff evaluation (DESIGN §7.4, the lookahead study).

Every stage's canonical head is fit **once, on FinCall's temporal training split**
(validation only for the ridge penalty), exactly as in Result Tables 1–4, and then
scored on two segments without any retraining, threshold change or feature
re-selection: the in-cutoff FinCall temporal **test** segment, and the
**post-cutoff** Earnings25 corpus (Q4 2025; after the training cut-off of every
frozen encoder in the stack — BGE-M3, FinBERT, WavLM, emotion2vec+). The same
extractors produced both feature matrices, so the columns align by construction.

Heads (the ridge canonicals of DECISIONS 2026-06-24, plus the Stage-0/1 baselines):

| stage | model | inputs |
|---|---|---|
| 0 | persistence | v_pre (level-v) / 0 (Δv) |
| 0 | har | train-fit log-HAR on rv_daily/weekly/monthly |
| 2 | ridge_text_pastvol | BGE-M3 section embeddings + FinBERT/surface + past-vol |
| 3 | ridge_wavlm_pastvol | WavLM + past-vol |
| 4 | ridge_fusion_pastvol | text + WavLM + eGeMAPS + emotion2vec+ + past-vol |

The Stage-1 ticker-fixed-effect GBDT is omitted by design: every Earnings25
company is unseen, so a ticker fixed effect has no post-cutoff meaning (which is
itself the point of DESIGN §3.1). Degradation per (stage, target, τ) is the
post-cutoff minus in-cutoff R²_OOS against persistence, with a cluster-bootstrap
CI (clusters = calendar month of day 0) on the post-cutoff R². Targets are the
primary (assume-after-hours) sets for both corpora so the anchor rule is held
fixed too. Output: `results/result_table_7.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import evaluate as E
from ecvol.eval import metrics as M
from ecvol.eval.significance import cluster_bootstrap_ci
from ecvol.features.audio.assemble import load_audio_blocks
from ecvol.features.text.assemble import build_text_matrix
from ecvol.models import baselines as B
from ecvol.models import heads

PASTVOL = ["v_pre", "rv_daily", "rv_weekly", "rv_monthly"]
TRAIN_DATASET = "fincall"
POST_DATASET = "earnings25"
TARGETS = ("v", "dv")
STAGES = ("persistence", "har", "ridge_text_pastvol", "ridge_wavlm_pastvol", "ridge_fusion_pastvol")


def _features(root: Path, dataset: str):
    text_df, t_emb, t_other = build_text_matrix(root, dataset)
    audio_df, blocks = load_audio_blocks(root, dataset)
    feat = text_df.merge(audio_df, on="call_id", how="outer")
    feat["call_id"] = feat["call_id"].astype(str)
    cols = {
        "ridge_text_pastvol": list(t_emb) + list(t_other) + PASTVOL,
        "ridge_wavlm_pastvol": list(blocks["wavlm"]) + PASTVOL,
        "ridge_fusion_pastvol": list(t_emb)
        + list(t_other)
        + list(blocks["wavlm"])
        + list(blocks["egemaps"])
        + list(blocks["emotion2vec"])
        + PASTVOL,
    }
    return feat, cols


def _impute(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X = X.copy()
    nan = np.isnan(X)
    if nan.any():
        X[nan] = np.take(med, np.where(nan)[1])
    return X


def run_lookahead(root: Path, *, n_resamples: int = 1000) -> pd.DataFrame:
    train_df = E.load_eval_frame(root, TRAIN_DATASET)
    post_df = E.load_eval_frame(root, POST_DATASET)
    assign = pd.read_csv(
        root / "splits" / f"{TRAIN_DATASET}_temporal.csv", dtype={"call_id": str}
    ).set_index("call_id")["split"]
    train_df["split"] = train_df["call_id"].map(assign).fillna("excluded")
    feat_tr, cols = _features(root, TRAIN_DATASET)
    feat_post, cols_post = _features(root, POST_DATASET)
    for k in cols:
        assert cols[k] == cols_post[k], f"feature columns differ for {k}"

    rows = []
    for tau in E.HORIZONS:
        at = train_df[train_df["horizon"] == tau].merge(feat_tr, on="call_id", how="left")
        po = post_df[post_df["horizon"] == tau].merge(feat_post, on="call_id", how="left")
        tr = (at["split"] == "train").to_numpy()
        va = (at["split"] == "val").to_numpy()
        te = (at["split"] == "test").to_numpy()
        # frozen HAR: coefficients from FinCall train only
        coef = B.har_fit(
            B.har_design(at.loc[tr, "rv_daily"], at.loc[tr, "rv_weekly"], at.loc[tr, "rv_monthly"]),
            at.loc[tr, "v_post"].to_numpy(),
        )
        har_in = B.har_predict(
            B.har_design(at["rv_daily"], at["rv_weekly"], at["rv_monthly"]), coef
        )
        har_post = B.har_predict(
            B.har_design(po["rv_daily"], po["rv_weekly"], po["rv_monthly"]), coef
        )
        for target in TARGETS:
            y_in = E.target_truth(at, target, har_in)
            y_post = E.target_truth(po, target, har_post)
            base_in = E.persistence_pred(at, target)
            base_post = E.persistence_pred(po, target)
            preds = {
                "persistence": (base_in, base_post),
                "har": (
                    E.vpost_to_target(har_in, at, target, har_in),
                    E.vpost_to_target(har_post, po, target, har_post),
                ),
            }
            fit = tr & np.isfinite(y_in)
            for stage, c in cols.items():
                X = at[c].to_numpy(dtype=np.float64)
                med = np.nanmedian(X[fit], axis=0)
                med = np.where(np.isnan(med), 0.0, med)
                X = _impute(X, med)
                Xp = _impute(po[c].to_numpy(dtype=np.float64), med)  # frozen: train medians
                scaler, model, _ = heads.ridge_fit(X[fit], y_in[fit], X[va], y_in[va])
                preds[stage] = (heads.predict(scaler, model, X), heads.predict(scaler, model, Xp))
            months = pd.to_datetime(po["as_of"]).dt.to_period("M").astype(str).to_numpy()
            for stage in STAGES:
                p_in, p_post = preds[stage]
                r2_in = float(M.r2_oos(y_in[te], p_in[te], base_in[te]))
                r2_post = float(M.r2_oos(y_post, p_post, base_post))
                # cluster bootstrap on the post-cutoff R² (clusters = month of day 0)
                err = (y_post - p_post) ** 2
                err_b = (y_post - base_post) ** 2
                keep = np.isfinite(err) & np.isfinite(err_b)
                idx = np.arange(len(err))[keep]

                def stat(sub_idx, _e=err, _b=err_b):
                    sub_idx = sub_idx.astype(int)
                    return 1.0 - _e[sub_idx].sum() / max(_b[sub_idx].sum(), 1e-12)

                _, lo, hi = cluster_bootstrap_ci(
                    idx, months[keep], statistic=stat, n_resamples=n_resamples, seed=0
                )
                rows.append(
                    {
                        "stage": stage,
                        "target": target,
                        "horizon": int(tau),
                        "n_in": int(te.sum()),
                        "n_post": int(keep.sum()),
                        "mse_in": float(M.mse(y_in[te], p_in[te])),
                        "mse_post": float(M.mse(y_post, p_post)),
                        "r2_oos_in": r2_in,
                        "r2_oos_post": r2_post,
                        "r2_oos_post_lo": float(lo),
                        "r2_oos_post_hi": float(hi),
                        "degradation": r2_post - r2_in,
                    }
                )
    table = pd.DataFrame(rows)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_7.csv", index=False, lineterminator="\n")
    return table
