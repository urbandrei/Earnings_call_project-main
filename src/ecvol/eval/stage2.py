"""Stage-2 content models → Result Table 2 (DESIGN §6 Stage 2, §7).

Ridge + shallow-MLP heads (models/heads.py) on the frozen text features (T3.2), across every
(dataset x split x target x horizon), three covariate variants — **text**, **pastvol**,
**text_pastvol** — and the val/test segments, 5 seeds. Reuses the Table-1 harness
(eval/evaluate.py) for the eval frame, the train-only HAR fit, target transforms, per-cell
metrics, and the Stage-1 ticker-FE GBDT. Each cell carries DM p-values vs **persistence** (the
R²_OOS baseline), vs **HAR-RV** (Stage-0), and vs **Stage-1** GBDT — the §7.5 confirmatory
comparisons. Output: long-format `data/results/result_table_2.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import evaluate as E
from ecvol.eval.significance import diebold_mariano
from ecvol.features.text.assemble import build_text_matrix
from ecvol.models import baselines as B
from ecvol.models import heads

VARIANTS = ("text", "pastvol", "text_pastvol")
PASTVOL = ["v_pre", "rv_daily", "rv_weekly", "rv_monthly"]
HEAD_NAMES = ("ridge", "mlp")


def _dm_p(y_true, y_pred, y_ref) -> float:
    return diebold_mariano(y_true - y_pred, y_true - y_ref).p_value


def _variant_cols(variant, emb_cols, other_cols):
    """(all_cols, emb_block, non_emb_block) for a covariate variant."""
    if variant == "text":
        return emb_cols + other_cols, emb_cols, other_cols
    if variant == "pastvol":
        return list(PASTVOL), [], list(PASTVOL)
    return emb_cols + other_cols + PASTVOL, emb_cols, other_cols + PASTVOL


def _predict(head, at, tr, val, te, y_all, variant, emb_cols, other_cols, seeds):
    """(pred_val, pred_test, seed_std) for one head x variant at this (target, τ)."""
    cols, emb, other = _variant_cols(variant, emb_cols, other_cols)
    X = at[cols].to_numpy(dtype=np.float64).copy()  # writeable (for in-place impute)
    # Impute missing covariates with the train-column median (leakage-safe); a few MAEC rows
    # have NaN rv_monthly — insufficient 22-session history though the target is computable.
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    nan = np.isnan(X)
    if nan.any():
        X[nan] = np.take(med, np.where(nan)[1])
    fit = tr & np.isfinite(y_all)  # fit only on finite-target train rows (har_resid has NaNs)
    ytr = y_all[fit]
    tmask = np.isfinite(y_all[te])
    if head == "ridge":
        pv, pte, _ = heads.ridge_fit_predict(X[fit], ytr, X[val], y_all[val], X[te])
        return pv, pte, float("nan")
    # MLP: train-fit PCA on the embedding block (if present), concat the rest.
    if emb:
        ei = [cols.index(c) for c in emb]
        oi = [cols.index(c) for c in other]
        e_fit, e_val, e_te = heads.pca_reduce(X[fit][:, ei], X[val][:, ei], X[te][:, ei])
        xtr = np.hstack([e_fit, X[fit][:, oi]])
        xval = np.hstack([e_val, X[val][:, oi]])
        xte = np.hstack([e_te, X[te][:, oi]])
    else:
        xtr, xval, xte = X[fit], X[val], X[te]
    pvs, ptes = [], []
    for s in seeds:
        pv, pte = heads.mlp_fit_predict(xtr, ytr, xval, xte, seed=s)
        pvs.append(pv)
        ptes.append(pte)
    seed_std = (
        float(np.std([np.mean((y_all[te][tmask] - p[tmask]) ** 2) for p in ptes]))
        if tmask.any()
        else 0.0
    )
    return np.mean(pvs, axis=0), np.mean(ptes, axis=0), seed_std


def _row2(dataset, scheme, target, tau, model, seg, cell) -> dict:
    return {
        "dataset": dataset,
        "split": scheme,
        "target": target,
        "horizon": int(tau),
        "model": model,
        "segment": seg,
        "n": int(cell["n"]),
        "mse": float(cell["mse"]),
        "mae": float(cell["mae"]),
        "r2_oos": float(cell["r2_oos"]),
        "spearman_q": float(cell["spearman_q"]),
        "seed_std": float(cell.get("seed_std", np.nan)),
        "dm_p_vs_persistence": float(cell["dm_p_vs_persistence"]),
        "dm_p_vs_har": float(cell["dm_p_vs_har"]),
        "dm_p_vs_stage1": float(cell["dm_p_vs_stage1"]),
    }


def evaluate_stage2_dataset(df, text_df, emb_cols, other_cols, dataset, splits_dir, *, seeds):
    feat_cols = emb_cols + other_cols
    rows: list[dict] = []
    for scheme in E.SPLIT_SCHEMES:
        split_csv = splits_dir / f"{dataset}_{scheme}.csv"
        if not split_csv.is_file():
            continue
        assign = pd.read_csv(split_csv, dtype={"call_id": str}).set_index("call_id")["split"]
        for tau in E.HORIZONS:
            at = df[df["horizon"] == tau].copy()
            at["split"] = at["call_id"].map(assign).fillna("excluded")
            at = at.merge(text_df, on="call_id", how="left")
            at[feat_cols] = at[feat_cols].fillna(0.0)
            tr = (at["split"] == "train").to_numpy()
            val = (at["split"] == "val").to_numpy()
            te = (at["split"] == "test").to_numpy()
            if tr.sum() == 0:
                continue
            x_all = B.har_design(at["rv_daily"], at["rv_weekly"], at["rv_monthly"])
            coef = B.har_fit(
                B.har_design(
                    at.loc[tr, "rv_daily"], at.loc[tr, "rv_weekly"], at.loc[tr, "rv_monthly"]
                ),
                at.loc[tr, "v_post"].to_numpy(),
            )
            har_vpost = B.har_predict(x_all, coef)
            for target in E.TARGETS:
                y_all = E.target_truth(at, target, har_vpost)
                base_all = E.persistence_pred(at, target)
                har_ref = E.vpost_to_target(har_vpost, at, target, har_vpost)
                gbdt = E._gbdt_predictions(at, target, y_all, seeds)
                stage1_ref = np.mean(list(gbdt.values()), axis=0)
                for variant in VARIANTS:
                    for head in HEAD_NAMES:
                        pv, pte, seed_std = _predict(
                            head, at, tr, val, te, y_all, variant, emb_cols, other_cols, seeds
                        )
                        for seg, mask, pred in (("val", val, pv), ("test", te, pte)):
                            if mask.sum() == 0:
                                continue
                            yt, base = y_all[mask], base_all[mask]
                            cell = E._cell_metrics(at[mask], yt, pred, base)
                            cell["seed_std"] = seed_std
                            cell["dm_p_vs_persistence"] = _dm_p(yt, pred, base)
                            cell["dm_p_vs_har"] = _dm_p(yt, pred, har_ref[mask])
                            cell["dm_p_vs_stage1"] = _dm_p(yt, pred, stage1_ref[mask])
                            rows.append(
                                _row2(dataset, scheme, target, tau, f"{head}_{variant}", seg, cell)
                            )
    return rows


def run_stage2(root: Path, *, seeds=E.DEFAULT_SEEDS) -> pd.DataFrame:
    """Evaluate Stage-2 heads on every dataset and write Result Table 2."""
    all_rows: list[dict] = []
    for dataset in E.DATASETS:
        if not (root / dataset / "text_embeddings.parquet").is_file():
            continue
        df = E.load_eval_frame(root, dataset)
        text_df, emb_cols, other_cols = build_text_matrix(root, dataset)
        all_rows.extend(
            evaluate_stage2_dataset(
                df, text_df, emb_cols, other_cols, dataset, root / "splits", seeds=seeds
            )
        )
    table = (
        pd.DataFrame(all_rows)
        .sort_values(["dataset", "split", "target", "horizon", "model", "segment"])
        .reset_index(drop=True)
    )
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "result_table_2.csv", index=False, lineterminator="\n")
    return table
