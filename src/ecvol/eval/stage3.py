"""Stage-3 audio heads → Result Table 3 (DESIGN §6 Stage 3, §7; T4.4, FinCall-only).

Ridge + shallow-MLP heads (models/heads.py) on the frozen audio features (T4.1–T4.3): eGeMAPS,
WavLM-Large, emotion2vec+, WavLM+eGeMAPS, and a WavLM+text fusion preview. Two covariate variants
(audio, audio+past-vol), 5 seeds, val/test, all targets × horizons × splits. Each cell carries DM
p-values vs **persistence**, **HAR-RV** (Stage-0), **Stage-1** GBDT, and **Stage-2** text ridge —
the §7.5 confirmatory comparisons for RQ1-audio. Reuses the Table-1/2 harness (eval/evaluate.py,
eval/stage2.py). Output: long-format `data/results/result_table_3.csv` (MAEC has no audio).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import evaluate as E
from ecvol.eval.significance import diebold_mariano
from ecvol.features.audio.assemble import load_audio_blocks
from ecvol.features.text.assemble import build_text_matrix
from ecvol.models import baselines as B
from ecvol.models import heads

PASTVOL = ["v_pre", "rv_daily", "rv_weekly", "rv_monthly"]
HEAD_NAMES = ("ridge", "mlp")


@dataclass(frozen=True)
class FeatureSet:
    name: str
    emb_blocks: tuple[str, ...]  # embedding blocks (PCA-reduced for the MLP)
    dense_blocks: tuple[str, ...]  # dense blocks used raw
    text: bool = False  # also include the Stage-2 text matrix (fusion preview)


FEATURE_SETS = (
    FeatureSet("egemaps", (), ("egemaps",)),
    FeatureSet("wavlm", ("wavlm",), ()),
    FeatureSet("emotion2vec", ("emotion2vec",), ()),
    FeatureSet("wavlm_egemaps", ("wavlm",), ("egemaps",)),
    FeatureSet("wavlm_text", ("wavlm",), (), text=True),  # audio+text fusion preview
)
COVARIATES = ("audio", "audio_pastvol")


def _dm_p(y_true, y_pred, y_ref) -> float:
    return diebold_mariano(y_true - y_pred, y_true - y_ref).p_value


@dataclass
class _Cols:
    emb: list[str] = field(default_factory=list)
    dense: list[str] = field(default_factory=list)


def _resolve_cols(fs: FeatureSet, blocks, text_emb, text_other, covariate) -> _Cols:
    emb = [c for b in fs.emb_blocks for c in blocks[b]]
    dense = [c for b in fs.dense_blocks for c in blocks[b]]
    if fs.text:
        emb = emb + text_emb
        dense = dense + text_other
    if covariate == "audio_pastvol":
        dense = dense + PASTVOL
    return _Cols(emb=emb, dense=dense)


def _predict(head, at, tr, val, te, y_all, cols: _Cols, seeds):
    """(pred_val, pred_test, seed_std) for one head over an explicit emb/dense column split."""
    allcols = cols.emb + cols.dense
    X = at[allcols].to_numpy(dtype=np.float64).copy()
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    nan = np.isnan(X)
    if nan.any():
        X[nan] = np.take(med, np.where(nan)[1])
    fit = tr & np.isfinite(y_all)
    ei = list(range(len(cols.emb)))
    di = list(range(len(cols.emb), len(allcols)))
    if head == "ridge":
        pv, pte, _ = heads.ridge_fit_predict(X[fit], y_all[fit], X[val], y_all[val], X[te])
        return pv, pte, float("nan")
    if ei:
        e_fit, e_val, e_te = heads.pca_reduce(X[fit][:, ei], X[val][:, ei], X[te][:, ei])
        xtr = np.hstack([e_fit, X[fit][:, di]])
        xval = np.hstack([e_val, X[val][:, di]])
        xte = np.hstack([e_te, X[te][:, di]])
    else:
        xtr, xval, xte = X[fit], X[val], X[te]
    pvs, ptes = [], []
    for s in seeds:
        pv, pte = heads.mlp_fit_predict(xtr, y_all[fit], xval, xte, seed=s)
        pvs.append(pv)
        ptes.append(pte)
    tmask = np.isfinite(y_all[te])
    seed_std = (
        float(np.std([np.mean((y_all[te][tmask] - p[tmask]) ** 2) for p in ptes]))
        if tmask.any()
        else 0.0
    )
    return np.mean(pvs, axis=0), np.mean(ptes, axis=0), seed_std


def _stage2_ref(at, tr, val, te, y_all, text_emb, text_other) -> dict:
    """Stage-2 text reference (ridge on text+past-vol); per-segment preds for the DM comparison."""
    cols = _Cols(emb=list(text_emb), dense=list(text_other) + PASTVOL)
    pv, pte, _ = _predict("ridge", at, tr, val, te, y_all, cols, seeds=(0,))
    return {"val": pv, "test": pte}


def _row3(dataset, scheme, target, tau, model, seg, cell) -> dict:
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
        "dm_p_vs_stage2": float(cell["dm_p_vs_stage2"]),
    }


def evaluate_stage3(root: Path, dataset: str = "fincall", *, seeds=E.DEFAULT_SEEDS) -> list[dict]:
    df = E.load_eval_frame(root, dataset)
    audio_df, blocks = load_audio_blocks(root)
    text_df, text_emb, text_other = build_text_matrix(root, dataset)
    feat = audio_df.merge(text_df, on="call_id", how="outer")
    all_feat_cols = [c for c in feat.columns if c != "call_id"]
    rows: list[dict] = []
    for scheme in E.SPLIT_SCHEMES:
        split_csv = root / "splits" / f"{dataset}_{scheme}.csv"
        if not split_csv.is_file():
            continue
        assign = pd.read_csv(split_csv, dtype={"call_id": str}).set_index("call_id")["split"]
        for tau in E.HORIZONS:
            at = df[df["horizon"] == tau].copy()
            at["split"] = at["call_id"].map(assign).fillna("excluded")
            at = at.merge(feat, on="call_id", how="left")
            at[all_feat_cols] = at[all_feat_cols].fillna(0.0)
            tr = (at["split"] == "train").to_numpy()
            val = (at["split"] == "val").to_numpy()
            te = (at["split"] == "test").to_numpy()
            if tr.sum() == 0 or te.sum() == 0:
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
                stage2_ref = _stage2_ref(at, tr, val, te, y_all, text_emb, text_other)
                for fs in FEATURE_SETS:
                    covs = ("audio_pastvol",) if fs.text else COVARIATES
                    for covariate in covs:
                        cols = _resolve_cols(fs, blocks, text_emb, text_other, covariate)
                        for head in HEAD_NAMES:
                            pv, pte, seed_std = _predict(head, at, tr, val, te, y_all, cols, seeds)
                            for seg, mask, pred in (("val", val, pv), ("test", te, pte)):
                                if mask.sum() == 0:
                                    continue
                                yt, base = y_all[mask], base_all[mask]
                                cell = E._cell_metrics(at[mask], yt, pred, base)
                                cell["seed_std"] = seed_std
                                cell["dm_p_vs_persistence"] = _dm_p(yt, pred, base)
                                cell["dm_p_vs_har"] = _dm_p(yt, pred, har_ref[mask])
                                cell["dm_p_vs_stage1"] = _dm_p(yt, pred, stage1_ref[mask])
                                cell["dm_p_vs_stage2"] = _dm_p(yt, pred, stage2_ref[seg])
                                name = f"{head}_{fs.name}_{covariate}"
                                rows.append(_row3(dataset, scheme, target, tau, name, seg, cell))
    return rows


def run_stage3(root: Path, *, seeds=E.DEFAULT_SEEDS) -> pd.DataFrame:
    """Evaluate Stage-3 audio heads (FinCall) and write Result Table 3."""
    rows = (
        evaluate_stage3(root, "fincall", seeds=seeds)
        if (root / "fincall" / "audio_wavlm.parquet").is_file()
        else []
    )
    table = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "split", "target", "horizon", "model", "segment"])
        .reset_index(drop=True)
    )
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_3.csv", index=False, lineterminator="\n")
    return table
