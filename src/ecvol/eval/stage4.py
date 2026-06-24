"""Stage-4 fusion evaluation → fusion rows for Result Table 4 (T5.1; DESIGN §6 Stage 4, §7).

Evaluates the two fusion heads (gated fusion, late-fusion stacking; models/fusion.py) over the
frozen modality embeddings + past-vol, with covariates {in, out}, 5 seeds, across split × target ×
horizon. Each cell carries DM p-values vs **persistence**, **Stage-1** GBDT, **Stage-2** text
(best unimodal text), and **Stage-3** audio (best unimodal audio) — the §7.5 "Stage-4 vs best
unimodal" confirmatory comparison. Reuses the Stage-2/3 harness. FinCall-only (audio needed).
Output: long-format `data/results/result_table_4_fusion.csv` (T5.2 merges the full grid).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import evaluate as E
from ecvol.eval.significance import diebold_mariano
from ecvol.eval.stage3 import PASTVOL, _Cols, _predict
from ecvol.features.audio.assemble import load_audio_blocks
from ecvol.features.text.assemble import build_text_matrix
from ecvol.models import baselines as B
from ecvol.models import fusion

COVARIATES = ("fusion", "fusion_pastvol")


def _block(frame, cols, mask):
    """Masked feature block as float64 (empty (n,0) array when no columns)."""
    return frame.loc[mask, cols].to_numpy(np.float64) if cols else np.zeros((int(mask.sum()), 0))


def _dm(y, p, ref):
    return diebold_mariano(y - p, y - ref).p_value


def _row(dataset, scheme, target, tau, model, seg, cell):
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
        "dm_p_vs_stage1": float(cell["dm_p_vs_stage1"]),
        "dm_p_vs_stage2": float(cell["dm_p_vs_stage2"]),
        "dm_p_vs_stage3": float(cell["dm_p_vs_stage3"]),
    }


def evaluate_stage4(root: Path, dataset="fincall", *, seeds=E.DEFAULT_SEEDS) -> list[dict]:
    df = E.load_eval_frame(root, dataset)
    audio_df, ablocks = load_audio_blocks(root)
    text_df, t_emb, t_other = build_text_matrix(root, dataset)
    feat = audio_df.merge(text_df, on="call_id", how="outer")
    fcols = [c for c in feat.columns if c != "call_id"]
    wav, e2v, eg = ablocks["wavlm"], ablocks["emotion2vec"], ablocks["egemaps"]
    rows: list[dict] = []
    for scheme in E.SPLIT_SCHEMES:
        sp = root / "splits" / f"{dataset}_{scheme}.csv"
        if not sp.is_file():
            continue
        assign = pd.read_csv(sp, dtype={"call_id": str}).set_index("call_id")["split"]
        for tau in E.HORIZONS:
            at = df[df["horizon"] == tau].copy()
            at["split"] = at["call_id"].map(assign).fillna("excluded")
            at = at.merge(feat, on="call_id", how="left")
            at[fcols] = at[fcols].fillna(0.0)
            tr = (at["split"] == "train").to_numpy()
            va = (at["split"] == "val").to_numpy()
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

            def block(cols, mask, _frame=at):
                return _block(_frame, cols, mask)

            for target in E.TARGETS:
                y = E.target_truth(at, target, har_vpost)
                base = E.persistence_pred(at, target)
                # references
                gbdt = E._gbdt_predictions(at, target, y, seeds)
                ref1 = np.mean(list(gbdt.values()), axis=0)  # Stage-1, full length
                ref2_v, ref2_t, _ = _predict(
                    "ridge",
                    at,
                    tr,
                    va,
                    te,
                    y,
                    _Cols(emb=list(t_emb), dense=list(t_other) + PASTVOL),
                    seeds,
                )
                ref3_v, ref3_t, _ = _predict(
                    "ridge",
                    at,
                    tr,
                    va,
                    te,
                    y,
                    _Cols(emb=list(wav) + list(eg), dense=list(PASTVOL)),
                    seeds,
                )
                ref = {"val": {"s2": ref2_v, "s3": ref3_v}, "test": {"s2": ref2_t, "s3": ref3_t}}
                fy = tr & np.isfinite(y)
                emb_blocks = [t_emb, wav, e2v, eg]
                btr = [block(b, fy) for b in emb_blocks]
                bva = [block(b, va) for b in emb_blocks]
                bte = [block(b, te) for b in emb_blocks]
                for cov in COVARIATES:
                    dcols = (list(t_other) + PASTVOL) if cov == "fusion_pastvol" else list(t_other)
                    dense = (block(dcols, fy), block(dcols, va), block(dcols, te))
                    # gated fusion (5 seeds)
                    gv, gt = [], []
                    for s in seeds:
                        pv, pt = fusion.gated_fusion_fit_predict(
                            btr, bva, bte, dense, y[fy], seed=s
                        )
                        gv.append(pv)
                        gt.append(pt)
                    gated = {"val": np.mean(gv, axis=0), "test": np.mean(gt, axis=0)}
                    gstd = (
                        float(
                            np.std(
                                [
                                    np.mean(
                                        (y[te][np.isfinite(y[te])] - p[np.isfinite(y[te])]) ** 2
                                    )
                                    for p in gt
                                ]
                            )
                        )
                        if np.isfinite(y[te]).any()
                        else 0.0
                    )
                    # late-fusion stack over [text, audio, stage1] base preds
                    base_va = np.column_stack([ref2_v, ref3_v, ref1[va]])
                    base_te = np.column_stack([ref2_t, ref3_t, ref1[te]])
                    sv, st = fusion.stack_fit_predict(base_va, base_te, y[va])
                    for name, pred, sd in (
                        (f"gated_{cov}", gated, gstd),
                        (f"stack_{cov}", {"val": sv, "test": st}, float("nan")),
                    ):
                        for seg, mask in (("val", va), ("test", te)):
                            if mask.sum() == 0:
                                continue
                            yt, bt = y[mask], base[mask]
                            cell = E._cell_metrics(at[mask], yt, pred[seg], bt)
                            cell["seed_std"] = sd
                            cell["dm_p_vs_persistence"] = _dm(yt, pred[seg], bt)
                            cell["dm_p_vs_stage1"] = _dm(yt, pred[seg], ref1[mask])
                            cell["dm_p_vs_stage2"] = _dm(yt, pred[seg], ref[seg]["s2"])
                            cell["dm_p_vs_stage3"] = _dm(yt, pred[seg], ref[seg]["s3"])
                            rows.append(_row(dataset, scheme, target, tau, name, seg, cell))
    return rows


def run_stage4(root: Path, *, seeds=E.DEFAULT_SEEDS) -> pd.DataFrame:
    """Evaluate Stage-4 fusion heads (FinCall) and write the fusion result rows."""
    rows = (
        evaluate_stage4(root, "fincall", seeds=seeds)
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
    table.to_csv(out / "result_table_4_fusion.csv", index=False, lineterminator="\n")
    return table
