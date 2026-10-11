"""Paired increment of call content over price history → Result Table 8 (DECISIONS 2026-10-11).

For FinCall, each split, target and horizon, the canonical content models of the grid (Stage-2
text ridge, Stage-3 WavLM+eGeMAPS ridge, Stage-4 stack, all with the past-volatility block) are
compared on the same test calls with the **price-only ridge** on that past-volatility block,
with HAR-RV and with persistence. The quantity is the relative MSE reduction
(MSE_ref − MSE_model) / MSE_ref, with a 95% percentile bootstrap interval that resamples
companies, a per-call Diebold–Mariano p-value, and the pre-registered reading: improvement if
the lower bound exceeds the 2% threshold, no meaningful benefit if the upper bound is below it,
inconclusive otherwise. Run on the measured anchor and on the forward-looking `origin` variant.
Output: `data/results/result_table_8.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import evaluate as E
from ecvol.eval.significance import diebold_mariano
from ecvol.eval.stage3 import PASTVOL, _Cols, _predict
from ecvol.models import baselines as B
from ecvol.models import fusion

ANCHORS = ("measured", "origin")
TARGETS = ("dv", "v")
THRESHOLD = 0.02  # practical relative MSE reduction, fixed in DECISIONS 2026-10-11
N_RESAMPLES = 1000
CONTENT = ("S2_text", "S3_audio", "S4_fusion")
REFERENCES = ("persistence", "price_ridge", "HAR")


def rel_reduction_ci(y, pred, ref, clusters, *, n_resamples=N_RESAMPLES, seed=0):
    """(point, lo, hi): relative MSE reduction of `pred` over `ref`, companies resampled."""
    lm = (np.asarray(y) - np.asarray(pred)) ** 2
    lr = (np.asarray(y) - np.asarray(ref)) ** 2
    point = 1.0 - lm.sum() / lr.sum()
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sm, sr = np.bincount(inv, lm), np.bincount(inv, lr)
    draws = np.random.default_rng(seed).integers(0, len(sm), size=(n_resamples, len(sm)))
    ratio = 1.0 - sm[draws].sum(axis=1) / sr[draws].sum(axis=1)
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def verdict(lo: float, hi: float, threshold: float = THRESHOLD) -> str:
    if lo > threshold:
        return "improvement"
    if hi < threshold:
        return "no meaningful benefit"
    return "inconclusive"


def _cell_predictions(at, target, tr, va, te, t_emb, t_other, wav, eg, seeds) -> dict:
    """Test-segment predictions of every model and reference for one (split, τ, target)."""
    x_all = B.har_design(at["rv_daily"], at["rv_weekly"], at["rv_monthly"])
    coef = B.har_fit(
        B.har_design(at.loc[tr, "rv_daily"], at.loc[tr, "rv_weekly"], at.loc[tr, "rv_monthly"]),
        at.loc[tr, "v_post"].to_numpy(),
    )
    har_vpost = B.har_predict(x_all, coef)
    y = E.target_truth(at, target, har_vpost)
    gbdt = E._gbdt_predictions(at, target, y, seeds)
    s1 = np.mean(list(gbdt.values()), axis=0)
    _, price = _predict("ridge", at, tr, va, te, y, _Cols(dense=list(PASTVOL)), seeds)[:2]
    s2_va, s2_te, _ = _predict(
        "ridge", at, tr, va, te, y, _Cols(emb=list(t_emb), dense=list(t_other) + PASTVOL), seeds
    )
    s3_va, s3_te, _ = _predict(
        "ridge", at, tr, va, te, y, _Cols(emb=list(wav) + list(eg), dense=list(PASTVOL)), seeds
    )
    _, s4 = fusion.stack_fit_predict(
        np.column_stack([s2_va, s3_va, s1[va]]), np.column_stack([s2_te, s3_te, s1[te]]), y[va]
    )
    return {
        "y": y[te],
        "persistence": E.persistence_pred(at, target)[te],
        "train_mean": E.train_mean_pred(y, tr)[te],
        "HAR": E.vpost_to_target(har_vpost, at, target, har_vpost)[te],
        "price_ridge": price,
        "S1_GBDT": s1[te],
        "S2_text": s2_te,
        "S3_audio": s3_te,
        "S4_fusion": s4,
    }


def run_increment(root: Path, *, seeds=E.DEFAULT_SEEDS) -> pd.DataFrame:
    from ecvol.features.audio.assemble import load_audio_blocks
    from ecvol.features.text.assemble import build_text_matrix

    audio_df, ablocks = load_audio_blocks(root)
    text_df, t_emb, t_other = build_text_matrix(root, "fincall")
    feat = audio_df.merge(text_df, on="call_id", how="outer")
    fcols = [c for c in feat.columns if c != "call_id"]
    rows: list[dict] = []
    for anchor in ANCHORS:
        df = E.load_eval_frame(root, "fincall", anchor=anchor)
        for scheme in E.SPLIT_SCHEMES:
            assign = pd.read_csv(
                root / "splits" / f"fincall_{scheme}.csv", dtype={"call_id": str}
            ).set_index("call_id")["split"]
            for tau in E.HORIZONS:
                at = df[df["horizon"] == tau].copy()
                at["split"] = at["call_id"].map(assign).fillna("excluded")
                at = at.merge(feat, on="call_id", how="left")
                at[fcols] = at[fcols].fillna(0.0)
                tr, va, te = ((at["split"] == s).to_numpy() for s in ("train", "val", "test"))
                tick = at.loc[te, "ticker"].to_numpy()
                for target in TARGETS:
                    p = _cell_predictions(
                        at, target, tr, va, te, t_emb, t_other, ablocks["wavlm"],
                        ablocks["egemaps"], seeds,
                    )  # fmt: skip
                    y = p["y"]
                    ok = np.isfinite(y)
                    for model in ("train_mean", "HAR", "price_ridge", "S1_GBDT", *CONTENT):
                        for ref in REFERENCES:
                            if ref == model:
                                continue
                            pt, lo, hi = rel_reduction_ci(y[ok], p[model][ok], p[ref][ok], tick[ok])
                            rows.append(
                                {
                                    "anchor": anchor,
                                    "split": scheme,
                                    "target": target,
                                    "horizon": int(tau),
                                    "model": model,
                                    "reference": ref,
                                    "n": int(ok.sum()),
                                    "n_companies": int(len(np.unique(tick[ok]))),
                                    "mse_model": float(np.mean((y[ok] - p[model][ok]) ** 2)),
                                    "mse_ref": float(np.mean((y[ok] - p[ref][ok]) ** 2)),
                                    "rel_reduction": pt,
                                    "ci_lo": lo,
                                    "ci_hi": hi,
                                    "dm_p": float(
                                        diebold_mariano(
                                            y[ok] - p[model][ok], y[ok] - p[ref][ok]
                                        ).p_value
                                    ),
                                    "verdict": verdict(lo, hi),
                                }
                            )
    table = (
        pd.DataFrame(rows)
        .sort_values(["anchor", "split", "target", "horizon", "model", "reference"])
        .reset_index(drop=True)
    )
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_8.csv", index=False, lineterminator="\n")
    return table
