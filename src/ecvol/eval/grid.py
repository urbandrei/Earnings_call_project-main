"""Result Table 4 — the consolidated §7.6 ablation grid (T5.2).

Assembles one canonical model per stage/modality (pre-registered, not score-selected) from the
committed per-stage result CSVs into a single modality × split × target × horizon grid, and
Holm-corrects the confirmatory Stage-k-vs-Stage-1 DM tests on Δv (across the 4 horizons, per
(stage, split); DESIGN §7.2/§7.5). Also a per-year R²_OOS breakdown (exploratory, §7.5) for the
canonical models on the FinCall temporal test set. Output: `data/results/result_table_4.csv`
(+ `result_table_4_peryear.csv`); `ecvol report` renders the Markdown/LaTeX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval.significance import holm_correction

# (stage label, source CSV, canonical model name) — fixed by rule, not by validation score.
CANONICAL = [
    ("S0_persistence", "result_table_1.csv", "persistence"),
    ("S0_HAR", "result_table_1.csv", "har"),
    ("S1_GBDT", "result_table_1.csv", "gbdt_tickerFE"),
    ("S2_text", "result_table_2.csv", "ridge_text_pastvol"),
    ("S3_audio", "result_table_3.csv", "ridge_wavlm_egemaps_audio_pastvol"),
    ("S4_fusion", "result_table_4_fusion.csv", "stack_fusion_pastvol"),
]
CONFIRMATORY_STAGES = ("S2_text", "S3_audio", "S4_fusion")  # have a DM-vs-Stage-1 column


def build_grid(root: Path) -> pd.DataFrame:
    """Consolidate the canonical models into one long grid + Holm-corrected Δv DM p-values."""
    results = root / "results"
    frames = []
    for stage, csv, model in CANONICAL:
        path = results / csv
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        sel = df[df["model"] == model].copy()
        if sel.empty:
            continue
        sel["stage"] = stage
        dm = sel["dm_p_vs_stage1"] if "dm_p_vs_stage1" in sel.columns else np.nan
        keep = sel[
            ["stage", "dataset", "split", "target", "horizon", "segment", "n", "r2_oos", "mse"]
        ].copy()
        keep["dm_p_vs_stage1"] = dm
        frames.append(keep)
    grid = pd.concat(frames, ignore_index=True)

    # Holm across the 4 horizons, per (stage, split) — confirmatory family, Δv test only.
    grid["holm_p_vs_stage1"] = np.nan
    mask = (
        (grid["target"] == "dv")
        & (grid["segment"] == "test")
        & grid["stage"].isin(CONFIRMATORY_STAGES)
    )
    for (_stage, _split, _ds), g in grid[mask].groupby(["stage", "split", "dataset"]):
        g = g.dropna(subset=["dm_p_vs_stage1"])
        if g.empty:
            continue
        adj = holm_correction(g["dm_p_vs_stage1"].to_numpy())
        grid.loc[g.index, "holm_p_vs_stage1"] = adj
    grid = grid.sort_values(["dataset", "split", "target", "horizon", "stage", "segment"])
    out = results / "result_table_4.csv"
    grid.to_csv(out, index=False, lineterminator="\n")
    return grid


# --- per-year breakdown (exploratory) ----------------------------------------


def _canonical_test_predictions(root: Path):
    """Recompute the canonical models' FinCall temporal test predictions (level-v, Δv) + as_of.

    Reuses the stage harnesses so the per-year numbers are consistent with Result Tables 1–4.
    """
    from ecvol.eval import evaluate as E
    from ecvol.eval.stage3 import PASTVOL, _Cols, _predict
    from ecvol.features.audio.assemble import load_audio_blocks
    from ecvol.features.text.assemble import build_text_matrix
    from ecvol.models import baselines as B
    from ecvol.models import fusion

    df = E.load_eval_frame(root, "fincall")
    audio_df, ablocks = load_audio_blocks(root)
    text_df, t_emb, t_other = build_text_matrix(root, "fincall")
    feat = audio_df.merge(text_df, on="call_id", how="outer")
    fcols = [c for c in feat.columns if c != "call_id"]
    wav, eg = ablocks["wavlm"], ablocks["egemaps"]
    assign = pd.read_csv(
        root / "splits" / "fincall_temporal.csv", dtype={"call_id": str}
    ).set_index("call_id")["split"]
    tau = 30  # the long-horizon, regime-sensitive view
    at = df[df["horizon"] == tau].copy()
    at["split"] = at["call_id"].map(assign).fillna("excluded")
    at = at.merge(feat, on="call_id", how="left")
    at[fcols] = at[fcols].fillna(0.0)
    tr = (at["split"] == "train").to_numpy()
    va = (at["split"] == "val").to_numpy()
    te = (at["split"] == "test").to_numpy()
    x_all = B.har_design(at["rv_daily"], at["rv_weekly"], at["rv_monthly"])
    coef = B.har_fit(
        B.har_design(at.loc[tr, "rv_daily"], at.loc[tr, "rv_weekly"], at.loc[tr, "rv_monthly"]),
        at.loc[tr, "v_post"].to_numpy(),
    )
    har_vpost = B.har_predict(x_all, coef)
    text_cols = _Cols(emb=list(t_emb), dense=list(t_other) + PASTVOL)
    audio_cols = _Cols(emb=list(wav) + list(eg), dense=list(PASTVOL))
    out = {}
    for target in ("v", "dv"):
        y = E.target_truth(at, target, har_vpost)
        base = E.persistence_pred(at, target)
        gbdt = E._gbdt_predictions(at, target, y, E.DEFAULT_SEEDS)
        s1 = np.mean(list(gbdt.values()), axis=0)  # full length
        s2_va, s2_te, _ = _predict("ridge", at, tr, va, te, y, text_cols, E.DEFAULT_SEEDS)
        s3_va, s3_te, _ = _predict("ridge", at, tr, va, te, y, audio_cols, E.DEFAULT_SEEDS)
        base_va = np.column_stack([s2_va, s3_va, s1[va]])
        base_te = np.column_stack([s2_te, s3_te, s1[te]])
        _, s4 = fusion.stack_fit_predict(base_va, base_te, y[va])
        out[target] = {
            "as_of": at.loc[te, "as_of"].to_numpy(),
            "y": y[te],
            "base": base[te],
            "S1_GBDT": s1[te],
            "S2_text": s2_te,
            "S3_audio": s3_te,
            "S4_fusion": s4,
            "S0_HAR": E.vpost_to_target(har_vpost, at, target, har_vpost)[te],
        }
    return out


def per_year_breakdown(root: Path) -> pd.DataFrame:
    """Per-calendar-year test R²_OOS for the canonical models (FinCall temporal, τ=30)."""
    from ecvol.eval.metrics import r2_oos

    preds = _canonical_test_predictions(root)
    rows = []
    for target, d in preds.items():
        years = pd.Series(d["as_of"]).str[:4].to_numpy()
        for yr in sorted(set(years)):
            m = years == yr
            if m.sum() < 5:
                continue
            for model in ("S0_HAR", "S1_GBDT", "S2_text", "S3_audio", "S4_fusion"):
                rows.append(
                    {
                        "target": target,
                        "year": yr,
                        "n": int(m.sum()),
                        "model": model,
                        "r2_oos": float(r2_oos(d["y"][m], d[model][m], d["base"][m])),
                    }
                )
    grid = pd.DataFrame(rows)
    out = root / "results" / "result_table_4_peryear.csv"
    grid.to_csv(out, index=False, lineterminator="\n")
    return grid


def run_grid(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Result Table 4 + the per-year breakdown."""
    return build_grid(root), per_year_breakdown(root)
