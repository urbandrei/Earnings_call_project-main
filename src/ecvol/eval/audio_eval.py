"""Audio identity controls + §3.5 gender-confound analysis (T4.4, FinCall-only).

Two diagnostics over the Stage-3 audio features, mirroring the text controls (T3.4):

1. **Identity probe** — a linear logistic probe predicting ticker from the frozen WavLM (and
   emotion2vec+) per-call embedding; accuracy vs chance, directly comparable to the text probe's
   89.5%. Tests whether audio embeddings encode company identity.
2. **§3.5 gender-confound analysis** — an eGeMAPS-F0 pitch proxy splits calls into low/high
   dominant-pitch groups (proxy for speaker gender); we report per-group test MSE of a headline
   WavLM head and the F0↔error / F0↔prediction correlations. Analyze-and-report only (no
   debiasing, v1). Covers ~100% of calls (eGeMAPS F0 is finite for every decoded call).

Outputs: `data/results/audio_probe.csv`, `audio_gender.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ecvol.eval import evaluate as E
from ecvol.features.audio.assemble import load_audio_blocks
from ecvol.models import heads

F0_COL = "F0semitoneFrom27.5Hz_sma3nz_amean"  # eGeMAPS mean F0 (semitones) — the gender proxy
# ~165 Hz ≈ 31 semitones above 27.5 Hz; a common male/female F0 split. Reported as a proxy only.
F0_SPLIT_SEMITONES = 31.0


def _embed_probe(vectors: np.ndarray, tickers: np.ndarray, *, seed: int = 0) -> dict:
    """Logistic probe embedding→ticker (tickers with ≥2 calls); accuracy vs chance."""
    vc = pd.Series(tickers).value_counts()
    keep = np.isin(tickers, vc[vc >= 2].index)
    X, y = vectors[keep], tickers[keep]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[tr]), y[tr])
    acc = float(clf.score(scaler.transform(X[te]), y[te]))
    nc = int(len(np.unique(y)))
    return {"n_calls": int(len(y)), "n_tickers": nc, "probe_accuracy": acc, "chance": 1.0 / nc}


def identity_probe(root: Path) -> pd.DataFrame:
    """WavLM + emotion2vec+ identity probes (FinCall)."""
    audio_df, blocks = load_audio_blocks(root)
    calls = pd.read_parquet(root / "fincall" / "calls.parquet", columns=["call_id", "ticker"])
    calls["call_id"] = calls["call_id"].astype(str)
    audio_df = audio_df.merge(calls, on="call_id", how="left").dropna(subset=["ticker"])
    rows = []
    for name in ("wavlm", "emotion2vec"):
        cols = blocks[name]
        sub = audio_df.dropna(subset=cols)
        V = sub[cols].to_numpy(np.float64)
        res = _embed_probe(V, sub["ticker"].to_numpy(), seed=0)
        res = {
            "embedding": name,
            **res,
            "accuracy_over_chance": res["probe_accuracy"] / res["chance"],
        }
        rows.append(res)
    out = root / "results" / "audio_probe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    return pd.DataFrame(rows)


def _headline_audio_preds(root: Path, dataset="fincall", *, seed=0):
    """WavLM-ridge (audio-only), temporal split, level-v test: (test_df, y_true, y_pred, F0)."""
    df = E.load_eval_frame(root, dataset)
    audio_df, blocks = load_audio_blocks(root)
    assign = pd.read_csv(
        root / "splits" / f"{dataset}_temporal.csv", dtype={"call_id": str}
    ).set_index("call_id")["split"]
    at = df[df["horizon"] == 30].copy()
    at["split"] = at["call_id"].map(assign).fillna("excluded")
    at = at.merge(audio_df, on="call_id", how="left")
    cols = blocks["wavlm"] + [F0_COL]
    at[cols] = at[cols].fillna(0.0)
    tr = (at["split"] == "train").to_numpy()
    val = (at["split"] == "val").to_numpy()
    te = (at["split"] == "test").to_numpy()
    wav = blocks["wavlm"]
    X = at[wav].to_numpy(np.float64)
    y = at["v_post"].to_numpy()
    _, pte, _ = heads.ridge_fit_predict(X[tr], y[tr], X[val], y[val], X[te])
    sub = at[te]
    return sub, y[te], pte, sub[F0_COL].to_numpy(np.float64)


def gender_analysis(root: Path, dataset="fincall") -> pd.DataFrame:
    """§3.5: eGeMAPS-F0 pitch proxy → per-group test MSE + F0↔error/prediction correlations."""
    sub, y_true, y_pred, f0 = _headline_audio_preds(root, dataset)
    err2 = (y_true - y_pred) ** 2
    finite = np.isfinite(f0)
    coverage = float(finite.mean())
    low = finite & (f0 < F0_SPLIT_SEMITONES)  # lower pitch (male-proxy)
    high = finite & (f0 >= F0_SPLIT_SEMITONES)  # higher pitch (female-proxy)
    r_err = float(pearsonr(f0[finite], err2[finite])[0]) if finite.sum() > 2 else float("nan")
    r_pred = float(pearsonr(f0[finite], y_pred[finite])[0]) if finite.sum() > 2 else float("nan")
    rows = [
        ("dataset", dataset),
        ("model", "wavlm_ridge_audio (temporal, level-v, tau=30, test)"),
        ("f0_proxy_coverage", round(coverage, 4)),
        ("f0_split_semitones", F0_SPLIT_SEMITONES),
        ("n_low_pitch", int(low.sum())),
        ("n_high_pitch", int(high.sum())),
        ("mse_low_pitch", float(np.mean(err2[low])) if low.any() else float("nan")),
        ("mse_high_pitch", float(np.mean(err2[high])) if high.any() else float("nan")),
        ("corr_f0_sq_error", round(r_err, 4)),
        ("corr_f0_prediction", round(r_pred, 4)),
    ]
    out = root / "results" / "audio_gender.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(out, index=False, lineterminator="\n")
    return pd.DataFrame(rows, columns=["metric", "value"])


def audio_shuffle_control(root: Path, dataset="fincall") -> pd.DataFrame:
    """Same-ticker AUDIO shuffle on the headline WavLM+past-vol Δv head (the framing-gate test).

    Replaces each test call's WavLM features with a same-ticker sibling's (within), and any call's
    (global), leaving past-vol covariates intact — so a Δv gain that survives the within-shuffle is
    audio *content*, while one that vanishes is ticker identity. Run on both splits, all horizons.
    """
    from ecvol.eval import evaluate as E
    from ecvol.eval.controls import _impute, _shuffle_text
    from ecvol.eval.metrics import r2_oos
    from ecvol.eval.stage3 import PASTVOL
    from ecvol.models import baselines as B
    from ecvol.models.heads import predict as _pred
    from ecvol.models.heads import ridge_fit

    df = E.load_eval_frame(root, dataset)
    audio_df, blocks = load_audio_blocks(root)
    wav = blocks["wavlm"]
    feat_cols = wav + PASTVOL
    rows = []
    for scheme in ("temporal", "ticker_disjoint"):
        assign = pd.read_csv(
            root / "splits" / f"{dataset}_{scheme}.csv", dtype={"call_id": str}
        ).set_index("call_id")["split"]
        for tau in E.HORIZONS:
            at = df[df["horizon"] == tau].copy()
            at["split"] = at["call_id"].map(assign).fillna("excluded")
            at = at.merge(audio_df[["call_id", *wav]], on="call_id", how="left")
            at[wav] = at[wav].fillna(0.0)
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
            y_all = E.target_truth(at, "dv", B.har_predict(x_all, coef))
            base = E.persistence_pred(at, "dv")
            X = _impute(at[feat_cols].to_numpy(np.float64).copy(), tr)
            wav_idx = list(range(len(wav)))  # shuffle only the WavLM columns, not past-vol
            te_tickers = at.loc[te, "ticker"].to_numpy()
            rng = np.random.default_rng(0)
            Xte = X[te]
            conds = {
                "real": Xte,
                "within_shuffle": _shuffle_text(Xte, te_tickers, wav_idx, "within_shuffle", rng),
                "global_shuffle": _shuffle_text(Xte, te_tickers, wav_idx, "global_shuffle", rng),
            }
            fit = tr & np.isfinite(y_all)
            scaler, model, _ = ridge_fit(X[fit], y_all[fit], X[val], y_all[val])
            for cond, Xc in conds.items():
                yp = _pred(scaler, model, Xc)
                rows.append(
                    {
                        "split": scheme,
                        "horizon": int(tau),
                        "condition": cond,
                        "r2_oos": float(r2_oos(y_all[te], yp, base[te])),
                    }
                )
    out = root / "results" / "audio_shuffle.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    return pd.DataFrame(rows)


def run_audio_eval(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Identity probe + §3.5 gender + audio shuffle (FinCall). Returns (probe, gender, shuffle)."""
    probe = identity_probe(root)
    gender = gender_analysis(root)
    shuffle = audio_shuffle_control(root)
    return probe, gender, shuffle
