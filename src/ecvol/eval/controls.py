"""Identity-control suite → Result Controls table (DESIGN §7.3, T3.4).

Three controls that test whether Stage-2 'content' signal is really ticker identity:

1. **Ticker-only** — the train per-ticker target mean (no content); compared to the heads.
2. **Same-ticker transcript shuffle** — at test, each call's text features are replaced by a
   *same-ticker* sibling's (within-ticker) and, as a floor, by any call's (global). A head whose
   R²_OOS is unchanged under the within-ticker shuffle is reading identity, not call content.
3. **Identity linear probe** — multinomial logistic regression predicting ticker from frozen
   full-call embeddings; high accuracy ⇒ the embeddings encode identity.

Outputs: `data/results/result_controls.csv` (ticker-only + per-head real/within/global R²_OOS)
and `data/results/controls_probe.csv`. These feed the §4 framing-gate decision (Path A vs B).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ecvol.eval import evaluate as E
from ecvol.eval import metrics as M
from ecvol.eval.stage2 import HEAD_NAMES, VARIANTS, _variant_cols
from ecvol.features.text.assemble import build_text_matrix
from ecvol.models import baselines as B
from ecvol.models import heads
from ecvol.models.ticker_only import ticker_mean_fit_predict

SHUFFLE_CONDITIONS = ("real", "within_shuffle", "global_shuffle")


def _shuffle_text(Xte, te_tickers, text_idx, mode, rng):
    """Permute the text columns among test rows (within-ticker or global); no-op if no text cols."""
    Xs = Xte.copy()
    if not text_idx:
        return Xs
    n = Xte.shape[0]
    if mode == "global_shuffle":
        perm = rng.permutation(n)
    else:  # within_shuffle: permute inside each ticker group (singletons unchanged)
        perm = np.arange(n)
        for t in np.unique(te_tickers):
            idx = np.where(te_tickers == t)[0]
            if len(idx) > 1:
                perm[idx] = idx[rng.permutation(len(idx))]
    Xs[:, text_idx] = Xte[perm][:, text_idx]
    return Xs


def _impute(X, tr):
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    nan = np.isnan(X)
    if nan.any():
        X[nan] = np.take(med, np.where(nan)[1])
    return X


def _head_predict_variants(head, Xfit, yfit, Xval, yval, Xte_list, emb_idx, other_idx, seeds):
    """Train one head; predict each test matrix in Xte_list (fit once, score many)."""
    if head == "ridge":
        scaler, model, _ = heads.ridge_fit(Xfit, yfit, Xval, yval)
        return [heads.predict(scaler, model, Xte) for Xte in Xte_list]
    if emb_idx:
        k = min(heads.PCA_DIM, Xfit[:, emb_idx].shape[0], len(emb_idx))
        pca = PCA(n_components=k, random_state=0).fit(Xfit[:, emb_idx])

        def red(X):
            return np.hstack([pca.transform(X[:, emb_idx]), X[:, other_idx]])
    else:

        def red(X):
            return X

    rfit = red(Xfit)
    preds = [np.zeros(len(Xte)) for Xte in Xte_list]
    for s in seeds:
        scaler, model = heads.mlp_fit(rfit, yfit, seed=s)
        for i, Xte in enumerate(Xte_list):
            preds[i] = preds[i] + heads.predict(scaler, model, red(Xte))
    return [p / len(seeds) for p in preds]


def _controls_dataset(df, text_df, emb_cols, other_cols, dataset, splits_dir, *, seeds):
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
            if tr.sum() == 0 or te.sum() == 0:
                continue
            te_tickers = at.loc[te, "ticker"].to_numpy()
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
                base = E.persistence_pred(at, target)
                yte, bte = y_all[te], base[te]
                # 1) ticker-only
                tmean = ticker_mean_fit_predict(
                    at.loc[tr, "ticker"].to_numpy(), y_all[tr], te_tickers
                )
                rows.append(
                    _crow(dataset, scheme, target, tau, "ticker_only", "real", yte, tmean, bte)
                )
                # 2) shuffle per head x variant
                for variant in VARIANTS:
                    cols, emb, other = _variant_cols(variant, emb_cols, other_cols)
                    X = _impute(at[cols].to_numpy(dtype=np.float64).copy(), tr)
                    fit = tr & np.isfinite(y_all)
                    emb_idx = [cols.index(c) for c in emb]
                    other_idx = [cols.index(c) for c in other]
                    # Shuffle ONLY true text columns — never the past-vol covariates (which
                    # appear in `other` for the *_pastvol variants).
                    text_set = set(emb_cols) | set(other_cols)
                    text_idx = [i for i, c in enumerate(cols) if c in text_set]
                    rng = np.random.default_rng(0)
                    Xte = X[te]
                    te_list = [
                        Xte,
                        _shuffle_text(Xte, te_tickers, text_idx, "within_shuffle", rng),
                        _shuffle_text(Xte, te_tickers, text_idx, "global_shuffle", rng),
                    ]
                    for head in HEAD_NAMES:
                        preds = _head_predict_variants(
                            head,
                            X[fit],
                            y_all[fit],
                            X[val],
                            y_all[val],
                            te_list,
                            emb_idx,
                            other_idx,
                            seeds,
                        )
                        for cond, pred in zip(SHUFFLE_CONDITIONS, preds, strict=True):
                            rows.append(
                                _crow(
                                    dataset,
                                    scheme,
                                    target,
                                    tau,
                                    f"{head}_{variant}",
                                    cond,
                                    yte,
                                    pred,
                                    bte,
                                )
                            )
    return rows


def _crow(dataset, scheme, target, tau, model, condition, y_true, y_pred, base) -> dict:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    return {
        "dataset": dataset,
        "split": scheme,
        "target": target,
        "horizon": int(tau),
        "segment": "test",
        "model": model,
        "condition": condition,
        "n": int(finite.sum()),
        "r2_oos": float(M.r2_oos(y_true, y_pred, base)),
        "mse": float(M.mse(y_true, y_pred)),
    }


def identity_probe(root: Path, dataset: str, *, seed: int = 0) -> dict:
    """Multinomial logistic probe: full-call embedding → ticker; accuracy vs chance."""
    emb = pd.read_parquet(root / dataset / "text_embeddings.parquet")
    emb = emb[emb["scope"] == "full"].copy()
    emb["call_id"] = emb["call_id"].astype(str)
    calls = pd.read_parquet(root / dataset / "calls.parquet", columns=["call_id", "ticker"])
    calls["call_id"] = calls["call_id"].astype(str)
    emb = emb.merge(calls, on="call_id", how="left")
    # keep tickers with >=2 calls (need both train and test representation)
    vc = emb["ticker"].value_counts()
    emb = emb[emb["ticker"].isin(vc[vc >= 2].index)]
    X = np.vstack(emb["vector"].map(np.asarray).to_numpy())
    y = emb["ticker"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(X[tr]), y[tr])
    acc = float(clf.score(scaler.transform(X[te]), y[te]))
    n_classes = int(len(np.unique(y)))
    return {
        "dataset": dataset,
        "n_calls": int(len(y)),
        "n_tickers": n_classes,
        "probe_accuracy": acc,
        "chance": 1.0 / n_classes,
        "accuracy_over_chance": acc * n_classes,
    }


def run_controls(root: Path, *, seeds=E.DEFAULT_SEEDS):
    """Run the §7.3 control suite on every dataset; write the two CSVs."""
    ctrl_rows: list[dict] = []
    probe_rows: list[dict] = []
    for dataset in E.DATASETS:
        if not (root / dataset / "text_embeddings.parquet").is_file():
            continue
        df = E.load_eval_frame(root, dataset)
        text_df, emb_cols, other_cols = build_text_matrix(root, dataset)
        ctrl_rows.extend(
            _controls_dataset(
                df, text_df, emb_cols, other_cols, dataset, root / "splits", seeds=seeds
            )
        )
        probe_rows.append(identity_probe(root, dataset))
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    ctrl = (
        pd.DataFrame(ctrl_rows)
        .sort_values(["dataset", "split", "target", "horizon", "model", "condition"])
        .reset_index(drop=True)
    )
    ctrl.to_csv(out / "result_controls.csv", index=False, lineterminator="\n")
    probe = pd.DataFrame(probe_rows)
    probe.to_csv(out / "controls_probe.csv", index=False, lineterminator="\n")
    return ctrl, probe
