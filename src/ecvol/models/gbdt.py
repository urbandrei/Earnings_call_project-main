"""Stage-1 LightGBM baseline with a ticker fixed effect (DESIGN §6 Stage 1).

The in-house "Same-Company-Same-Signal" baseline: gradient-boosted trees on
tabular features only — past-vol (`v_pre`, RV_d/w/m) + trivial transcript
metadata (length, turn count) + a **ticker fixed effect** (categorical). It
answers "how much of published 'multimodal' performance is identity + metadata?"
before any call content is read.

Sector and market-cap features (DESIGN §6 Stage 1) are **omitted** — neither
corpus carries that metadata (documented, not silently dropped). Trained per
(split, target, τ) on the train rows; multi-seed averaging is the caller's job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Numeric features available from targets.parquet + calls.parquet.
NUMERIC_FEATURES = ["v_pre", "rv_daily", "rv_weekly", "rv_monthly", "n_chars", "n_turns"]
CATEGORICAL_FEATURES = ["ticker"]  # the fixed effect

# Small-data regime (10³–10⁴ rows): shallow, regularized, fixed (no tuning).
PARAMS = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbosity": -1,
    # Reproducibility: single-threaded + deterministic so Result Table 1
    # regenerates byte-identically (T2.3 acceptance).
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
}


def _prepare(
    train: pd.DataFrame, predict: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared category coding so train/predict ticker codes line up."""
    xtr = train[feature_cols].copy()
    xpr = predict[feature_cols].copy()
    for c in cat_cols:
        cats = pd.Index(sorted(set(train[c]) | set(predict[c])))
        xtr[c] = pd.Categorical(train[c], categories=cats)
        xpr[c] = pd.Categorical(predict[c], categories=cats)
    return xtr, xpr


def train_predict_gbdt(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    target_col: str,
    *,
    seed: int,
    feature_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> np.ndarray:
    """Fit LightGBM on `train` rows and predict `predict` rows. Deterministic per seed.

    Train rows with a NaN target are dropped; NaN features are fine (LightGBM
    handles missing). Returns predictions aligned to `predict`'s row order.
    """
    import lightgbm as lgb

    feature_cols = feature_cols or (NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    cat_cols = cat_cols or CATEGORICAL_FEATURES

    y = train[target_col].to_numpy(dtype=float)
    keep = ~np.isnan(y)
    xtr, xpr = _prepare(train.loc[keep], predict, feature_cols, cat_cols)
    model = lgb.LGBMRegressor(random_state=seed, **PARAMS)
    model.fit(xtr, y[keep], categorical_feature=cat_cols)
    return model.predict(xpr)
