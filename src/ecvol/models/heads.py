"""Stage-2 prediction heads over frozen text features (T3.3).

Two small, regularized heads on standardized features (sklearn, CPU, deterministic):

- **ridge** — L2 linear; alpha chosen on the *validation* split from a fixed grid, then the
  train-only model predicts test (no retrain-on-train+val, avoiding the §3 selection-overfit).
- **mlp** — one hidden layer (256), L2 + internal early-stopping, seeded; 5 seeds → mean±std.

The embedding block is high-dimensional (2048-d for prepared+qa); ridge's L2 absorbs it
directly, the MLP takes a **train-fit PCA** reduction of it (`pca_reduce`) to keep params small
in this low-n regime. All transforms are fit on train only (leakage-safe).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
MLP_HIDDEN = (256,)
MLP_ALPHA = 1e-3
MLP_MAX_ITER = 300
PCA_DIM = 256


def ridge_fit_predict(X_tr, y_tr, X_val, y_val, X_test, *, alphas=RIDGE_ALPHAS):
    """Fit on train, pick alpha by val MSE, predict val+test → (pred_val, pred_test, alpha)."""
    scaler = StandardScaler().fit(X_tr)
    xtr, xval, xte = scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_test)
    vmask = np.isfinite(y_val)
    best = None
    for a in alphas:
        model = Ridge(alpha=a).fit(xtr, y_tr)
        mse = (
            float(np.mean((y_val[vmask] - model.predict(xval)[vmask]) ** 2))
            if vmask.any()
            else np.inf
        )
        if best is None or mse < best[0]:
            best = (mse, a, model)
    _, alpha, model = best
    return model.predict(xval), model.predict(xte), alpha


def mlp_fit_predict(X_tr, y_tr, X_val, X_test, *, seed):
    """Seeded 1-hidden-layer MLP (L2 + internal early-stop). Returns (pred_val, pred_test)."""
    scaler = StandardScaler().fit(X_tr)
    xtr, xval, xte = scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_test)
    model = MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN,
        alpha=MLP_ALPHA,
        max_iter=MLP_MAX_ITER,
        early_stopping=True,
        random_state=seed,
    )
    model.fit(xtr, y_tr)
    return model.predict(xval), model.predict(xte)


def pca_reduce(emb_tr, emb_val, emb_test, *, dim=PCA_DIM):
    """Train-fit PCA on the embedding block (clipped to its rank); transform all splits."""
    k = min(dim, emb_tr.shape[0], emb_tr.shape[1])
    pca = PCA(n_components=k, random_state=0).fit(emb_tr)
    return pca.transform(emb_tr), pca.transform(emb_val), pca.transform(emb_test)
