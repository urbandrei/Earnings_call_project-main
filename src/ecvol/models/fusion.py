"""Stage-4 multimodal fusion heads (T5.1, sklearn/CPU, small-data discipline).

Two fusion strategies over the frozen modality embeddings (text BGE-M3, WavLM, emotion2vec+,
eGeMAPS) + past-vol covariates, each a single pooled vector per call:

- **gated fusion** — each modality block is PCA-reduced (train-fit) to a fixed width and
  L2-normalized so no modality dominates by raw scale (the "gate"), then concatenated with the
  dense covariates and fed to a shallow MLP. <5M params, <2 GB, deterministic per seed.
- **late-fusion stacking** — a meta-ridge over the base learners' predictions (the unimodal text
  and audio heads + the Stage-1 ticker-FE GBDT); the meta-learner is fit on the *validation*
  predictions (bases fit on train) and applied to test, avoiding train-prediction leakage.

Genuine cross-attention is omitted: with one pooled vector per modality it degenerates to a 2–3
token interaction ≈ gated fusion (DECISIONS 2026-06-24).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ecvol.models import heads

MODALITY_PCA_DIM = 64  # per-modality reduction width (keeps the gated MLP small)


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def gated_fuse_blocks(blocks_tr, blocks_va, blocks_te, *, dim=MODALITY_PCA_DIM):
    """Per-modality train-fit PCA + L2-norm, then concat. blocks_* are lists of (tr,va,te) arrays.

    Returns (Xtr, Xva, Xte) ready for the MLP. The per-modality L2-norm is the scale "gate" so a
    high-variance modality can't dominate the concatenation by magnitude alone.
    """
    cat_tr, cat_va, cat_te = [], [], []
    for tr, va, te in zip(blocks_tr, blocks_va, blocks_te, strict=True):
        if tr.shape[1] == 0:
            continue
        k = min(dim, tr.shape[0], tr.shape[1])
        pca = PCA(n_components=k, random_state=0).fit(tr)
        cat_tr.append(_l2norm(pca.transform(tr)))
        cat_va.append(_l2norm(pca.transform(va)))
        cat_te.append(_l2norm(pca.transform(te)))
    return np.hstack(cat_tr), np.hstack(cat_va), np.hstack(cat_te)


def gated_fusion_fit_predict(blocks_tr, blocks_va, blocks_te, dense, y_tr, *, seed):
    """Gated-fusion MLP over PCA'd+normed blocks + dense; returns (val, test)."""
    Xtr, Xva, Xte = gated_fuse_blocks(blocks_tr, blocks_va, blocks_te)
    dtr, dva, dte = dense
    if dtr.shape[1]:
        Xtr = np.hstack([Xtr, dtr])
        Xva = np.hstack([Xva, dva])
        Xte = np.hstack([Xte, dte])
    scaler = StandardScaler().fit(Xtr)
    model = MLPRegressor(
        hidden_layer_sizes=heads.MLP_HIDDEN,
        alpha=heads.MLP_ALPHA,
        max_iter=heads.MLP_MAX_ITER,
        early_stopping=True,
        random_state=seed,
    ).fit(scaler.transform(Xtr), y_tr)
    return model.predict(scaler.transform(Xva)), model.predict(scaler.transform(Xte))


def stack_fit_predict(base_val, base_test, y_val, *, alphas=heads.RIDGE_ALPHAS):
    """Late-fusion meta-ridge over base predictions. base_* : (n, n_base). Fit on val, predict test.

    Returns (pred_val, pred_test). Meta-learner fit on validation base-predictions (bases trained on
    train) → no train-prediction leakage; alpha chosen on the same val (standard stacking).
    """
    vmask = np.isfinite(y_val)
    scaler = StandardScaler().fit(base_val[vmask])
    best = None
    xva = scaler.transform(base_val)
    for a in alphas:
        m = Ridge(alpha=a).fit(xva[vmask], y_val[vmask])
        mse = float(np.mean((y_val[vmask] - m.predict(xva[vmask])) ** 2))
        if best is None or mse < best[0]:
            best = (mse, m)
    model = best[1]
    return model.predict(xva), model.predict(scaler.transform(base_test))
