"""Ticker-only (pure-identity) baseline for the §7.3 control suite (T3.4).

The simplest "no content" predictor: the train-set mean of the target per ticker (an unseen
ticker falls back to the global train mean). If a content model matches this on level-v, that
model is reading identity, not call content (DESIGN §3.1). Deterministic.
"""

from __future__ import annotations

import numpy as np


def ticker_mean_fit_predict(train_tickers, train_y, eval_tickers) -> np.ndarray:
    """Per-ticker train-mean target, predicted for each eval ticker (unseen → global mean)."""
    train_y = np.asarray(train_y, dtype=float)
    tickers = np.asarray(train_tickers)
    finite = np.isfinite(train_y)
    global_mean = float(np.mean(train_y[finite])) if finite.any() else 0.0
    means: dict[str, float] = {}
    for t in np.unique(tickers):
        m = finite & (tickers == t)
        if m.any():
            means[t] = float(np.mean(train_y[m]))
    return np.array([means.get(t, global_mean) for t in eval_tickers], dtype=float)
