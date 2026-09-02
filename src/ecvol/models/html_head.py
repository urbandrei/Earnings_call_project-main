"""HTML sentence-level transformer head (Yang et al., WWW 2020) for the reproduction study (T6R.2).

A faithful port of the *sentence-level* half of HTML — the token-level WWM-BERT
encoder is replaced by our frozen chunk embeddings (BGE-M3, 1024-d; a documented
substitution, see DECISIONS 2026-09-03 (later)): a call is a sequence of chunk
vectors + a learned position embedding → 2 layers × 2 heads of self-attention with
residual + LayerNorm and an MLP block (dropout 0.5) → average pooling → two linear
regressors (main: τ-day average log volatility; auxiliary: single-day log
volatility at day τ) trained with the weighted loss
``α·MSE_main + (1−α)·MSE_aux``, Adam lr 2e-5, batch 4, α tuned on validation over
{0.1,…,0.9} (paper Table 1 / §5.3). The legacy notebook `legacy/4-Reproduce_HTML.ipynb`
is the prior working reimplementation this follows.

torch is imported lazily (gpu dependency group; CI is torch-free).
"""

from __future__ import annotations

import numpy as np

HEADS = 2
DEPTH = 2
DROPOUT = 0.5
LR = 2e-5
BATCH = 4
EPOCHS = 30
MAX_LEN = 520
ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_model(emb: int, seed: int):
    torch, nn = _torch()
    torch.manual_seed(seed)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(emb, HEADS, dropout=DROPOUT, batch_first=True)
            self.n1 = nn.LayerNorm(emb)
            self.n2 = nn.LayerNorm(emb)
            self.ff = nn.Sequential(
                nn.Linear(emb, 4 * emb), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(4 * emb, emb)
            )

        def forward(self, x, pad):
            a, _ = self.attn(x, x, x, key_padding_mask=pad)
            x = self.n1(x + a)
            return self.n2(x + self.ff(x))

    class HTML(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos = nn.Embedding(MAX_LEN, emb)
            self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
            self.drop = nn.Dropout(DROPOUT)
            self.main = nn.Linear(emb, 1)
            self.aux = nn.Linear(emb, 1)

        def forward(self, x, pad):
            t = x.size(1)
            x = self.drop(x + self.pos(torch.arange(t, device=x.device))[None])
            for b in self.blocks:
                x = b(x, pad)
            keep = (~pad).float().unsqueeze(-1)
            pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            return self.main(pooled).squeeze(-1), self.aux(pooled).squeeze(-1)

    return HTML()


def _batches(seqs: list[np.ndarray], idx: np.ndarray, device):
    torch, _ = _torch()
    for i in range(0, len(idx), BATCH):
        sel = idx[i : i + BATCH]
        lens = [min(len(seqs[j]), MAX_LEN) for j in sel]
        t = max(lens)
        emb = seqs[sel[0]].shape[1]
        x = np.zeros((len(sel), t, emb), dtype=np.float32)
        pad = np.ones((len(sel), t), dtype=bool)
        for k, j in enumerate(sel):
            x[k, : lens[k]] = seqs[j][: lens[k]]
            pad[k, : lens[k]] = False
        yield sel, torch.from_numpy(x).to(device), torch.from_numpy(pad).to(device)


def fit_predict(
    seqs: list[np.ndarray],
    y_main: np.ndarray,
    y_aux: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    alpha: float,
    seed: int = 0,
    epochs: int = EPOCHS,
    device: str | None = None,
) -> np.ndarray:
    """Train on `train_idx` (targets standardised on train), return main-task
    predictions for `eval_idx` in the original scale."""
    torch, _ = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb = seqs[0].shape[1]
    model = build_model(emb, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mu, sd = float(np.mean(y_main[train_idx])), float(np.std(y_main[train_idx]) or 1.0)
    mu2, sd2 = float(np.mean(y_aux[train_idx])), float(np.std(y_aux[train_idx]) or 1.0)
    ym = torch.tensor((y_main - mu) / sd, dtype=torch.float32, device=device)
    ya = torch.tensor((y_aux - mu2) / sd2, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(train_idx)
        for sel, x, pad in _batches(seqs, order, device):
            pm, pa = model(x, pad)
            idx = torch.as_tensor(sel, device=device)
            loss = alpha * ((pm - ym[idx]) ** 2).mean() + (1 - alpha) * ((pa - ya[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    out = np.zeros(len(eval_idx), dtype=np.float64)
    pos = {j: k for k, j in enumerate(eval_idx)}
    with torch.no_grad():
        for sel, x, pad in _batches(seqs, np.asarray(eval_idx), device):
            pm, _ = model(x, pad)
            for j, v in zip(sel, pm.cpu().numpy(), strict=True):
                out[pos[j]] = v * sd + mu
    return out
