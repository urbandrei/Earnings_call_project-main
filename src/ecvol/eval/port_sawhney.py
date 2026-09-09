"""T6R.3 — Sawhney et al. (ACM MM 2020), *harness port* (DECISIONS 2026-09-09).

The released code cannot run as shipped: it needs an unshipped call inventory
(`stock_data.csv`), unshipped per-ticker price files, two unsourced lexicons, and a
TensorFlow 2.1 / Keras 2.3.1 / tensorflow-addons 0.8.3 stack that does not install on this
machine. The model is ~0.3–0.65 M parameters, so the port is the smaller job. What is ported
here follows `feature_extraction/Financial/generate_price_vol_data.py`,
`feature_extraction/Text/text_feature_extraction.py`, `models/bilstm_reg.py`,
`models/SVR.py` and `models/ensemble_reg.py` line by line; every deviation is labelled:

* **Labels (theirs, recomputed from our EC price store):** returns on a descending-date
  frame, `future_τ = ln sqrt(Σ(r−r̄)²/τ)` over the τ sessions after the call and
  `past_τ` over the τ sessions before it (mean-subtracted — unlike Qin & Yang and unlike
  our targets). Split = their crude within-2017 sort key `month*30 + day`, then
  60/20/20 (train/val/test) by position.
* **Text branch:** sentence vector = mean of 300-d word vectors of the in-vocabulary
  tokens (their `union_vocab.csv`, 6,875 words = top-5000 corpus words ∪ pragmatic
  lexicon words, shipped); calls padded to 516 sentences. *Deviation W1:* plain GloVe
  6B-300d instead of the Mittens retrofit (W2, optional); tokenisation is a regex +
  scikit-learn stop-word list instead of NLTK's, and no contraction expansion.
  Model: Masking → BiLSTM(100) → Dropout(d) → Linear(1); Adam 1e-3, batch 32, **2 epochs**,
  best-val checkpoint; d = 0.4/0.2/0.2/0.3 for τ = 3/7/15/30. *Deviation:* Keras'
  recurrent dropout is replaced by input dropout of the same rate.
* **Financial branch:** for each of the 30 sessions before the call, the τ-session
  rolling volatility ending that session (their `Past_Volatility_τ`), NaNs → row mean;
  `GridSearchCV(SVR(rbf), C∈{1e-3..10}, γ∈{1e-3..1})` (5-fold, default scoring).
* **Ensemble:** `α·text + (1−α)·finance` on a 51-point grid. The authors pick α on the
  **test** set; we report that (`tuned_on=test`, to match their number) and the honest
  variant tuned on validation (`tuned_on=val`). The audio branch (W3) needs the 26-d
  Praat/prosodic features and is added when they exist.
* **Seeds:** the authors set none; we run seeds 0–4 and report mean ± std.

Published (second-hand, KeFVP Table 2 "Ensemble(Text+Audio)"): 0.601 / 0.308 / 0.181 / 0.119.
Output: `results/result_table_6r_sawhney.csv`.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import metrics as M

TAUS = (3, 7, 15, 30)
DROPOUT = {3: 0.4, 7: 0.2, 15: 0.2, 30: 0.3}
MAX_SENTENCES = 516
UNITS = 100
BATCH = 32
EPOCHS = 2
LR = 1e-3
PUBLISHED = {3: 0.601, 7: 0.308, 15: 0.181, 30: 0.119}
REPO_REL = "raw/ref/repos/Sawhney2020"
GLOVE_ZIP = "raw/ref/glove/glove.6B.zip"
GLOVE_MEMBER = "glove.6B.300d.txt"
TOKEN_RE = re.compile(r"[a-z][a-z'-]+")


# --- labels (their convention) --------------------------------------------------


def their_volatility(returns: np.ndarray) -> float:
    """ln sqrt( Σ (r − r̄)² / n ) — `volatality_past/fut` in generate_price_vol_data.py."""
    r = np.asarray(returns, dtype=float)
    return float(np.log(np.sqrt(np.sum((r - r.mean()) ** 2) / len(r))))


def their_labels(closes: dict[str, float], call_date: str) -> dict[str, float] | None:
    """`calculate_vol` on the descending frame `[index−30, index+31]` around the call date.

    Returns on a descending frame are r_i = (p_i − p_{i+1}) / p_{i+1}, i.e. the return *on*
    day i. future_τ uses positions 30−τ … 29 (days +1 … +τ), past_τ positions 31 … 30+τ
    (days −1 … −τ). None if the window is not fully available.
    """
    dates = sorted(closes)
    if call_date not in closes:
        return None
    idx = dates.index(call_date)
    if idx < 31 or idx + 30 >= len(dates):
        return None
    window = dates[idx - 31 : idx + 31][::-1]  # descending: position 30 = call date
    p = np.array([closes[d] for d in window], dtype=float)
    r = (p[:-1] - p[1:]) / p[1:]
    out = {}
    for tau in TAUS:
        out[f"future_{tau}"] = their_volatility(r[30 - tau : 30])
        out[f"past_{tau}"] = their_volatility(r[31 : 31 + tau])
    return out


def their_split(df: pd.DataFrame) -> pd.Series:
    """`total_days = month*30 + day`, stable sort, 60/20/20 by position."""
    key = df["call_date"].str[5:7].astype(int) * 30 + df["call_date"].str[8:10].astype(int)
    order = np.argsort(key.to_numpy(), kind="stable")
    n = len(df)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    split = np.empty(n, dtype=object)
    split[order[:n_tr]] = "train"
    split[order[n_tr : n_tr + n_va]] = "val"
    split[order[n_tr + n_va :]] = "test"
    return pd.Series(split, index=df.index)


# --- text features --------------------------------------------------------------


def union_vocab(root: Path) -> list[str]:
    df = pd.read_csv(root / REPO_REL / "feature_extraction" / "Text" / "union_vocab.csv")
    return [str(w).lower() for w in df.iloc[:, 1].tolist()]


def glove_vectors(root: Path, vocab: set[str]) -> dict[str, np.ndarray]:
    """300-d GloVe 6B vectors for `vocab`, read straight from the zip."""
    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(root / GLOVE_ZIP) as z, z.open(GLOVE_MEMBER) as f:
        for raw in f:
            line = raw.decode("utf-8")
            w, _, rest = line.partition(" ")
            if w in vocab:
                out[w] = np.asarray(rest.split(), dtype=np.float32)
    return out


def tokenize(text: str, stop: set[str]) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in stop and len(t) > 2]


def sentence_matrix(sentences: list[str], vec: dict[str, np.ndarray], stop: set[str]) -> np.ndarray:
    rows = []
    for s in sentences[:MAX_SENTENCES]:
        toks = [vec[t] for t in tokenize(s, stop) if t in vec]
        rows.append(np.mean(toks, axis=0) if toks else np.zeros(300, np.float32))
    return np.vstack(rows).astype(np.float32) if rows else np.zeros((1, 300), np.float32)


# --- financial features ---------------------------------------------------------


def past_vol_vector(closes: dict[str, float], call_date: str, tau: int) -> np.ndarray | None:
    """`GenerateRegresFeatures`: for the 30 sessions before the call, the τ-session
    rolling mean-subtracted log volatility ending at that session (their `Past_Volatility_τ`)."""
    dates = sorted(closes)
    if call_date not in closes:
        return None
    idx = dates.index(call_date)
    if idx < 30 + tau + 2:
        return None
    p = np.array([closes[d] for d in dates[: idx + 1]], dtype=float)
    ret = p[1:] / p[:-1] - 1.0  # return on each session (ascending)
    vals = []
    for k in range(1, 31):  # sessions −1 … −30
        end = len(ret) - k  # ret[end - 1] is the return on session −k
        window = ret[end - 1 - tau : end - 1]  # sessions −k−1 … −k−τ (their iloc[idx+1:idx+τ+1])
        vals.append(their_volatility(window) if len(window) == tau else np.nan)
    x = np.array(vals, dtype=float)
    x[~np.isfinite(x)] = np.nan  # a zero-variance window gives ln 0 = −inf; treat as their NaN
    if np.isnan(x).any():
        x = np.where(np.isnan(x), np.nanmean(x), x)
    return x


# --- models ---------------------------------------------------------------------


def fit_text_bilstm(
    S_tr, y_tr, S_va, y_va, S_te, *, dropout: float, seed: int, epochs: int = EPOCHS
):
    """Masking → BiLSTM(100) → Dropout → Linear; best-val epoch; returns test predictions."""
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_do = nn.Dropout(dropout)  # stands in for Keras recurrent/input dropout
            self.lstm = nn.LSTM(300, UNITS, batch_first=True, bidirectional=True)
            self.do = nn.Dropout(dropout)
            self.out = nn.Linear(2 * UNITS, 1)

        def forward(self, x, lengths):
            packed = pack_padded_sequence(
                self.in_do(x), lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h, _) = self.lstm(packed)
            return self.out(self.do(torch.cat([h[0], h[1]], dim=-1))).squeeze(-1)

    def batches(S, idx):
        for i in range(0, len(idx), BATCH):
            b = idx[i : i + BATCH]
            seqs = [torch.from_numpy(S[k]) for k in b]
            yield (
                pad_sequence(seqs, batch_first=True).to(dev),
                torch.tensor([len(s) for s in seqs], device=dev),
                b,
            )

    def predict(S):
        net.eval()
        out = np.zeros(len(S), dtype=float)
        with torch.no_grad():
            for x, ln, b in batches(S, np.arange(len(S))):
                out[b] = net(x, ln).cpu().numpy()
        return out

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    best, best_state = np.inf, None
    for _ in range(epochs):
        net.train()
        order = np.random.permutation(len(S_tr))  # Keras fit shuffles by default
        for x, ln, b in batches(S_tr, order):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(x, ln), y_tr_t[b])
            loss.backward()
            opt.step()
        val = float(M.mse(y_va, predict(S_va)))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return predict(S_te), predict(S_va)


def fit_svr(X_tr, y_tr, X_te, X_va):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR

    grid = GridSearchCV(
        SVR(kernel="rbf"), {"C": [0.001, 0.01, 0.1, 1, 10], "gamma": [0.001, 0.01, 0.1, 1]}
    )
    grid.fit(X_tr, y_tr)
    return grid.predict(X_te), grid.predict(X_va), grid.best_params_


RATIO_GRID = np.linspace(0, 1, 51)  # their `ratio_range`


def ensemble(p_text, p_fin, y, grid=None):
    """Their `combined()` restricted to two branches: argmin over α of MSE on `y`."""
    best = (np.inf, None)
    for a in RATIO_GRID if grid is None else grid:
        mse = float(M.mse(y, a * p_text + (1 - a) * p_fin))
        if mse < best[0]:
            best = (mse, float(a))
    return best[1]


AUDIO_DROPOUT = {3: 0.6, 7: 0.4, 15: 0.45, 30: 0.4}  # aligned_audio_reg.py
AUDIO_EPOCHS = 1  # aligned_audio_reg.py trains one epoch, `last_epoch` checkpoint
ATT_HIDDEN, ATT_HEADS = 100, 5


def fit_aligned_audio(
    S_tr, A_tr, y_tr, S_va, A_va, S_te, A_te, *, dropout, seed, epochs=AUDIO_EPOCHS
):
    """`AlignClassModel` (aligned_audio_reg_model.py): BiLSTM(100, seq) over text, BiLSTM(100,
    seq) over audio, multi-head attention with Q = text, K = V = audio (5 heads, hidden 100,
    no mask), BiLSTM(100) over the attended sequence, Linear(1). Adam 1e-3, batch 32,
    1 epoch, last-epoch weights (their `last_epoch` flag). Returns (test, val) predictions."""
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d_audio = A_tr[0].shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_enc = nn.LSTM(300, UNITS, batch_first=True, bidirectional=True)
            self.speech_enc = nn.LSTM(d_audio, UNITS, batch_first=True, bidirectional=True)
            self.wq = nn.Linear(2 * UNITS, ATT_HIDDEN)
            self.wk = nn.Linear(2 * UNITS, ATT_HIDDEN)
            self.wv = nn.Linear(2 * UNITS, ATT_HIDDEN)
            self.wo = nn.Linear(ATT_HIDDEN, ATT_HIDDEN)
            self.align_enc = nn.LSTM(ATT_HIDDEN, UNITS, batch_first=True, bidirectional=True)
            self.do = nn.Dropout(dropout)
            self.out = nn.Linear(2 * UNITS, 1)

        def _seq(self, lstm, x, lengths):
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = lstm(packed)
            return pad_packed_sequence(out, batch_first=True)[0]

        def forward(self, xt, lt, xa, la):
            t = self.do(self._seq(self.text_enc, xt, lt))  # [b, T, 200]
            a = self.do(self._seq(self.speech_enc, xa, la))  # [b, A, 200]
            b, T, _ = t.shape
            dh = ATT_HIDDEN // ATT_HEADS
            q = self.wq(t).view(b, T, ATT_HEADS, dh).transpose(1, 2)
            k = self.wk(a).view(b, a.shape[1], ATT_HEADS, dh).transpose(1, 2)
            v = self.wv(a).view(b, a.shape[1], ATT_HEADS, dh).transpose(1, 2)
            att = torch.softmax(q @ k.transpose(-1, -2) / dh**0.5, dim=-1) @ v
            att = self.do(self.wo(att.transpose(1, 2).reshape(b, T, ATT_HIDDEN)))
            packed = pack_padded_sequence(att, lt.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.align_enc(packed)
            return self.out(self.do(torch.cat([h[0], h[1]], -1))).squeeze(-1)

    def batches(S, A, idx):
        for i in range(0, len(idx), BATCH):
            bb = idx[i : i + BATCH]
            st = [torch.from_numpy(S[k]) for k in bb]
            sa = [torch.from_numpy(A[k]) for k in bb]
            yield (
                pad_sequence(st, batch_first=True).to(dev),
                torch.tensor([len(x) for x in st], device=dev),
                pad_sequence(sa, batch_first=True).to(dev),
                torch.tensor([len(x) for x in sa], device=dev),
                bb,
            )

    def predict(S, A):
        net.eval()
        out = np.zeros(len(S))
        with torch.no_grad():
            for xt, lt, xa, la, bb in batches(S, A, np.arange(len(S))):
                out[bb] = net(xt, lt, xa, la).cpu().numpy()
        return out

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        net.train()
        for xt, lt, xa, la, bb in batches(S_tr, A_tr, np.random.permutation(len(S_tr))):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xt, lt, xa, la), y_t[bb])
            loss.backward()
            opt.step()
    return predict(S_te, A_te), predict(S_va, A_va)


def ensemble3(p_text, p_audio, p_fin, y):
    """Their `combined()`: α·text + β·audio + (1−α−β)·finance, 51-point grid, β ≤ 1−α."""
    best = (np.inf, None, None)
    for a in RATIO_GRID:
        for bt in RATIO_GRID:
            if bt <= 1 - a + 1e-12:
                mse = float(M.mse(y, a * p_text + bt * p_audio + (1 - a - bt) * p_fin))
                if mse < best[0]:
                    best = (mse, float(a), float(bt))
    return best[1], best[2]


def audio_matrices(root: Path, call_ids, n_sentences: dict[str, int]) -> list[np.ndarray] | None:
    """27 Praat features per sentence (H3/W3 cache), z-scored, NaN → 0; None if the cache
    does not exist. Their audio is 26-d (18 Praat + 8 prosodic, unshipped): substitution."""
    from ecvol.features.audio.praat import CACHE_FILE, sentence_audio

    if not (root / "ec" / "cache" / CACHE_FILE).is_file():
        return None
    aud = sentence_audio(root, n_sentences)
    allv = np.vstack(list(aud.values()))
    mu, sd = np.nanmean(allv, 0), np.nanstd(allv, 0)
    sd = np.where(sd > 0, sd, 1.0)
    out = []
    for c in call_ids:
        a = aud.get(c)
        if a is None:
            a = np.zeros((max(n_sentences.get(c, 1), 1), allv.shape[1]), np.float32)
        out.append(np.nan_to_num((a - mu) / sd).astype(np.float32)[:MAX_SENTENCES])
    return out


# --- driver ---------------------------------------------------------------------


def build_frame(root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    from ecvol.data.prices import load_close_series

    calls = pd.read_parquet(
        root / "ec" / "calls.parquet", columns=["call_id", "ticker", "call_date", "status"]
    )
    calls = calls[(calls["status"] == "ok") & (calls["ticker"] != "")].reset_index(drop=True)
    closes: dict[str, dict[str, float]] = {}
    rows = []
    for r in calls.itertuples(index=False):
        if r.ticker not in closes:
            closes[r.ticker] = load_close_series(root / "prices", r.ticker)
        lab = their_labels(closes[r.ticker], r.call_date)
        if lab is None or not all(np.isfinite(v) for v in lab.values()):
            continue  # window unavailable, or a zero-variance window (ln 0)
        rows.append({"call_id": r.call_id, "ticker": r.ticker, "call_date": r.call_date, **lab})
    df = pd.DataFrame(rows)
    df["split"] = their_split(df)
    return df, closes


def run_sawhney_port(
    root: Path, *, seeds=(0, 1, 2, 3, 4), epochs: int = EPOCHS, log=print
) -> pd.DataFrame:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    from ecvol.eval.faithful_html import ec_sentences

    df, closes = build_frame(root)
    log(f"  {len(df)} EC calls with their labels; split {df['split'].value_counts().to_dict()}")
    vocab = set(union_vocab(root))
    vec = glove_vectors(root, vocab)
    log(f"  vocab {len(vocab)} words, {len(vec)} with GloVe-6B vectors")
    stop = set(ENGLISH_STOP_WORDS)
    sents = ec_sentences(root)
    S = [sentence_matrix(sents.get(c, []), vec, stop) for c in df["call_id"]]
    A = audio_matrices(
        root, df["call_id"].tolist(), {c: len(sents.get(c, [])) for c in df["call_id"]}
    )
    log(f"  audio branch: {'27-d Praat per sentence' if A is not None else 'absent (no cache)'}")
    idx = {s: np.where(df["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}
    tr, va, te = idx["train"], idx["val"], idx["test"]
    rows = []
    for tau in TAUS:
        y = df[f"future_{tau}"].to_numpy(dtype=float)
        pers = float(M.mse(y[te], df.loc[te, f"past_{tau}"].to_numpy(dtype=float)))
        X = np.vstack(
            [
                past_vol_vector(closes[t], d, tau)
                for t, d in zip(df["ticker"], df["call_date"], strict=True)
            ]
        )
        p_fin_te, p_fin_va, params = fit_svr(X[tr], y[tr], X[te], X[va])
        rows.append(
            dict(
                branch="finance_svr",
                horizon=tau,
                seed=-1,
                mse=float(M.mse(y[te], p_fin_te)),
                tuned_on="cv",
                persistence_mse=pers,
                note=str(params),
            )
        )
        for seed in seeds:
            p_te, p_va = fit_text_bilstm(
                [S[i] for i in tr],
                y[tr],
                [S[i] for i in va],
                y[va],
                [S[i] for i in te],
                dropout=DROPOUT[tau],
                seed=seed,
                epochs=epochs,
            )
            rows.append(
                dict(
                    branch="text_bilstm",
                    horizon=tau,
                    seed=seed,
                    mse=float(M.mse(y[te], p_te)),
                    tuned_on="val",
                    persistence_mse=pers,
                    note="",
                )
            )
            if A is not None:
                p_a_te, p_a_va = fit_aligned_audio(
                    [S[i] for i in tr], [A[i] for i in tr], y[tr],
                    [S[i] for i in va], [A[i] for i in va],
                    [S[i] for i in te], [A[i] for i in te],
                    dropout=AUDIO_DROPOUT[tau], seed=seed,
                )  # fmt: skip
                rows.append(
                    dict(
                        branch="aligned_audio",
                        horizon=tau,
                        seed=seed,
                        mse=float(M.mse(y[te], p_a_te)),
                        tuned_on="last_epoch",
                        persistence_mse=pers,
                        note="",
                    )  # noqa: E501
                )
                for tuned, (aa, bb) in (
                    ("test", ensemble3(p_te, p_a_te, p_fin_te, y[te])),
                    ("val", ensemble3(p_va, p_a_va, p_fin_va, y[va])),
                ):
                    comb = aa * p_te + bb * p_a_te + (1 - aa - bb) * p_fin_te
                    rows.append(
                        dict(
                            branch="ensemble_text_audio_finance",
                            horizon=tau,
                            seed=seed,
                            mse=float(M.mse(y[te], comb)),
                            tuned_on=tuned,
                            persistence_mse=pers,
                            note=f"alpha={aa:.2f} beta={bb:.2f}",
                        )  # noqa: E501
                    )
            a_test = ensemble(p_te, p_fin_te, y[te])
            a_val = ensemble(p_va, p_fin_va, y[va])
            for tuned, a in (("test", a_test), ("val", a_val)):
                rows.append(
                    dict(
                        branch="ensemble_text_finance",
                        horizon=tau,
                        seed=seed,
                        mse=float(M.mse(y[te], a * p_te + (1 - a) * p_fin_te)),
                        tuned_on=tuned,
                        persistence_mse=pers,
                        note=f"alpha={a:.2f}",
                    )
                )
        fin = next(r["mse"] for r in rows if r["branch"] == "finance_svr" and r["horizon"] == tau)
        log(f"  tau={tau}: finance {fin:.3f}; persistence {pers:.3f}")
    t = pd.DataFrame(rows)
    t["n_test"] = len(te)
    t["published_mse"] = t["horizon"].map(PUBLISHED)
    t["text_features"] = "glove6b_mean_union_vocab"
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "result_table_6r_sawhney.csv", index=False, lineterminator="\n")
    return t
