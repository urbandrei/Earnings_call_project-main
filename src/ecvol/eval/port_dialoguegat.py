"""T6R.3 — DialogueGAT (Sang & Bao, Findings of EMNLP 2022), *harness port*.

A faithful run is impossible: the training pickle (`data/data_swd.pkl`) is not released,
the corpus is the authors' private SeekingAlpha re-scrape (2015–2018, real speaker names),
labels need a CRSP extract, and DGL ships no wheels for torch 2.11. The architecture and
protocol are ported from `gat.py` / `data_helper.py` / `preprocess.ipynb` onto the one
corpus we hold with genuine speaker turns — FinCall-Surprise — with every substitution
labelled:

* **Corpus:** FinCall turns (`transcript_json`, role-tagged), years 2019/2020/2021 in
  place of their 2015/2016/2017-18 splits; chronological sort then 70/10/20 by position
  inside each year (`get_data`). Labels: our trading-day `v_post(τ)` and `v_pre(τ)` as
  `v_past` (theirs: ln σ of CRSP RETX over τ+1 days incl. the call day).
* **Speakers:** their speaker nodes are *named persons shared across the whole year*;
  FinCall carries only roles (management / analyst / operator), so speaker nodes are
  role-typed and shared globally — closest to the paper's "random speaker embedding"
  ablation (Table 3), not to its headline row.
* **Text encoder:** their TextCNN (100 channels × kernels 3/4/5 over frozen 300-d GloVe,
  max-pooled → 300-d) — with GloVe **6B** instead of 840B; lower-cased regex tokens, stop
  words removed (`data_swd` = stop-words-deleted), ≤256 tokens per turn.
* **Graph:** utterance chain (bidirectional) + utterance↔speaker edges, homogeneous;
  5 GAT layers (PyG `GATConv(300, 300, heads=5, residual=True)`, head-mean, dropout 0.1)
  in place of DGL's; context attention over speaker nodes and over utterance nodes,
  `v_past → Linear(1, 300)`, `Linear(900, 1)`. Adam 1e-5, weight decay 1e-6, batch 4,
  ≤100 epochs with patience 5 on validation MSE, grad-clip 15, seed 1234 (theirs).
* Horizons 3/7/15 as published, plus 30 (the paper stops at 15).

Published (Table 2, their data, **reference only**): τ=3/7/15 — 2015 0.4530/0.3236/0.1898,
2016 0.4549/0.2884/0.1810, 2017-18 0.4090/0.2886/0.2036.
Output: `results/result_table_6r_dialoguegat.csv`.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import metrics as M
from ecvol.eval.port_sawhney import GLOVE_MEMBER, GLOVE_ZIP

TAUS = (3, 7, 15, 30)
YEARS = (2019, 2020, 2021)
MAX_TOKENS = 256
EMB = 300
N_STEPS = 5
N_HEADS = 5
DROPOUT = 0.1
LR = 1e-5
L2 = 1e-6
BATCH = 4
EPOCHS = 100
PATIENCE = 5
CLIP = 15.0
SEED = 1234
ROLES = ("management", "analyst", "operator", "unknown")
PUBLISHED = {  # year → τ → MSE (their corpus; reference only)
    2015: {3: 0.4530, 7: 0.3236, 15: 0.1898},
    2016: {3: 0.4549, 7: 0.2884, 15: 0.1810},
    2017: {3: 0.4090, 7: 0.2886, 15: 0.2036},
}
TOKEN_RE = re.compile(r"[a-z][a-z'-]+")


# --- corpus ---------------------------------------------------------------------


def load_calls(root: Path, dataset: str = "fincall") -> pd.DataFrame:
    calls = pd.read_parquet(
        root / dataset / "calls.parquet",
        columns=["call_id", "ticker", "call_date", "transcript_json", "status"],
    )
    calls = calls[(calls["status"] == "ok") & (calls["ticker"] != "")].copy()
    calls["call_id"] = calls["call_id"].astype(str)
    calls["year"] = calls["call_date"].str[:4].astype(int)
    return calls.reset_index(drop=True)


def turns(transcript_json: str) -> list[tuple[str, str]]:
    """[(role, text)] for non-empty turns."""
    out = []
    for t in json.loads(transcript_json):
        text = (t.get("text") or "").strip()
        if text:
            role = t.get("role") or "unknown"
            out.append((role if role in ROLES else "unknown", text))
    return out


def tokenize(text: str, stop: set[str]) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in stop]


def build_vocab(
    calls: pd.DataFrame, stop: set[str], root: Path
) -> tuple[dict[str, int], np.ndarray]:
    """Corpus words ∩ GloVe-6B → (word2idx, W[len+1, 300]); index 0 is padding."""
    corpus: set[str] = set()
    for tj in calls["transcript_json"]:
        for _, text in turns(tj):
            corpus.update(tokenize(text, stop))
    vec: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(root / GLOVE_ZIP) as z, z.open(GLOVE_MEMBER) as f:
        for raw in f:
            w, _, rest = raw.decode("utf-8").partition(" ")
            if w in corpus:
                vec[w] = np.asarray(rest.split(), dtype=np.float32)
    words = sorted(vec)
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    W = np.zeros((len(words) + 1, EMB), dtype=np.float32)
    for w, i in word2idx.items():
        W[i] = vec[w]
    return word2idx, W


def encode_call(transcript_json: str, word2idx: dict[str, int], stop: set[str]):
    """→ (token id matrix [n_turns, 256], role per turn). Turns with no known token are dropped."""
    ids, roles = [], []
    for role, text in turns(transcript_json):
        toks = [word2idx[w] for w in tokenize(text, stop) if w in word2idx][:MAX_TOKENS]
        if not toks:
            continue
        ids.append(toks + [0] * (MAX_TOKENS - len(toks)))
        roles.append(role)
    return np.asarray(ids, dtype=np.int64), roles


def year_split(calls: pd.DataFrame, year: int) -> pd.DataFrame:
    """`get_data`: keys of that year, sorted chronologically, 70/10/20 by position."""
    d = calls[calls["year"] == year].copy()
    d["key"] = d["call_date"].str.replace("-", "") + "_" + d["ticker"]
    d = d.sort_values("key", kind="stable").reset_index(drop=True)
    n = len(d)
    n_tr, n_va = int(n * 0.7), int(n * 0.1)
    d["split"] = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
    return d


# --- model (PyG) ------------------------------------------------------------------


def build_model(W: np.ndarray, n_parties: int):
    import torch
    from torch import nn
    from torch_geometric.nn import GATConv

    class ContextAttention(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.hidden = nn.Linear(d, d)
            self.context = nn.Linear(d, 1, bias=False)

        def forward(self, x):  # x: [n, d] for one graph → [d]
            alpha = torch.softmax(self.context(torch.tanh(self.hidden(x))), dim=0)
            return (alpha * x).sum(0)

    class DialogueGAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(W), freeze=True)
            self.convs = nn.ModuleList([nn.Conv2d(1, 100, kernel_size=(w, EMB)) for w in (3, 4, 5)])
            self.party_emb = nn.Embedding(n_parties, EMB)
            nn.init.orthogonal_(self.party_emb.weight)
            self.gats = nn.ModuleList(
                [GATConv(EMB, EMB, heads=N_HEADS, residual=True) for _ in range(N_STEPS)]
            )
            self.dropout = nn.Dropout(DROPOUT)
            self.party_attention = ContextAttention(EMB)
            self.sen_attention = ContextAttention(EMB)
            self.v_linear = nn.Linear(1, EMB)
            self.output = nn.Linear(EMB * 3, 1)

        def text_cnn(self, x):  # x: [n_utt, 256] ids → [n_utt, 300]
            e = self.word_emb(x).unsqueeze(1)
            pooled = [c(e).squeeze(-1).max(dim=2)[0] for c in self.convs]  # no ReLU, as theirs
            return torch.cat(pooled, dim=1)

        def forward(self, tokens, edge_index, is_utt, party_ids, graph_id, n_graphs, v_past):
            feat = torch.empty(is_utt.numel(), EMB, device=tokens.device)
            feat[is_utt] = self.text_cnn(tokens)
            feat[~is_utt] = self.party_emb(party_ids)
            h = feat
            for gat in self.gats:
                h = gat(h, edge_index).view(-1, N_HEADS, EMB).mean(1)
                h = self.dropout(h)
            px, sx = [], []
            for g in range(n_graphs):
                m = graph_id == g
                px.append(self.party_attention(h[m & ~is_utt]))
                sx.append(self.sen_attention(h[m & is_utt]))
            ox = torch.cat(
                [torch.stack(px), torch.stack(sx), self.v_linear(v_past.view(-1, 1))], -1
            )
            return self.output(ox).squeeze(-1)

    return DialogueGAT()


def collate(items, p2gid: dict[str, int], device):
    """Concatenate call graphs: utterance nodes then speaker nodes per call."""
    import torch

    tok, src, dst, is_utt, pids, gid, offset = [], [], [], [], [], [], 0
    for g, (ids, roles) in enumerate(items):
        n = len(roles)
        parties = sorted(set(roles))
        q2id = {p: i for i, p in enumerate(parties)}
        for i in range(n - 1):  # utterance chain, both directions
            src += [offset + i, offset + i + 1]
            dst += [offset + i + 1, offset + i]
        for i, r in enumerate(roles):  # utterance ↔ its speaker
            q = offset + n + q2id[r]
            src += [offset + i, q]
            dst += [q, offset + i]
        tok.append(torch.from_numpy(ids))
        is_utt += [True] * n + [False] * len(parties)
        pids += [p2gid[p] for p in parties]
        gid += [g] * (n + len(parties))
        offset += n + len(parties)
    return (
        torch.cat(tok).to(device),
        torch.tensor([src, dst], dtype=torch.long, device=device),
        torch.tensor(is_utt, device=device),
        torch.tensor(pids, dtype=torch.long, device=device),
        torch.tensor(gid, dtype=torch.long, device=device),
        len(items),
    )


def fit_eval(model, data, y, vp, idx, *, epochs=EPOCHS, seed=SEED, log=print):
    """Their `train.py` loop: Adam, MSE, clip 15, early stop (patience 5) on val loss;
    returns (test_mse, val_mse, best_epoch, test predictions)."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    p2gid = {r: i for i, r in enumerate(ROLES)}
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev)
    vp_t = torch.tensor(vp, dtype=torch.float32, device=dev)

    def run(ids, train):
        model.train(train)
        preds = np.zeros(len(ids))
        order = (
            ids if not train else ids[np.argsort([-len(data[i][1]) for i in ids], kind="stable")]
        )
        for s in range(0, len(order), BATCH):
            b = order[s : s + BATCH]
            tokens, ei, iu, pid, gid, ng = collate([data[i] for i in b], p2gid, dev)
            if train:
                opt.zero_grad()
                out = model(tokens, ei, iu, pid, gid, ng, vp_t[b])
                loss = torch.nn.functional.mse_loss(out, y_t[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
            else:
                with torch.no_grad():
                    out = model(tokens, ei, iu, pid, gid, ng, vp_t[b])
            pos = {c: k for k, c in enumerate(ids)}
            preds[[pos[c] for c in b]] = out.detach().cpu().numpy()
        return preds

    best, best_state, wait = np.inf, None, 0
    best_epoch = -1
    for e in range(epochs):
        run(idx["train"], True)
        val = float(M.mse(y[idx["val"]], run(idx["val"], False)))
        if val < best:
            best, wait, best_epoch = val, 0, e
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    p_te = run(idx["test"], False)
    return float(M.mse(y[idx["test"]], p_te)), best, best_epoch, p_te


def run_dialoguegat_port(
    root: Path, *, dataset: str = "fincall", years=YEARS, taus=TAUS, epochs=EPOCHS, log=print
) -> pd.DataFrame:
    import torch
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = set(ENGLISH_STOP_WORDS)
    calls = load_calls(root, dataset)
    word2idx, W = build_vocab(calls, stop, root)
    log(f"  {len(calls)} calls; vocab {len(word2idx)} GloVe-6B words")
    targets = pd.read_parquet(root / dataset / "targets.parquet")
    targets = targets[targets["status"] == "ok"]
    targets["call_id"] = targets["call_id"].astype(str)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for year in years:
        d = year_split(calls, year)
        enc = {
            c: encode_call(tj, word2idx, stop)
            for c, tj in zip(d["call_id"], d["transcript_json"], strict=True)
        }
        d = d[[len(enc[c][1]) > 0 for c in d["call_id"]]].reset_index(drop=True)
        for tau in taus:
            t = targets[targets["horizon"] == tau].set_index("call_id")
            dd = d[d["call_id"].isin(t.index)].reset_index(drop=True)
            y = t.loc[dd["call_id"], "v_post"].to_numpy(dtype=float)
            vp = t.loc[dd["call_id"], "v_pre"].to_numpy(dtype=float)
            data = [enc[c] for c in dd["call_id"]]
            idx = {s: np.where(dd["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}
            model = build_model(W, len(ROLES)).to(dev)
            test_mse, val_mse, best_epoch, _ = fit_eval(
                model, data, y, vp, idx, epochs=epochs, log=log
            )
            pers = float(M.mse(y[idx["test"]], vp[idx["test"]]))
            rows.append(
                {
                    "dataset": dataset,
                    "year": year,
                    "horizon": tau,
                    "n_train": len(idx["train"]),
                    "n_val": len(idx["val"]),
                    "n_test": len(idx["test"]),
                    "mse": test_mse,
                    "val_mse": val_mse,
                    "best_epoch": best_epoch,
                    "persistence_mse": pers,
                    "speaker_nodes": "role_typed_global",
                    "text_encoder": "textcnn_glove6b",
                    "published_reference": PUBLISHED.get(
                        {2019: 2015, 2020: 2016, 2021: 2017}[year], {}
                    ).get(tau, np.nan),
                }
            )
            log(
                f"  {year} tau={tau:<2} MSE {test_mse:.3f} (val {val_mse:.3f}, "
                f"epoch {best_epoch}, persistence {pers:.3f})"
            )
    table = pd.DataFrame(rows)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_6r_dialoguegat.csv", index=False, lineterminator="\n")
    return table
