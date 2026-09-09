"""T6R.3 — HTML (Yang et al. 2020) run *faithfully*: the authors' own model classes on
the authors' own data, with only the driver rewritten (DECISIONS 2026-09-09).

What is verbatim, what is not (the fidelity ledger, `docs/faithful_ledger.md`):

* **Verbatim** — `SelfAttention`, `TransformerBlock`, `RTransformer` from the mirrored
  repository (`Model/Sentence-Level-Transformer/transformers/transformers_gpu.py`),
  loaded by `exec` of the class definitions from line 27 onward. The 26 import lines
  above them are replaced (they pull TensorFlow 1's `set_random_seed`, torchtext,
  matplotlib and a package the repo never ships); nothing inside the classes changes.
* **Rebuilt** — the inputs. The 1024-d sentence vectors exist only behind a dead Drive
  link; we recompute them with the recipe in the repo's `Bert-As-A-Service-Readme.md`:
  BERT-WWM-Large, mean over tokens of hidden layer −2, one `TextSequence.txt` line per
  sentence (`encode_ec_sentences`). Labels are the lineage's shipped files (VolTAGE
  `future_τ`/`past_τ`, KeFVP `future_Single_τ`), the same files `reproduce.py` uses.
* **Rewritten** — the driver `run_gpu.go`. As shipped it splits features, main labels
  and auxiliary labels with three *independent, unseeded* `train_test_split` calls,
  so X and y are decorrelated; it also seeds nothing and its lr warm-up is a no-op.
  Three conditions are run, each labelled:
    - `code`:  random 70/10/20 split (aligned), dropout 0.0 (the code's value), 10
      epochs, Adam 2e-5, batch 4, grad-clip 1.0, min-over-epochs on validation —
      the shipped protocol with the decorrelation bug removed.
    - `paper`: chronological 7:1:2 split (paper §5.3), dropout 0.5 (Table 1),
      otherwise as `code`.
    - `published_split`: VolTAGE `*_split3.csv` (the split every later paper reuses),
      dropout 0.5.
  α is tuned on validation over the paper's grid {0.1,…,1.0} with seed 0; the test MSE
  is read at the epoch of minimum validation MSE (the code's selection rule); seeds
  0,1,2 give the spread. Persistence (`past_τ`) is the floor on every cell.

Output: `results/result_table_6r_html_faithful.csv`.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import metrics as M
from ecvol.eval.reproduce import TAUS, published_labels

REPO_REL = "raw/ref/repos/HTML"
MODEL_FILE = "Model/Sentence-Level-Transformer/transformers/transformers_gpu.py"
ENCODER = "google-bert/bert-large-uncased-whole-word-masking"
CACHE_FILE = "html_bert_wwm_large_l2_sentences.parquet"
MAX_SENTENCES = 520  # the authors' `max_length`
EPOCHS = 10
LR = 2e-5
BATCH = 4
CLIP = 1.0
ALPHAS = tuple(round(a, 1) for a in np.arange(0.1, 1.05, 0.1))
PUBLISHED_TEXT_MSE = {3: 1.175, 7: 0.372, 15: 0.153, 30: 0.133}
CONDITIONS = {"code": 0.0, "paper": 0.5, "published_split": 0.5}  # → dropout


# --- the authors' classes, verbatim -------------------------------------------


def load_authors_classes(source: str):
    """Exec the class definitions of `transformers_gpu.py` (from `class SelfAttention`)
    in a namespace that provides what its stripped import block provided."""
    import math

    import torch
    import torch.nn.functional as F
    from torch import nn

    start = source.index("class SelfAttention")
    body = source[start:]
    util = types.SimpleNamespace(contains_nan=lambda t: bool((t != t).sum() > 0))

    def d(tensor=None):
        if tensor is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if tensor.is_cuda else "cpu"

    def mask_(matrices, maskval=0.0, mask_diagonal=True):
        b, h, w = matrices.size()
        idx = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
        matrices[:, idx[0], idx[1]] = maskval

    ns = {"torch": torch, "nn": nn, "F": F, "math": math, "util": util, "d": d, "mask_": mask_}
    exec(compile(body, "transformers_gpu.py", "exec"), ns)  # noqa: S102 — mirrored source
    return ns["RTransformer"]


def authors_model(root: Path, emb: int, dropout: float):
    src = (root / REPO_REL / MODEL_FILE).read_text(encoding="utf-8")
    RTransformer = load_authors_classes(src)
    # depth 2, heads 2, seq_length 520, mean pooling: Table 1 / the notebooks' args
    return RTransformer(
        emb=emb,
        heads=2,
        depth=2,
        seq_length=MAX_SENTENCES,
        num_tokens=0,
        num_classes=1,
        max_pool=False,
        dropout=dropout,
    )


# --- inputs ---------------------------------------------------------------------


def ec_sentences(root: Path) -> dict[str, list[str]]:
    base = root / "raw/ec/extracted/ACL19_Release"
    out = {}
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        f = folder / "TextSequence.txt"
        if f.is_file():
            lines = [
                ln.strip() for ln in f.read_text(encoding="utf-8", errors="replace").splitlines()
            ]
            out[folder.name] = [ln for ln in lines if ln]
    return out


def encode_ec_sentences(root: Path, *, batch_size: int = 32, device: str | None = None) -> Path:
    """BERT-WWM-Large, hidden layer −2, mean over non-special tokens → cache parquet
    (call_id, sent_idx, vector[1024] float32). Idempotent: returns the cache if present."""
    cache = root / "ec" / "cache" / CACHE_FILE
    if cache.is_file():
        return cache
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER, output_hidden_states=True).to(device).eval()
    rows = []
    for call_id, sents in ec_sentences(root).items():
        for i in range(0, len(sents), batch_size):
            chunk = sents[i : i + batch_size]
            enc = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc).hidden_states[-2]  # layer −2, as the bert-as-service recipe
            mask = enc["attention_mask"].clone()
            # drop [CLS] and [SEP] from the mean (bert-as-service REDUCE_MEAN excludes them)
            mask[:, 0] = 0
            last = enc["attention_mask"].sum(1) - 1
            mask[torch.arange(mask.size(0)), last] = 0
            m = mask.unsqueeze(-1).to(hs.dtype)
            vec = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
            vec = vec.float().cpu().numpy()
            for j, v in enumerate(vec):
                rows.append({"call_id": call_id, "sent_idx": i + j, "vector": v.tolist()})
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(cache, index=False)
    return cache


def sentence_arrays(root: Path) -> dict[str, np.ndarray]:
    cache = pd.read_parquet(encode_ec_sentences(root))
    cache = cache.sort_values(["call_id", "sent_idx"])
    return {
        str(c): np.vstack(g["vector"].to_numpy()).astype(np.float32)[:MAX_SENTENCES]
        for c, g in cache.groupby("call_id", sort=False)
    }


def _pad(seqs: list[np.ndarray]) -> np.ndarray:
    t = max(s.shape[0] for s in seqs)
    out = np.zeros((len(seqs), t, seqs[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


# --- splits ---------------------------------------------------------------------


def chronological_split(labels: pd.DataFrame, call_ids: list[str]) -> pd.DataFrame:
    """Paper §5.3: sort calls by date, then 7:1:2 train/val/test (earliest first)."""
    d = labels.loc[call_ids, ["year", "month", "day"]].astype(int)
    order = d.sort_values(["year", "month", "day"], kind="stable").index.tolist()
    n = len(order)
    n_tr, n_va = int(0.7 * n), int(0.1 * n)
    split = {c: "train" for c in order[:n_tr]}
    split.update({c: "val" for c in order[n_tr : n_tr + n_va]})
    split.update({c: "test" for c in order[n_tr + n_va :]})
    return pd.DataFrame({"call_id": order, "split": [split[c] for c in order]})


def random_split(call_ids: list[str], seed: int) -> pd.DataFrame:
    """The code's 70/10/20 (`test_size=0.2` then `0.125`), one permutation for X and y."""
    rng = np.random.RandomState(seed)
    order = list(np.array(call_ids)[rng.permutation(len(call_ids))])
    n = len(order)
    n_te = int(round(0.2 * n))
    n_va = int(round(0.125 * (n - n_te)))
    rest = order[n_te:]
    split = {c: "test" for c in order[:n_te]}
    split.update({c: "val" for c in rest[:n_va]})
    split.update({c: "train" for c in rest[n_va:]})
    return pd.DataFrame({"call_id": order, "split": [split[c] for c in order]})


# --- the faithful training loop -------------------------------------------------


def fit_eval(
    root: Path,
    S: list[np.ndarray],
    y: np.ndarray,
    ya: np.ndarray,
    idx: dict[str, np.ndarray],
    *,
    alpha: float,
    dropout: float,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[float, float, int]:
    """Train as `run_gpu.go` does (Adam 2e-5, batch 4, clip 1.0, MSE multi-task loss,
    no warm-up, no shuffling between epochs) and return (val_mse_min, test_mse_at_that
    epoch, best_epoch)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = authors_model(root, S[0].shape[1], dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr = idx["train"]

    def predict(ids: np.ndarray) -> np.ndarray:
        model.train(False)
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), 16):
                xb = torch.from_numpy(_pad([S[k] for k in ids[i : i + 16]])).to(dev)
                pa, _ = model(xb)
                out.append(pa.reshape(-1).cpu().numpy())
        return np.concatenate(out)

    best = (np.inf, np.nan, -1)
    for e in range(epochs):
        model.train(True)
        for i in range(0, len(tr), BATCH):
            b = tr[i : i + BATCH]
            xb = torch.from_numpy(_pad([S[k] for k in b])).to(dev)
            yb = torch.tensor(y[b], dtype=torch.float32, device=dev)
            yab = torch.tensor(ya[b], dtype=torch.float32, device=dev)
            opt.zero_grad()
            pa, pb = model(xb)
            loss = alpha * F.mse_loss(pa.reshape(-1), yb) + (1 - alpha) * F.mse_loss(
                pb.reshape(-1), yab
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        val_mse = float(M.mse(y[idx["val"]], predict(idx["val"])))
        if val_mse < best[0]:
            best = (val_mse, float(M.mse(y[idx["test"]], predict(idx["test"]))), e)
    return best


def run_html_faithful(
    root: Path,
    *,
    conditions=tuple(CONDITIONS),
    seeds=(0, 1, 2),
    epochs: int = EPOCHS,
    alphas=ALPHAS,
) -> pd.DataFrame:
    seqs = sentence_arrays(root)
    labels = published_labels(root)
    labels = labels[labels.index.isin(seqs)]
    call_ids = labels.index.tolist()
    rows = []
    for cond in conditions:
        dropout = CONDITIONS[cond]
        for tau in TAUS:
            y_all = labels[f"future_{tau}"].to_numpy(dtype=float)
            ya_all = labels[f"future_Single_{tau}"].to_numpy(dtype=float)
            base_all = labels[f"past_{tau}"].to_numpy(dtype=float)
            keep = np.isfinite(y_all) & np.isfinite(ya_all) & np.isfinite(base_all)
            ids = [c for c, k in zip(call_ids, keep, strict=True) if k]
            y, ya, base = y_all[keep], ya_all[keep], base_all[keep]
            S = [seqs[c] for c in ids]
            per_seed = []
            for seed in seeds:
                if cond == "code":
                    split = random_split(ids, seed)
                elif cond == "paper":
                    split = chronological_split(labels, ids)
                else:
                    split = pd.read_csv(root / "splits" / "ec_published.csv", dtype=str)
                    split = split[split["call_id"].isin(ids)]
                pos = {c: i for i, c in enumerate(ids)}
                idx = {
                    s: np.array([pos[c] for c in split.loc[split["split"] == s, "call_id"]], int)
                    for s in ("train", "val", "test")
                }
                # α on validation with seed 0 of this condition, then held fixed
                if seed == seeds[0]:
                    scores = {
                        a: fit_eval(
                            root, S, y, ya, idx, alpha=a, dropout=dropout, seed=seed, epochs=epochs
                        )
                        for a in alphas
                    }
                    best_alpha = min(scores, key=lambda a: scores[a][0])
                    val_mse, test_mse, ep = scores[best_alpha]
                else:
                    val_mse, test_mse, ep = fit_eval(
                        root,
                        S,
                        y,
                        ya,
                        idx,
                        alpha=best_alpha,
                        dropout=dropout,
                        seed=seed,
                        epochs=epochs,
                    )
                pers = float(M.mse(y[idx["test"]], base[idx["test"]]))
                per_seed.append((val_mse, test_mse, ep, pers, len(idx["test"])))
            arr = np.array([p[:2] for p in per_seed])
            rows.append(
                {
                    "model": "html_text_faithful",
                    "condition": cond,
                    "dropout": dropout,
                    "horizon": tau,
                    "alpha": best_alpha,
                    "n_calls": len(ids),
                    "n_test": per_seed[0][4],
                    "val_mse_min": float(arr[:, 0].mean()),
                    "mse": float(arr[:, 1].mean()),
                    "mse_seed_std": float(arr[:, 1].std(ddof=0)),
                    "best_epoch_mean": float(np.mean([p[2] for p in per_seed])),
                    "persistence_mse": float(np.mean([p[3] for p in per_seed])),
                    "published_mse": PUBLISHED_TEXT_MSE[tau],
                    "n_seeds": len(seeds),
                }
            )
    table = pd.DataFrame(rows)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_6r_html_faithful.csv", index=False, lineterminator="\n")
    return table
