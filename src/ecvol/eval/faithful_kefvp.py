"""T6R.3 — KeFVP (Niu et al., Findings of EMNLP 2023) with the authors' own code.

`final_series_infer.py` is run unmodified except for forward-port patches applied into a
scratch build (`data/work/kefvp`), never committed:

- every `/your/project/path/` and `/your/dataset/path/` placeholder → the build's
  `proj/` and `dataset/` directories (the script crashes at its log FileHandler otherwise);
- the log filename's `%H:%M:%S` → `%H-%M-%S` (colons are illegal in Windows names);
- `float(price_df[...]['future_label_τ'])` on a length-1 Series → `.iloc[0]` (pandas ≥ 2.2).

The EC headline setting is `run_ec_for_kept.sh` verbatim: CondAutoformer, 200 epochs,
lr 2e-4, weight decay 5e-2, mu 0.7, `--text_indim 1024`, the released KePt-BERT-large
embedding pickle (`text_embedding/<name>.pkl`, Drive-only — HANDOFF), 10 repeats. The
audio branch is commented out upstream (`CondInfer.forward`), so the faithful KeFVP is
text + price only and the missing HuBERT pickle is immaterial (zeros are substituted by
the script's own `try/except`).

MAEC-15/16 (`run_for_differ_threshold.sh`): `--text_embedding raw_bert_base_uncased`,
`--audio_indim 29`. Upstream never released that pickle ("too large"); we regenerate it
with the authors' `generatePtmEmbeddings.py` (patched: `Text.txt` → MAEC's `text.txt`;
sentences encoded in chunks of 64 instead of one call per transcript — numerically
identical, avoids OOM; only the folders named in the split files are encoded — the
script substitutes zeros for missing keys either way). That is a **substitution**
(our BERT-base pooler outputs, not theirs) and is labelled so.

Each run's ten test MSEs are read from the script's own
`output/3GCN_LSTM_boxplot_cond_avg_day_mse_df.csv` (overwritten per duration, so copied
after each). Output: `results/result_table_6r_kefvp.csv` — (dataset, horizon, run,
mse, mse_single) long rows plus per-(dataset, horizon) summary columns.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_REL = "raw/ref/repos/KeFVP"
WORK_REL = "work/kefvp"
TAUS = (3, 7, 15, 30)
REPEATS = 10  # `for i in range(10)` in the authors' main
PUBLISHED = {  # Table 2, mean of 10 runs (std)
    "ec": {3: 0.610, 7: 0.291, 15: 0.183, 30: 0.114},
    "15": {3: 0.418, 7: 0.187, 15: 0.122, 30: 0.087},
    "16": {3: 0.445, 7: 0.279, 15: 0.303, 30: 0.177},
}
PUBLISHED_STD = {
    "ec": {3: 0.0331, 7: 0.0133, 15: 0.0089, 30: 0.0063},
    "15": {3: 0.0123, 7: 0.0027, 15: 0.0032, 30: 0.0017},
    "16": {3: 0.0636, 7: 0.0442, 15: 0.0365, 30: 0.0333},
}
EC_EMBEDDING = "emnlp_202308_bert_large_unfreeze_6layers/ec_embed_bert_large_uncased_kept_epoch_6"
MAEC_EMBEDDING = "raw_bert_base_uncased"
MAEC_RAW_REL = "raw/maec/repo/MAEC_Dataset"
EC_RAW_REL = "raw/ec/extracted/ACL19_Release"


def patch_infer(src: str, proj: str, dataset_dir: str) -> str:
    """Forward-port patches for `kefvp/final_series_infer.py` (paths are POSIX-style)."""
    out = src.replace("/your/project/path/", proj.rstrip("/") + "/")
    out = out.replace("/your/dataset/path/", dataset_dir.rstrip("/") + "/")
    out = out.replace(
        'strftime("%Y-%m-%d_%H:%M:%S", localtime())', 'strftime("%Y-%m-%d_%H-%M-%S", localtime())'
    )
    sel = "price_df[price_df.text_file_name == row['text_file_name']]"
    for tau in TAUS:
        old = f"float({sel}['future_label_{tau}'])"
        new = f"float({sel}['future_label_{tau}'].iloc[0])"
        assert src.count(old) == 1, old
        out = out.replace(old, new)
    assert "/your/" not in out
    return out


def patch_generator(src: str, dataset_dir: str, wanted_list: str) -> str:
    """`pretrain/generatePtmEmbeddings.py`: MAEC file name, chunked encoding, folder subset."""
    out = src.replace(
        "text_path = args.data_path + data_dir + '/Text.txt'   # For ec",
        "text_path = args.data_path + data_dir + '/text.txt'   # patched: MAEC",
    )
    out = out.replace("data_dir = '/your/dataset/path'", f"data_dir = '{dataset_dir.rstrip('/')}/'")
    old_list = (
        "    data_list = os.listdir(args.data_path)\n    output_dct = {}\n    all_sent_num = []"
    )
    new_list = (
        "    data_list = os.listdir(args.data_path)\n"
        f"    _wanted = set(open('{wanted_list}').read().split('\\n'))\n"
        "    data_list = [d for d in data_list if d in _wanted]  # patched: split folders only\n"
        "    output_dct = {}\n    all_sent_num = []"
    )
    assert src.count(old_list) == 1
    out = out.replace(old_list, new_list)
    old_enc = (
        "        with torch.no_grad():\n"
        "            model_out = model(input['input_ids'].cuda(), input['attention_mask'].cuda())\n"
    )
    new_enc = (
        "        with torch.no_grad():  # patched: chunked encoding, same outputs\n"
        "            _outs = []\n"
        "            for _s in range(0, input['input_ids'].shape[0], 64):\n"
        "                _o = model(input['input_ids'][_s:_s+64].cuda(),\n"
        "                           input['attention_mask'][_s:_s+64].cuda())\n"
        "                _outs.append(_o['pooler_output'])\n"
        "            model_out = {'pooler_output': torch.cat(_outs, dim=0)}\n"
    )
    assert src.count(old_enc) == 1
    out = out.replace(old_enc, new_enc)
    assert "data_dir = '/your/dataset/path'" not in out  # argparse defaults are overridden
    return out


def prepare_build(root: Path) -> Path:
    work = (root / WORK_REL).resolve()  # absolute: the scripts run from their own cwd
    repo = (root / REPO_REL).resolve()
    if not work.exists():
        shutil.copytree(repo, work, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    proj = (work / "proj").as_posix()
    dataset_dir = (work / "dataset").as_posix()
    for sub in ("log", "output", "preds_dir", "save_pkls", "save_features"):
        (work / "proj" / sub).mkdir(parents=True, exist_ok=True)
    (work / "dataset" / "text_embedding").mkdir(parents=True, exist_ok=True)
    if not (work / "dataset" / "price_data").exists():
        shutil.copytree(repo / "price_data", work / "dataset" / "price_data")
    if not (work / "dataset" / "data_process").exists():
        shutil.copytree(repo / "data_process", work / "dataset" / "data_process")
    infer = repo / "kefvp" / "final_series_infer.py"
    (work / "kefvp" / "final_series_infer.py").write_text(
        patch_infer(infer.read_text(encoding="utf-8"), proj, dataset_dir), encoding="utf-8"
    )
    # matplotlib/pylab are imported at module level by the attention layers but only used
    # for plotting that the training path never reaches: shadow them with no-op stubs
    stub = work / "kefvp" / "matplotlib"
    stub.mkdir(exist_ok=True)
    (stub / "__init__.py").write_text(
        "# patched stub: plotting is never reached in training\n"
        "def use(*a, **k):\n    return None\n",
        encoding="utf-8",
    )
    (stub / "pyplot.py").write_text(
        "def __getattr__(name):\n    return lambda *a, **k: None\n", encoding="utf-8"
    )
    (work / "kefvp" / "pylab.py").write_text(
        "rcParams = {}\n\n\ndef __getattr__(name):\n    return lambda *a, **k: None\n",
        encoding="utf-8",
    )
    # `latent` (KumaGate / kumadist) is imported at module level but never shipped and never
    # instantiated on the CondAutoformer path: stub it so the import resolves, fail loudly if used
    lat = work / "kefvp" / "latent"
    (lat / "nn").mkdir(parents=True, exist_ok=True)
    (lat / "__init__.py").write_text(
        "# patched stub: package not released upstream\n", encoding="utf-8"
    )
    (lat / "nn" / "__init__.py").write_text("", encoding="utf-8")
    unavailable = (
        "class {name}:\n    def __init__(self, *a, **k):\n"
        "        raise ImportError('KeFVP `latent` package is not released upstream')\n"
    )
    (lat / "nn" / "kuma_gate.py").write_text(unavailable.format(name="KumaGate"), encoding="utf-8")
    (lat / "kumadist.py").write_text(
        unavailable.format(name="IndependentLatentModel")
        + unavailable.format(name="DependentLatentModel"),
        encoding="utf-8",
    )
    # the released script does `from data_utils import set_seed`, but kefvp/data_utils.py
    # never defines it (it lives in pretrain/data_utils_pretrain_with_kg.py) — copy theirs
    du = work / "kefvp" / "data_utils.py"
    if "def set_seed" not in du.read_text(encoding="utf-8"):
        src = (repo / "pretrain" / "data_utils_pretrain_with_kg.py").read_text(encoding="utf-8")
        start = src.index("def set_seed(seed: int):")
        end = src.index("\ndef ", start + 1)
        du.write_text(
            du.read_text(encoding="utf-8").rstrip("\n") + "\n\n\n# patched in from pretrain/"
            "data_utils_pretrain_with_kg.py (the authors' own function)\n" + src[start:end] + "\n",
            encoding="utf-8",
        )
    return work


def maec_folders(root: Path) -> list[str]:
    names: set[str] = set()
    for ds in ("15", "16"):
        for part in ("train", "dev", "test"):
            f = root / REPO_REL / "price_data" / "maec" / ds / f"maec{ds}_{part}_avg_val.csv"
            names |= set(pd.read_csv(f, usecols=["text_file_name"])["text_file_name"].astype(str))
    return sorted(names)


def generate_maec_embeddings(root: Path, *, log=print) -> Path:
    """Regenerate `dataset/text_embedding/raw_bert_base_uncased.pkl` with the authors' script."""
    work = prepare_build(root)
    out = work / "dataset" / "text_embedding" / f"{MAEC_EMBEDDING}.pkl"
    if out.is_file():
        return out
    wanted = work / "maec_split_folders.txt"
    wanted.write_text("\n".join(maec_folders(root)), encoding="utf-8")
    gen_src = (root / REPO_REL / "pretrain" / "generatePtmEmbeddings.py").read_text(
        encoding="utf-8"
    )
    gen = work / "pretrain" / "generatePtmEmbeddings_maec.py"
    gen.write_text(
        patch_generator(gen_src, (work / "dataset").as_posix(), wanted.as_posix()), encoding="utf-8"
    )
    raw = (root / MAEC_RAW_REL).resolve().as_posix() + "/"
    cmd = [sys.executable, "-u", gen.name, "--ptm_type", "bert-base-uncased", "--data_path", raw,
           "--max_sent", "512", "--save_path", out.as_posix()]  # fmt: skip
    log(f"  generating MAEC BERT-base embeddings for {len(maec_folders(root))} folders …")
    proc = subprocess.run(cmd, cwd=gen.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:])
    return out


def infer_args(dataset: str, tau: int, *, raw_data_path: str) -> list[str]:
    common = ["--weight_decay", "5e-2", "--lr", "0.0002", "--epochs", "200", "--pid", "0.95",
              "--time_model", "CondAutoformer", "--duration", str(tau), "--run_mode", "reg",
              "--d_layers", "2", "--dist_threshold", "0.95", "--mu", "0.7", "--dataset", dataset,
              "--raw_data_path", raw_data_path]  # fmt: skip
    if dataset == "ec":
        return common + ["--audio_indim", "768", "--text_indim", "1024", "--text_embedding",
                         EC_EMBEDDING, "--pkl_save_path", "save_pkls/ec_kept",
                         "--log_save_path", "ec_kept"]  # fmt: skip
    return common + ["--audio_indim", "29", "--text_embedding", MAEC_EMBEDDING,
                     "--pkl_save_path", f"save_pkls/maec{dataset}",
                     "--log_save_path", f"maec{dataset}"]  # fmt: skip


def run_dataset(root: Path, dataset: str, *, taus=TAUS, log=print) -> pd.DataFrame:
    work = prepare_build(root)
    emb = (
        work
        / "dataset"
        / "text_embedding"
        / f"{EC_EMBEDDING if dataset == 'ec' else MAEC_EMBEDDING}.pkl"
    )
    if not emb.is_file():
        raise FileNotFoundError(f"text embedding pickle missing: {emb}")
    raw = (root / (EC_RAW_REL if dataset == "ec" else MAEC_RAW_REL)).resolve().as_posix() + "/"
    for sub in ("log/" + ("ec_kept" if dataset == "ec" else f"maec{dataset}"), "preds_dir/reg"):
        (work / "proj" / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for tau in taus:
        log(f"  KeFVP {dataset} tau={tau}: {REPEATS} repeats × 200 epochs …")
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                "final_series_infer.py",
                *infer_args(dataset, tau, raw_data_path=raw),
            ],  # noqa: E501
            cwd=work / "kefvp",
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"KeFVP {dataset} tau={tau} failed:\n{proc.stderr[-3000:]}")
        avg = pd.read_csv(work / "proj" / "output" / "3GCN_LSTM_boxplot_cond_avg_day_mse_df.csv")
        single = pd.read_csv(
            work / "proj" / "output" / "3GCN_LSTM_boxplot_cond_single_day_mse_df.csv"
        )
        for i, (a, s) in enumerate(zip(avg.iloc[:, -1], single.iloc[:, -1], strict=True)):
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": tau,
                    "run": i,
                    "mse": float(a),
                    "mse_single": float(s),
                }
            )
        log(f"    mean MSE {avg.iloc[:, -1].mean():.3f} (published {PUBLISHED[dataset][tau]:.3f})")
    t = pd.DataFrame(rows)
    g = t.groupby(["dataset", "horizon"])["mse"]
    t = t.merge(
        g.agg(mse_mean="mean", mse_std="std", mse_best="min").reset_index(),
        on=["dataset", "horizon"],
    )
    t["published_mse"] = [PUBLISHED[d][h] for d, h in zip(t["dataset"], t["horizon"], strict=True)]
    t["published_std"] = [
        PUBLISHED_STD[d][h] for d, h in zip(t["dataset"], t["horizon"], strict=True)
    ]
    t["text_embedding"] = (
        "released_kept_bert_large" if dataset == "ec" else "regenerated_bert_base_pooler"
    )
    return t


def write_table(root: Path, parts: list[pd.DataFrame]) -> Path:
    out = root / "results" / "result_table_6r_kefvp.csv"
    prev = pd.read_csv(out) if out.is_file() else None
    new = pd.concat(parts, ignore_index=True)
    if prev is not None:  # keep other datasets' rows from earlier runs of the command
        prev = prev[~prev["dataset"].astype(str).isin(new["dataset"].astype(str).unique())]
        new = pd.concat([prev, new], ignore_index=True)
    new["dataset"] = new["dataset"].astype(str)
    new = new.sort_values(["dataset", "horizon", "run"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out, index=False, lineterminator="\n")
    return out
