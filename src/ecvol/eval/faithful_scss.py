"""T6R.3 — Same-Company-Same-Signal's trained models run with the authors' own code.

The repository (`raw/ref/repos/SCSS_tree`, materialised file-by-file because 304 result
filenames contain `|` and cannot exist on Windows) is copied into a scratch build
directory and four forward-port patches are applied — nothing is committed:

- `data_provider/data_loader.py`: drop `from sktime.datasets import …` (unused symbol,
  removed from current sktime);
- `utils/reconfig_args.py`: `rt({year}|{quarter})` → `rt({year}-{quarter})` (`|` is
  illegal in Windows paths);
- `utils/tools.py`: `np.Inf` → `np.inf` (NumPy 2); matplotlib import → no-op (plotting
  is never reached).

Everything else — data loading, masks, TSMixer/TMLP, seed 2021, early stopping,
`drop_last=True` on train/val — is the authors' code. Each run writes
`earnings_results/<model>/<model_id>_<mse>.csv` exactly as upstream; the published MSE
for the same cell is read from the upstream filenames (`raw/ref/scss/
earnings_results_index.txt`, 304 entries). Output: `results/result_table_6r_scss_<model>.csv`
with (model, embedding, window, year, quarter, mse, published_mse, n_test).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

TREE_REL = "raw/ref/repos/SCSS_tree/SameCompanySameSignal"
INDEX_REL = "raw/ref/scss/earnings_results_index.txt"
WORK_REL = "work/scss"
EMBEDDINGS_REL = "raw/ref/scss_drive/Embeddings/openai"
YEARS = (2019, 2020, 2021, 2022, 2023)
QUARTERS = ("first", "second", "third", "fourth")
WINDOWS = (3, 7, 15, 30)
PATCHES: tuple[tuple[str, str, str], ...] = (
    (
        "data_provider/data_loader.py",
        "from sktime.datasets import load_from_tsfile_to_dataframe\n",
        "",
    ),
    (
        "utils/reconfig_args.py",
        "_rt({args.test_year}|{args.test_quarter})",
        "_rt({args.test_year}-{args.test_quarter})",
    ),
    ("utils/tools.py", "np.Inf", "np.inf"),
    (
        "utils/tools.py",
        "import matplotlib.pyplot as plt\n",
        "import types as _t\n"
        "plt = _t.SimpleNamespace(switch_backend=lambda *a, **k: None)  # patched: no plotting\n",
    ),
)
NAME_RE = re.compile(
    r"^(?P<model_id>.+)_win(?P<win>\d+)_rt\((?P<year>\d{4})[|-](?P<q>\w+)\)_(?P<mse>[\d.]+)\.csv$"
)


def apply_patches(text_by_file: dict[str, str]) -> dict[str, str]:
    """Pure function over {relative path: source}; every patch must hit exactly once."""
    out = dict(text_by_file)
    for rel, old, new in PATCHES:
        if rel not in out:
            continue
        assert out[rel].count(old) >= 1, f"patch target not found: {rel}: {old!r}"
        out[rel] = out[rel].replace(old, new)
    return out


def prepare_build(root: Path) -> Path:
    """Fresh copy of the mirrored tree + patches (always rebuilt: patches are not idempotent)."""
    work = root / WORK_REL
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(root / TREE_REL, work, ignore=shutil.ignore_patterns("__pycache__"))
    files = {rel: (work / rel).read_text(encoding="utf-8") for rel, _, _ in PATCHES}
    for rel, text in apply_patches(files).items():
        (work / rel).write_text(text, encoding="utf-8")
    # the OpenAI embeddings (Drive-only, mirrored under raw/ref/scss_drive) go where the
    # loader expects them; the Drive file `DEC2RandomTicker.npz` is the script's
    # `DECRandomTicker` (same 1800 ids, same shape) — renamed on copy, recorded in the ledger
    src = root / EMBEDDINGS_REL
    if src.is_dir():
        dst = work / "dataset" / "Embeddings" / "openai"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.npz"):
            shutil.copy2(f, dst / f.name.replace("DEC2RandomTicker", "DECRandomTicker"))
    return work


def parse_name(name: str) -> dict | None:
    m = NAME_RE.match(name)
    if not m:
        return None
    return {
        "model_id": m["model_id"],
        "window": int(m["win"]),
        "year": int(m["year"]),
        "quarter": m["q"],
        "mse": float(m["mse"]),
    }


def published_index(root: Path) -> pd.DataFrame:
    rows = []
    for line in (root / INDEX_REL).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("/")
        rec = parse_name(parts[-1])
        if rec:
            rec["model"] = parts[2]  # earnings_results/<model>/...
            rec["embedding"] = parts[3] if len(parts) > 4 else ""
            rows.append(rec)
    return pd.DataFrame(rows).rename(columns={"mse": "published_mse"})


def tsmixer_args(window: int, year: int, quarter: str) -> list[str]:
    """The exact argument list of `scripts/TSMixer_DEC.sh`, plus `--num_workers 0`."""
    return [
        "--task_name", "ts_earnings", "--is_training", "1", "--root_path", "./dataset",
        "--data_path", "DEC.csv", "--model_id", "mTSMixer_dDEC_seq22", "--ts_pattern",
        "volatility", "--model", "TSMixer", "--data", "ts_earnings", "--features", "S",
        "--freq", "b", "--seq_len", "22", "--label_len", "22", "--pred_len", "1",
        "--e_layers", "2", "--d_layers", "1", "--factor", "3", "--enc_in", "1", "--dec_in",
        "1", "--c_out", "1", "--batch_size", "16", "--d_model", "512", "--des", "Exp",
        "--itr", "1", "--learning_rate", "0.001", "--loss", "MSE", "--prediction_window",
        str(window), "--test_year", str(year), "--test_quarter", quarter, "--num_workers", "0",
    ]  # fmt: skip


def tmlp_args(window: int, year: int, quarter: str, emb_file: str) -> list[str]:
    """`scripts/TMLP_DEC.sh` (OpenAI text-embedding-3-large vectors, 3072-d)."""
    return [
        "--task_name", "text_earnings", "--is_training", "1", "--root_path", "./dataset",
        "--data_path", "DEC.csv", "--model_id", f"mTMLP_dDEC_e({emb_file})", "--model", "TMLP",
        "--data", "text_earnings", "--emb_vendor", "openai", "--emb_file", emb_file,
        "--emb_dim", "3072", "--batch_size", "16", "--des", "Exp", "--itr", "1",
        "--learning_rate", "0.001", "--loss", "MSE", "--prediction_window", str(window),
        "--test_year", str(year), "--test_quarter", quarter, "--num_workers", "0",
    ]  # fmt: skip


def run_grid(root: Path, model: str, *, emb_file: str = "DEC", log=print) -> pd.DataFrame:
    work = prepare_build(root)
    results_dir = work / "earnings_results" / model
    if results_dir.exists():
        shutil.rmtree(results_dir)
    cells = [
        (w, y, q)
        for w in WINDOWS
        for y in YEARS
        for q in QUARTERS
        if not (y == 2019 and q == "first")
    ]
    for i, (w, y, q) in enumerate(cells, 1):
        args = tsmixer_args(w, y, q) if model == "TSMixer" else tmlp_args(w, y, q, emb_file)
        proc = subprocess.run(
            [sys.executable, "-u", "run.py", *args], cwd=work, capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{model} win{w} {y}-{q} failed:\n{proc.stderr[-2000:]}")
        log(f"  [{i:3d}/{len(cells)}] {model} win{w:<2} {y}-{q}")
    rows = []
    for f in sorted(results_dir.rglob("*.csv")):
        rec = parse_name(f.name)
        if rec:
            rec["n_test"] = int(len(pd.read_csv(f)))
            rec["embedding"] = f.parent.name if f.parent != results_dir else ""
            rows.append(rec)
    ours = pd.DataFrame(rows)
    pub = published_index(root)
    pub = pub[pub["model"] == model]
    keys = ["model_id", "window", "year", "quarter"]
    table = ours.merge(pub[keys + ["published_mse"]], on=keys, how="left")
    table.insert(0, "model", model)
    table["abs_diff"] = (table["mse"] - table["published_mse"]).abs()
    table = table.sort_values(["embedding", "window", "year", "quarter"]).reset_index(drop=True)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        out / f"result_table_6r_scss_{model.lower()}.csv", index=False, lineterminator="\n"
    )
    return table
