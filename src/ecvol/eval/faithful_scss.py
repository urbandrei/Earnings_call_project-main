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
            [sys.executable, "-u", "run.py", *args],
            cwd=work,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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


# --- S4: Aug_PEV / Aug_STPEV on EC and MAEC (the authors' notebook `SS.ipynb`) ---------
#
# The notebook's three functions (`build_aug_stpev_mean`, `run_pev_stpev`, `run_EC_MAEC`)
# are exec-loaded from their cells unchanged and run on the shipped `dataset/EC|MAEC`
# files exactly as the notebook's driver cells do. The published MSEs are the tables the
# notebook printed when the authors ran it (stored cell outputs); their code rounds every
# MSE to 3 decimals, so "reproduced" here means equal at that precision.

NOTEBOOK_REL = "raw/ref/repos/SCSS_tree/SS.ipynb"
AUG_FUNCTIONS = ("build_aug_stpev_mean", "run_pev_stpev", "run_EC_MAEC")
AUG_METHODS = ("PEV", "STPEV", "Aug_PEV", "Aug_STPEV")
AUG_DATASETS = {  # key → (earnings file, augmented history file) under dataset/
    "ec": ("EC/EC_earnings.csv", "EC/augmented_EC_earnings_history.csv"),
    "maec15": ("MAEC/MAEC15_earnings.csv", "MAEC/augmented_MAEC15_earnings_history.csv"),
    "maec16": ("MAEC/MAEC16_earnings.csv", "MAEC/augmented_MAEC16_earnings_history.csv"),
}
_TABLE_ROW = re.compile(r"^(PEV|STPEV|Aug_PEV|Aug_STPEV)\s*\|((?:\s*[\d.]+\s*\|?)+)$")


def parse_printed_table(text: str) -> dict[str, list[float]]:
    """{method: [mse τ=3, 7, 15, 30]} from the notebook's printed comparison table."""
    out = {}
    for line in text.splitlines():
        m = _TABLE_ROW.match(line.strip())
        if m:
            vals = [float(v) for v in m[2].split("|") if v.strip()]
            out[m[1]] = vals[1:]  # first column is the mean over windows
    return out


def load_notebook_functions(nb: dict) -> dict:
    """Exec the code cells that define the three functions, verbatim."""
    import numpy as np

    ns: dict = {"pd": pd, "np": np}
    for name in AUG_FUNCTIONS:
        cells = [
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "code" and f"def {name}(" in "".join(c["source"])
        ]
        assert len(cells) == 1, f"{name}: expected one defining cell, found {len(cells)}"
        exec(cells[0], ns)  # noqa: S102 — third-party code, pinned commit, see ledger
    return ns


def published_aug(nb: dict) -> dict[str, dict[str, list[float]]]:
    """{dataset: {method: [4 MSEs]}} from the stored outputs of the notebook's driver cells."""
    out = {}
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] != "code" or "results = run_EC_MAEC(" not in src:
            continue
        key = next(k for k, (f, _) in AUG_DATASETS.items() if f.split("/")[-1] in src)
        text = "".join("".join(o.get("text", [])) for o in c.get("outputs", []))
        out[key] = parse_printed_table(text)
    return out


def history_window_crossings(
    earnings: pd.DataFrame, history: pd.DataFrame, tau: int
) -> tuple[int, int]:
    """(history rows the notebook uses, of which the τ-session window reaches the test period).

    Mirrors `build_aug_stpev_mean`'s filter (`day_earnings < min test date`, test tickers);
    a row "crosses" when day_earnings + τ business days ≥ the first test date (weekday
    approximation, no holiday calendar) — i.e. its target overlaps the test window.
    """
    import numpy as np

    test = earnings[earnings["cate"] == "test"]
    first = min(test["day_earnings"])
    used = history[(history["day_earnings"] < first) & history["ticker"].isin(test["ticker"])]
    days = used["day_earnings"].to_numpy(dtype="datetime64[D]")
    ends = np.busday_offset(days, tau, roll="forward")
    return int(len(used)), int((ends >= np.datetime64(first, "D")).sum())


def run_aug_baselines(root: Path) -> pd.DataFrame:
    import contextlib
    import io
    import json

    nb = json.loads((root / NOTEBOOK_REL).read_text(encoding="utf-8"))
    ns = load_notebook_functions(nb)
    published = published_aug(nb)
    data = root / TREE_REL / "dataset"
    rows = []
    for key, (earn_file, hist_file) in AUG_DATASETS.items():
        earnings = pd.read_csv(data / earn_file)
        if key != "ec":  # the notebook's MAEC driver cells
            earnings["day_earnings"] = earnings["time"]
        history = pd.read_csv(data / hist_file)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            aug = ns["build_aug_stpev_mean"](earnings, history)
            results = ns["run_EC_MAEC"](earnings, aug)
        n_test = int((earnings["cate"] == "test").sum())
        for method, mses in zip(AUG_METHODS, results, strict=True):
            for tau, mse in zip(WINDOWS, mses, strict=True):
                n_hist, n_cross = history_window_crossings(earnings, history, tau)
                pub = published[key][method][WINDOWS.index(tau)]
                rows.append(
                    {
                        "dataset": key,
                        "method": method,
                        "window": tau,
                        "mse": float(mse),
                        "published_mse": pub,
                        "abs_diff": abs(float(mse) - pub),
                        "n_test": n_test,
                        "n_aug_history": n_hist if method.startswith("Aug") else 0,
                        "n_aug_history_window_crosses_test": n_cross
                        if method.startswith("Aug")
                        else 0,
                    }
                )
    table = pd.DataFrame(rows)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_6r_scss_aug.csv", index=False, lineterminator="\n")
    return table


# --- T6R.4 controls: TSMixer / TMLP with only DEC's rolling masks rewritten ------------
#
# DECISIONS 2026-09-16. The authors' code is untouched; each condition is a copy of DEC.csv
# (same rows, same order — TMLP indexes its embeddings by row) whose
# `rolling_test_on_*_cate` columns are rewritten, passed via `--data_path`:
#   anchor_heldout  — their masks, test restricted to a seeded held-out third of tickers
#                     (the paired anchor: same test calls as `ticker_disjoint`);
#   ticker_disjoint — as anchor_heldout, and the held-out tickers' history leaves train/val;
#   embargoed       — their masks with train/val calls whose τ≤30-session window reaches the
#                     test quarter's first call removed (weekday approximation).

CONTROL_CONDITIONS = ("anchor_heldout", "ticker_disjoint", "embargoed")
CONTROL_SEED = 20260916
EMBARGO_SESSIONS = 30


def heldout_tickers(tickers, seed: int = CONTROL_SEED) -> set[str]:
    import numpy as np

    uniq = sorted(set(tickers))
    perm = np.random.RandomState(seed).permutation(len(uniq))
    return {uniq[i] for i in perm[: len(uniq) // 3]}


def control_masks(dec: pd.DataFrame, condition: str, seed: int = CONTROL_SEED) -> pd.DataFrame:
    import numpy as np

    out = dec.copy()
    held = out["ticker"].isin(heldout_tickers(out["ticker"], seed))
    days = out["day_earnings"].to_numpy(dtype="datetime64[D]")
    for col in [c for c in out.columns if c.startswith("rolling_test_on_")]:
        cate = out[col].to_numpy(dtype=object).copy()
        fit = np.isin(cate, ["train", "val"])
        if condition in ("anchor_heldout", "ticker_disjoint"):
            cate[(cate == "test") & ~held.to_numpy()] = "none"
            if condition == "ticker_disjoint":
                cate[fit & held.to_numpy()] = "none"
        elif condition == "embargoed":
            first = days[cate == "test"].min()
            reach = np.busday_offset(days, EMBARGO_SESSIONS, roll="forward") >= first
            cate[fit & reach] = "none"
        else:
            raise ValueError(condition)
        out[col] = cate
    return out


def _with(args: list[str], flag: str, value: str) -> list[str]:
    out = list(args)
    out[out.index(flag) + 1] = value
    return out


def run_controls(root: Path, model: str, *, log=print) -> pd.DataFrame:
    import numpy as np

    work = prepare_build(root)
    dec = pd.read_csv(work / "dataset" / "DEC.csv")
    for cond in CONTROL_CONDITIONS:
        control_masks(dec, cond).to_csv(work / "dataset" / f"DEC_{cond}.csv", index=False)
    results_dir = work / "earnings_results" / model
    if results_dir.exists():
        shutil.rmtree(results_dir)
    cells = [
        (c, w, y, q)
        for c in CONTROL_CONDITIONS
        for w in WINDOWS
        for y in YEARS
        for q in QUARTERS
        if not (y == 2019 and q == "first")
    ]
    for i, (cond, w, y, q) in enumerate(cells, 1):
        args = tsmixer_args(w, y, q) if model == "TSMixer" else tmlp_args(w, y, q, "DEC")
        base_id = args[args.index("--model_id") + 1]
        args = _with(args, "--data_path", f"DEC_{cond}.csv")
        args = _with(args, "--model_id", base_id.replace("_dDEC", f"_dDEC_{cond}"))
        proc = subprocess.run(
            [sys.executable, "-u", "run.py", *args],
            cwd=work,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{model} {cond} win{w} {y}-{q} failed:\n{proc.stderr[-2000:]}")
        log(f"  [{i:3d}/{len(cells)}] {model} {cond} win{w:<2} {y}-{q}")
    rows = []
    for f in sorted(results_dir.rglob("*.csv")):
        rec = parse_name(f.name)
        if not rec:
            continue
        cond = next(c for c in CONTROL_CONDITIONS if f"_dDEC_{c}" in rec["model_id"])
        pred = pd.read_csv(f).merge(dec, on="id", how="left")
        w = rec["window"]
        rec.update(
            model=model,
            condition=cond,
            n_test=int(len(pred)),
            persistence_mse=float(np.mean((pred["trues"] - pred[f"lv{w}_past_1"]) ** 2)),
        )
        rows.append(rec)
    table = pd.DataFrame(rows).sort_values(["condition", "window", "year", "quarter"])
    out = root / "results" / f"result_table_6r_scss_{model.lower()}_controls.csv"
    table.reset_index(drop=True).to_csv(out, index=False, lineterminator="\n")
    return table
