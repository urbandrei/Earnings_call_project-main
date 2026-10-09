"""ecvol-bench quickstart: from a release archive to a baseline, a prediction file and the
identity checks in one command.

    uv run python examples/quickstart.py --archive data/release/ecvol-bench_fincall_<sha>.zip

What it does, for the trading-day targets in the archive:
  1. loads targets, the temporal and ticker-disjoint splits and the pooled BGE-M3 call
     embeddings (scope "full") straight from the zip;
  2. fits a ridge on the past-volatility block alone and on past volatility plus the text
     embedding, per split and horizon (penalty chosen on the validation segment);
  3. prints R^2_OOS against persistence on the test segment;
  4. writes one prediction frame per variant in the format `ecvol audit predictions` reads
     (call_id, ticker, date, split, horizon, y_true, y_pred, y_persistence) and runs the
     audit on it, so the same identity checks the paper uses are applied to this baseline.

Replace step 2 with your own model and keep steps 3-4: that is the comparison the paper
asks for. Only pandas, numpy and the installed `ecvol` package are needed.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval.audit_predictions import run_audit
from ecvol.models import heads

PASTVOL = ["v_pre", "rv_daily", "rv_weekly", "rv_monthly"]
SPLITS = ("temporal", "ticker_disjoint")
VARIANTS = ("pastvol", "text_pastvol")


def _corpus(z: zipfile.ZipFile) -> str:
    names = [n for n in z.namelist() if n.endswith("/targets.parquet")]
    if len(names) != 1:
        raise SystemExit(f"expected one <corpus>/targets.parquet in the archive, found {names}")
    return names[0].split("/")[0]


def _read(z: zipfile.ZipFile, member: str) -> pd.DataFrame:
    data = io.BytesIO(z.read(member))
    return pd.read_parquet(data) if member.endswith(".parquet") else pd.read_csv(data)


def load_archive(path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """(corpus, targets, embeddings[call_id, e0..e{d-1}], {split_name: split frame})."""
    with zipfile.ZipFile(path) as z:
        corpus = _corpus(z)
        targets = _read(z, f"{corpus}/targets.parquet")
        emb = _read(z, f"{corpus}/text_embeddings.parquet")
        emb = emb[emb["scope"] == "full"]
        vec = np.vstack(emb["vector"].to_numpy())
        embeddings = pd.DataFrame(vec, columns=[f"e{i}" for i in range(vec.shape[1])])
        embeddings.insert(0, "call_id", emb["call_id"].to_numpy())
        splits = {s: _read(z, f"splits/{corpus}_{s}.csv") for s in SPLITS}
    targets = targets[targets["status"] == "ok"]
    return corpus, targets, embeddings, splits


def _r2_oos(y: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - ref) ** 2))


def run(archive: Path, out_dir: Path, target: str = "delta_v") -> dict[str, Path]:
    corpus, targets, embeddings, splits = load_archive(archive)
    emb_cols = [c for c in embeddings.columns if c != "call_id"]
    frames: dict[str, list[pd.DataFrame]] = {v: [] for v in VARIANTS}
    print(f"{corpus}: {targets['call_id'].nunique()} calls with targets, target={target}")
    print(f"{'split':16} {'variant':13} " + " ".join(f"tau={h:<3}" for h in (3, 7, 15, 30)))
    for split_name, split in splits.items():
        split = split[split["split"].isin(["train", "val", "test"])][["call_id", "split"]]
        scores: dict[str, list[str]] = {v: [] for v in VARIANTS}
        for h in sorted(targets["horizon"].unique()):
            at = targets[targets["horizon"] == h].merge(split, on="call_id")
            at = at.merge(embeddings, on="call_id", how="left")
            at = at.dropna(subset=[target])
            tr = (at["split"] == "train").to_numpy()
            va = (at["split"] == "val").to_numpy()
            te = (at["split"] == "test").to_numpy()
            y = at[target].to_numpy(dtype=float)
            persist = np.zeros(len(at)) if target == "delta_v" else at["v_pre"].to_numpy(float)
            for variant in VARIANTS:
                cols = PASTVOL if variant == "pastvol" else PASTVOL + emb_cols
                X = at[cols].to_numpy(dtype=np.float64).copy()
                med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
                nan = np.isnan(X)
                if nan.any():
                    X[nan] = np.take(med, np.where(nan)[1])
                pv, pte, _ = heads.ridge_fit_predict(X[tr], y[tr], X[va], y[va], X[te])
                pred = np.full(len(at), np.nan)
                pred[va], pred[te] = pv, pte
                scores[variant].append(f"{_r2_oos(y[te], pte, persist[te]):+.3f}")
                frames[variant].append(
                    pd.DataFrame(
                        {
                            "call_id": at["call_id"].to_numpy(),
                            "ticker": at["ticker"].to_numpy(),
                            "date": pd.to_datetime(at["as_of"]).dt.strftime("%Y-%m-%d").to_numpy(),
                            "split": at["split"].to_numpy(),
                            "horizon": int(h),
                            "y_true": y,
                            "y_pred": pred,
                            "y_persistence": persist,
                            "split_family": split_name,
                        }
                    )
                )
        for variant in VARIANTS:
            print(f"{split_name:16} {variant:13} " + " ".join(f"{s:>7}" for s in scores[variant]))
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for variant in VARIANTS:
        df = pd.concat(frames[variant], ignore_index=True)
        for split_name in SPLITS:
            frame = df[df["split_family"] == split_name].drop(columns="split_family")
            path = out_dir / f"{corpus}_{split_name}_{variant}_{target}.csv"
            frame.to_csv(path, index=False, lineterminator="\n")
            report = run_audit(path, out_dir / path.stem, name=path.stem)
            written[path.stem] = report
            print(f"wrote {path.name}; audit report in {report}")
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--archive", required=True, type=Path, help="ecvol-bench_<corpus>_<sha>.zip")
    p.add_argument("--target", default="delta_v", choices=["delta_v", "v_post"])
    p.add_argument("--out", type=Path, default=Path("examples") / "out")
    a = p.parse_args(argv)
    run(a.archive, a.out, a.target)


if __name__ == "__main__":
    main(sys.argv[1:])
