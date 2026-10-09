"""ecvol-bench quickstart.

    uv run python examples/quickstart.py --archive data/release/ecvol-bench_fincall_<sha>.zip

Loads a release archive, fits a ridge baseline (past volatility, then past volatility plus
the call's text embedding), prints R^2 against persistence, writes a prediction file per
model and split, and runs the identity checks on each file.

To test your own model, replace `fit_and_predict` and keep everything else.
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ecvol.eval.audit_predictions import run_audit

PAST_VOL = ["v_pre", "rv_daily", "rv_weekly", "rv_monthly"]
HORIZONS = [3, 7, 15, 30]
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]  # the grid the paper uses


def load_archive(path):
    """Return (corpus name, targets, embeddings, {split name: split table})."""
    with zipfile.ZipFile(path) as z:
        corpus = [n for n in z.namelist() if n.endswith("/targets.parquet")][0].split("/")[0]
        targets = pd.read_parquet(io.BytesIO(z.read(f"{corpus}/targets.parquet")))
        emb = pd.read_parquet(io.BytesIO(z.read(f"{corpus}/text_embeddings.parquet")))
        splits = {
            name: pd.read_csv(io.BytesIO(z.read(f"splits/{corpus}_{name}.csv")))
            for name in ["temporal", "ticker_disjoint"]
        }

    targets = targets[targets["status"] == "ok"]

    # one 1024-d vector per call (the "full" transcript), as columns e0..e1023
    emb = emb[emb["scope"] == "full"]
    embeddings = pd.DataFrame(np.vstack(emb["vector"]), index=emb["call_id"].values)
    embeddings.columns = [f"e{i}" for i in range(embeddings.shape[1])]
    embeddings = embeddings.reset_index(names="call_id")

    return corpus, targets, embeddings, splits


def fit_and_predict(X_train, y_train, X_val, y_val, X_test):
    """Ridge with the penalty chosen on the validation set. Replace this with your model."""
    scaler = StandardScaler().fit(X_train)
    best_model, best_mse = None, np.inf
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train)
        mse = np.mean((y_val - model.predict(scaler.transform(X_val))) ** 2)
        if mse < best_mse:
            best_model, best_mse = model, mse
    return best_model.predict(scaler.transform(X_val)), best_model.predict(scaler.transform(X_test))


def r2_vs_persistence(y, pred, persistence):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - persistence) ** 2)


def run_one(table, feature_cols, target):
    """Fit on train, choose on val, predict val and test. Returns (predictions, test R^2)."""
    train = (table["split"] == "train").values
    val = (table["split"] == "val").values
    test = (table["split"] == "test").values

    X = table[feature_cols].to_numpy(dtype=float)
    X = np.where(np.isnan(X), np.nanmedian(X[train], axis=0), X)  # fill gaps with train medians
    y = table[target].to_numpy(dtype=float)

    pred_val, pred_test = fit_and_predict(X[train], y[train], X[val], y[val], X[test])
    pred = np.full(len(table), np.nan)
    pred[val], pred[test] = pred_val, pred_test

    persistence = np.zeros(len(table)) if target == "delta_v" else table["v_pre"].values
    return pred, r2_vs_persistence(y[test], pred[test], persistence[test])


def prediction_file(table, pred, target, horizon):
    """One row per call in the format `ecvol audit predictions` reads."""
    persistence = np.zeros(len(table)) if target == "delta_v" else table["v_pre"].values
    return pd.DataFrame(
        {
            "call_id": table["call_id"].values,
            "ticker": table["ticker"].values,
            "date": pd.to_datetime(table["as_of"]).dt.strftime("%Y-%m-%d").values,
            "split": table["split"].values,
            "horizon": horizon,
            "y_true": table[target].values,
            "y_pred": pred,
            "y_persistence": persistence,
        }
    )


def run(archive, out_dir, target="delta_v"):
    corpus, targets, embeddings, splits = load_archive(archive)
    models = {
        "pastvol": PAST_VOL,
        "text_pastvol": PAST_VOL + [c for c in embeddings.columns if c != "call_id"],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}

    print(
        f"{corpus}: {targets['call_id'].nunique()} calls, target {target}, "
        "R^2 vs persistence on test"
    )
    print(f"{'split':16}{'model':14}" + "".join(f"tau={h:<6}" for h in HORIZONS))

    for split_name, split in splits.items():
        split = split[split["split"].isin(["train", "val", "test"])][["call_id", "split"]]
        for model_name, feature_cols in models.items():
            rows, scores = [], []
            for horizon in HORIZONS:
                table = targets[targets["horizon"] == horizon].merge(split, on="call_id")
                table = table.merge(embeddings, on="call_id", how="left").dropna(subset=[target])
                pred, r2 = run_one(table, feature_cols, target)
                rows.append(prediction_file(table, pred, target, horizon))
                scores.append(r2)
            print(f"{split_name:16}{model_name:14}" + "".join(f"{s:+.3f}    " for s in scores))

            name = f"{corpus}_{split_name}_{model_name}_{target}"
            path = out_dir / f"{name}.csv"
            pd.concat(rows).to_csv(path, index=False)
            reports[name] = run_audit(path, out_dir / name, name=name)

    print(f"\nprediction files and audit reports are in {out_dir}")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--archive", required=True, help="ecvol-bench_<corpus>_<sha>.zip")
    parser.add_argument("--target", default="delta_v", choices=["delta_v", "v_post"])
    parser.add_argument("--out", default="examples/out")
    args = parser.parse_args()
    run(args.archive, args.out, args.target)
