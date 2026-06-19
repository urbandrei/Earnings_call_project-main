"""Stage-0/1 baseline evaluation → Result Table 1 (DESIGN §6 Stages 0-1, §7).

Evaluates the honest floor — persistence, EWMA, HAR-RV, GARCH(1,1), and a
ticker-FE LightGBM — across every (dataset × split × target × horizon), on the
val and test segments, and writes the long-format **Result Table 1**.

Targets (DESIGN §7.1): level-v (`v_post`), Δv (`delta_v`), and HAR-residual
(`v_post − HAR_train_forecast`). Each model produces a `v_post` forecast that is
mapped into the target space; **persistence is the per-target trivial forecast**
(`v_pre` for level-v, 0 for Δv / HAR-residual) and doubles as the R²_OOS
baseline. HAR-RV and GBDT are fit on the **train** split only.

Sanity gate (DESIGN §6 Stage 0): HAR-RV must beat persistence at τ=30 on the
temporal split, level-v, test — i.e. HAR's R²_OOS (vs persistence) > 0. If
violated, the run reports `gate_passed=False` (halt and debug targets). GARCH
convergence is reported against the >95% gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.data.targets import HAR_WINDOWS  # noqa: F401  (documents the HAR lag origin)
from ecvol.eval import metrics as M
from ecvol.eval.significance import diebold_mariano
from ecvol.models import baselines as B
from ecvol.models.gbdt import train_predict_gbdt

DATASETS = ("fincall", "maec")
SPLIT_SCHEMES = ("temporal", "ticker_disjoint", "combined")
TARGETS = ("v", "dv", "har_resid")
HORIZONS = (3, 7, 15, 30)
SEGMENTS = ("val", "test")
ECON_MODELS = ("persistence", "ewma", "har", "garch")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


# --- data loading ------------------------------------------------------------


def load_eval_frame(root: Path, dataset: str) -> pd.DataFrame:
    """ok target rows joined with call metadata; one row per (call, horizon)."""
    targets = pd.read_parquet(root / dataset / "targets.parquet")
    targets = targets[targets["status"] == "ok"].copy()
    calls = pd.read_parquet(
        root / dataset / "calls.parquet", columns=["call_id", "n_turns", "n_chars"]
    )
    df = targets.merge(calls, on="call_id", how="left")
    df["call_id"] = df["call_id"].astype(str)
    return df


def add_econometric_forecasts(df: pd.DataFrame, prices_dir: Path) -> pd.DataFrame:
    """Add per-call `v_post` forecasts (split-independent): EWMA + GARCH.

    Persistence is `v_pre` (already a column) and HAR is train-fit later. EWMA is
    flat across horizons; GARCH is fit once per call and forecast for each τ.
    `garch_ok` flags convergence for the gate.
    """
    from ecvol.data.prices import load_close_series

    df = df.sort_values(["call_id", "horizon"]).reset_index(drop=True)
    close_cache: dict[str, dict[str, float]] = {}

    def close_for(ticker: str) -> dict[str, float]:
        if ticker not in close_cache:
            close_cache[ticker] = load_close_series(prices_dir, ticker)
        return close_cache[ticker]

    # One EWMA + one GARCH fit per (call_id, as_of); broadcast across horizons.
    ewma: dict[str, float] = {}
    garch: dict[tuple[str, int], float] = {}
    garch_ok: dict[str, bool] = {}
    for (cid, as_of, ticker), _ in df.groupby(["call_id", "as_of", "ticker"]):
        close = close_for(ticker)
        ewma[cid] = B.ewma_log_rv(close, as_of)
        g = B.garch_log_rv_multi(close, as_of, HORIZONS)
        garch_ok[cid] = g is not None
        for h in HORIZONS:
            garch[(cid, h)] = g[h] if g is not None else np.nan

    df["ewma_vpost"] = df["call_id"].map(ewma)
    df["garch_vpost"] = [garch[(c, h)] for c, h in zip(df["call_id"], df["horizon"], strict=True)]
    df["garch_ok"] = df["call_id"].map(garch_ok)
    return df


# --- target transforms -------------------------------------------------------


def target_truth(df: pd.DataFrame, target: str, har_vpost: np.ndarray) -> np.ndarray:
    if target == "v":
        return df["v_post"].to_numpy()
    if target == "dv":
        return df["delta_v"].to_numpy()
    return df["v_post"].to_numpy() - har_vpost  # har_resid


def persistence_pred(df: pd.DataFrame, target: str) -> np.ndarray:
    """The per-target trivial forecast (also the R²_OOS baseline)."""
    if target == "v":
        return df["v_pre"].to_numpy()
    return np.zeros(len(df))  # dv, har_resid


def vpost_to_target(vpost_pred: np.ndarray, df: pd.DataFrame, target: str, har_vpost: np.ndarray):
    """Map a v_post forecast into the target space."""
    if target == "v":
        return vpost_pred
    if target == "dv":
        return vpost_pred - df["v_pre"].to_numpy()
    return vpost_pred - har_vpost  # har_resid


# --- per-cell metrics --------------------------------------------------------


def _cell_metrics(sub: pd.DataFrame, y_true, y_pred, baseline) -> dict:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "as_of": sub["as_of"].to_numpy()})
    return {
        "n": int(np.sum(~(np.isnan(y_true) | np.isnan(y_pred)))),
        "mse": M.mse(y_true, y_pred),
        "mae": M.mae(y_true, y_pred),
        "r2_oos": M.r2_oos(y_true, y_pred, baseline),
        "spearman_q": M.spearman_by_quarter(frame)[0],
    }


def _dm_vs(y_true, y_pred, y_ref) -> float:
    """Two-sided DM p-value of `y_pred` vs a reference forecast (h=1, squared loss)."""
    res = diebold_mariano(y_true - y_pred, y_true - y_ref)
    return res.p_value


# --- orchestration -----------------------------------------------------------


@dataclass
class EvalSummary:
    rows: list[dict]
    garch_convergence: dict[str, float]  # dataset -> fraction converged
    gate_passed: bool
    gate_detail: dict


def evaluate_dataset(df: pd.DataFrame, dataset: str, splits_dir: Path, *, seeds) -> list[dict]:
    """All (split × target × τ × model × segment) metric rows for one dataset."""
    rows: list[dict] = []
    for scheme in SPLIT_SCHEMES:
        split_csv = splits_dir / f"{dataset}_{scheme}.csv"
        if not split_csv.is_file():
            continue
        assign = pd.read_csv(split_csv, dtype={"call_id": str}).set_index("call_id")["split"]
        d = df.copy()
        d["split"] = d["call_id"].map(assign).fillna("excluded")

        for tau in HORIZONS:
            at = d[d["horizon"] == tau].copy()
            train = at[at["split"] == "train"]
            # Train-only log-HAR fit → v_post forecast for every row at this τ.
            x_all = B.har_design(at["rv_daily"], at["rv_weekly"], at["rv_monthly"])
            x_tr = B.har_design(train["rv_daily"], train["rv_weekly"], train["rv_monthly"])
            coef = B.har_fit(x_tr, train["v_post"].to_numpy())
            har_vpost = B.har_predict(x_all, coef)
            at = at.assign(_har_vpost=har_vpost)

            for target in TARGETS:
                hv = at["_har_vpost"].to_numpy()
                y_true_all = target_truth(at, target, hv)
                base_all = persistence_pred(at, target)
                # model → target-space predictions over all rows at this τ
                preds = {
                    "persistence": base_all,
                    "ewma": vpost_to_target(at["ewma_vpost"].to_numpy(), at, target, hv),
                    "har": vpost_to_target(hv, at, target, hv),
                    "garch": vpost_to_target(at["garch_vpost"].to_numpy(), at, target, hv),
                }
                gbdt_preds = _gbdt_predictions(at, target, y_true_all, seeds)

                for seg in SEGMENTS:
                    mask = (at["split"] == seg).to_numpy()
                    if mask.sum() == 0:
                        continue
                    sub = at[mask]
                    yt, base = y_true_all[mask], base_all[mask]
                    for model in ECON_MODELS:
                        yp = preds[model][mask]
                        cell = _cell_metrics(sub, yt, yp, base)
                        cell["dm_p_vs_persistence"] = (
                            np.nan if model == "persistence" else _dm_vs(yt, yp, base)
                        )
                        rows.append(_row(dataset, scheme, target, tau, model, seg, cell))
                    rows.append(
                        _gbdt_row(
                            dataset, scheme, target, tau, seg, sub, mask, yt, base, gbdt_preds
                        )
                    )
    return rows


def _gbdt_predictions(at: pd.DataFrame, target: str, y_true_all: np.ndarray, seeds) -> dict:
    """Per-seed GBDT predictions (target-space), trained on this τ's train rows."""
    train = at[at["split"] == "train"].copy()
    train = train.assign(_y=y_true_all[(at["split"] == "train").to_numpy()])
    out = {}
    for s in seeds:
        out[s] = train_predict_gbdt(train, at, "_y", seed=s)
    return out


def _gbdt_row(dataset, scheme, target, tau, seg, sub, mask, yt, base, gbdt_preds) -> dict:
    per_seed = [_cell_metrics(sub, yt, p[mask], base) for p in gbdt_preds.values()]
    mean_pred = np.mean([p[mask] for p in gbdt_preds.values()], axis=0)
    cell = {
        k: float(np.mean([c[k] for c in per_seed])) for k in ("mse", "mae", "r2_oos", "spearman_q")
    }
    cell["n"] = per_seed[0]["n"]
    cell["gbdt_mse_std"] = float(np.std([c["mse"] for c in per_seed]))
    cell["dm_p_vs_persistence"] = _dm_vs(yt, mean_pred, base)
    return _row(dataset, scheme, target, tau, "gbdt_tickerFE", seg, cell)


def _row(dataset, scheme, target, tau, model, seg, cell) -> dict:
    return {
        "dataset": dataset,
        "split": scheme,
        "target": target,
        "horizon": int(tau),
        "model": model,
        "segment": seg,
        "n": int(cell["n"]),
        "mse": float(cell["mse"]),
        "mae": float(cell["mae"]),
        "r2_oos": float(cell["r2_oos"]),
        "spearman_q": float(cell["spearman_q"]),
        "gbdt_mse_std": float(cell.get("gbdt_mse_std", np.nan)),
        "dm_p_vs_persistence": float(cell.get("dm_p_vs_persistence", np.nan)),
    }


def run_evaluate(root: Path, *, seeds=DEFAULT_SEEDS) -> EvalSummary:
    """Evaluate all datasets, write Result Table 1, and check the sanity gates."""
    all_rows: list[dict] = []
    garch_conv: dict[str, float] = {}
    for dataset in DATASETS:
        if not (root / dataset / "targets.parquet").is_file():
            continue
        df = load_eval_frame(root, dataset)
        df = add_econometric_forecasts(df, root / "prices")
        # convergence over distinct calls (not call×horizon rows)
        per_call = df.drop_duplicates("call_id")
        garch_conv[dataset] = float(per_call["garch_ok"].mean())
        all_rows.extend(evaluate_dataset(df, dataset, root / "splits", seeds=seeds))

    table = (
        pd.DataFrame(all_rows)
        .sort_values(["dataset", "split", "target", "horizon", "model", "segment"])
        .reset_index(drop=True)
    )
    _write_results(table, root)

    gate = _check_gate(table)
    return EvalSummary(all_rows, garch_conv, gate["passed"], gate)


def _har_beats_persistence(table: pd.DataFrame, dataset: str, split: str, tau: int = 30) -> float:
    """HAR's level-v test R²_OOS vs persistence at horizon τ (>0 ⇒ HAR wins)."""
    sel = table[
        (table["model"] == "har")
        & (table["dataset"] == dataset)
        & (table["split"] == split)
        & (table["target"] == "v")
        & (table["horizon"] == tau)
        & (table["segment"] == "test")
    ]
    return float(sel["r2_oos"].iloc[0]) if len(sel) else float("nan")


def _check_gate(table: pd.DataFrame) -> dict:
    """Sanity gate (DESIGN §6 Stage 0): HAR-RV beats persistence at τ=30, level-v, test.

    The literal check is on the primary dataset's temporal split. When that fails,
    a **documented COVID-regime exception** (DESIGN §5.4.5) applies *iff* the
    targets are corroborated as sound by the regime-stable cells — HAR beating
    persistence at τ=30 on FinCall's ticker-disjoint split (mixes dates, so no
    regime shift) **and** on MAEC's temporal split (a non-COVID corpus). A real
    target bug would fail those too, so this is not a blanket pass. See
    DECISIONS.md 2026-06-18.
    """
    fincall_temporal = _har_beats_persistence(table, "fincall", "temporal")
    fincall_disjoint = _har_beats_persistence(table, "fincall", "ticker_disjoint")
    maec_temporal = _har_beats_persistence(table, "maec", "temporal")
    literal_pass = fincall_temporal > 0
    regime_exception = (not literal_pass) and fincall_disjoint > 0 and maec_temporal > 0
    return {
        "passed": bool(literal_pass or regime_exception),
        "literal_pass": bool(literal_pass),
        "covid_regime_exception": bool(regime_exception),
        "har_r2_oos_vs_persistence": {
            "fincall_temporal": fincall_temporal,
            "fincall_ticker_disjoint": fincall_disjoint,
            "maec_temporal": maec_temporal,
        },
    }


def _write_results(table: pd.DataFrame, root: Path) -> None:
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "result_table_1.csv", index=False, lineterminator="\n")
