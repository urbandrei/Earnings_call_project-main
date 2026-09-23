"""`ecvol audit predictions` — the identity-control suite for *anyone's* predictions.

A model that beats the volatility baselines on a shared-company split may be reading
which company is speaking rather than what was said (DESIGN §7.3; the paper's §6). This
command applies the controls that can be computed from a prediction frame alone — no
model, no features, no retraining — so another paper's numbers can be audited from the
file its authors would have to produce anyway.

Input: a long CSV/parquet with one row per (call, horizon):

    call_id, ticker, date (YYYY-MM-DD, the call's as-of date), split (train|val|test),
    horizon, y_true, y_pred  [+ optional y_persistence: the past-volatility forecast]

Train (and val) rows need `y_true`; `y_pred` is only read on test rows. Per horizon:

1. **split_integrity** — share of test calls whose ticker is in train, the gap in business
   days between the last train call and the first test call, and whether any train target
   window (horizon sessions) can reach the test period.
2. **floors** — MSE of the model against three training-free references: the persistence
   forecast (if given), the *ticker-only* model (per-ticker train mean; §7.3 control 1)
   and the train mean; R²_OOS against each.
3. **seen_vs_unseen** — MSE on test calls whose ticker was in train vs those whose was not.
4. **prediction_shuffle** — each test prediction replaced by that of another test call of
   the *same* ticker (within) or of a *different* ticker (global), MSE averaged over
   permutations. A within-ticker swap that leaves MSE unchanged means the predictions are
   a function of the company, not the call (the prediction-space form of §7.3 control 2).
5. **identity_share** — the fraction of test-prediction variance explained by ticker
   identity (between-ticker / total), and the same for the target, for comparison.
6. **significance** — Diebold–Mariano p-values vs the floors and a ticker-clustered
   bootstrap CI on the model − ticker-only squared-error difference.

Output: `report.csv` (long: horizon, check, metric, value, n, note) and `report.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import metrics as M
from ecvol.eval.significance import cluster_bootstrap_ci, diebold_mariano
from ecvol.models.ticker_only import ticker_mean_fit_predict

REQUIRED = ("call_id", "ticker", "date", "split", "horizon", "y_true", "y_pred")
OPTIONAL_BASELINE = M.COL_BASELINE  # y_persistence
FIT_SPLITS = ("train", "val")
N_PERMUTATIONS = 50


class FrameError(ValueError):
    """The prediction frame does not satisfy the input contract."""


def load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise FrameError(f"missing columns: {missing}; required: {list(REQUIRED)}")
    df = df.copy()
    df["call_id"] = df["call_id"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df["split"] = df["split"].astype(str).str.lower()
    df["horizon"] = df["horizon"].astype(int)
    for c in ("y_true", "y_pred", *([OPTIONAL_BASELINE] if OPTIONAL_BASELINE in df else [])):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise FrameError(f"`date` must be YYYY-MM-DD: {exc}") from exc
    dup = df.duplicated(["call_id", "horizon"])
    if dup.any():
        raise FrameError(f"{int(dup.sum())} duplicate (call_id, horizon) rows")
    unknown = sorted(set(df["split"]) - {*FIT_SPLITS, "test"})
    if unknown:
        raise FrameError(f"unknown split labels {unknown}; use train, val, test")
    for h, g in df.groupby("horizon"):
        te = g[g["split"] == "test"]
        if te.empty or te["y_pred"].isna().all():
            raise FrameError(f"horizon {h}: no test rows with y_pred")
        if g[g["split"].isin(FIT_SPLITS)]["y_true"].notna().sum() == 0:
            raise FrameError(f"horizon {h}: no train/val rows with y_true")
    return df


# --- checks (each returns long rows) ---------------------------------------------


def _row(h, check, metric, value, n=None, note=""):
    return {
        "horizon": int(h),
        "check": check,
        "metric": metric,
        "value": float(value) if value is not None and np.isfinite(value) else np.nan,
        "n": int(n) if n is not None else np.nan,
        "note": note,
    }


def split_integrity(h, fit: pd.DataFrame, te: pd.DataFrame) -> list[dict]:
    train = fit[fit["split"] == "train"]
    seen = te["ticker"].isin(set(fit["ticker"]))
    last_fit = np.datetime64(fit["date"].max(), "D")
    first_te = np.datetime64(te["date"].min(), "D")
    gap = int(np.busday_count(last_fit, first_te))  # negative if interleaved
    ends = np.busday_offset(fit["date"].to_numpy(dtype="datetime64[D]"), h, roll="forward")
    crossing = int((ends >= first_te).sum())
    return [
        _row(h, "split_integrity", "n_train", len(train)),
        _row(h, "split_integrity", "n_val", int((fit["split"] == "val").sum())),
        _row(h, "split_integrity", "n_test", len(te)),
        _row(h, "split_integrity", "test_calls_ticker_in_fit_pct", 100 * seen.mean(), len(te)),
        _row(
            h,
            "split_integrity",
            "test_tickers_in_fit_pct",
            100 * te["ticker"].drop_duplicates().isin(set(fit["ticker"])).mean(),
            te["ticker"].nunique(),
        ),
        _row(
            h,
            "split_integrity",
            "gap_business_days",
            gap,
            note="last train/val call → first test call; negative = interleaved",
        ),
        _row(
            h,
            "split_integrity",
            "fit_windows_reaching_test",
            crossing,
            len(fit),
            "train/val calls whose horizon-session window reaches the first test call",
        ),
    ]


def floors(h, fit, te) -> tuple[list[dict], dict[str, np.ndarray]]:
    ok = fit["y_true"].notna()
    y, p = te["y_true"].to_numpy(float), te["y_pred"].to_numpy(float)
    refs = {
        "ticker_only": ticker_mean_fit_predict(
            fit.loc[ok, "ticker"].to_numpy(), fit.loc[ok, "y_true"].to_numpy(), te["ticker"]
        ),
        "train_mean": np.full(len(te), float(fit.loc[ok, "y_true"].mean())),
    }
    if OPTIONAL_BASELINE in te.columns and te[OPTIONAL_BASELINE].notna().any():
        refs = {"persistence": te[OPTIONAL_BASELINE].to_numpy(float), **refs}
    rows = [_row(h, "floors", "mse_model", M.mse(y, p), len(te))]
    for name, r in refs.items():
        rows.append(_row(h, "floors", f"mse_{name}", M.mse(y, r), len(te)))
        rows.append(_row(h, "floors", f"r2_oos_vs_{name}", M.r2_oos(y, p, r), len(te)))
    return rows, refs


def seen_vs_unseen(h, fit, te) -> list[dict]:
    seen = te["ticker"].isin(set(fit["ticker"])).to_numpy()
    y, p = te["y_true"].to_numpy(float), te["y_pred"].to_numpy(float)
    out = []
    for name, m in (("seen_ticker", seen), ("unseen_ticker", ~seen)):
        out.append(
            _row(
                h,
                "seen_vs_unseen",
                f"mse_{name}",
                M.mse(y[m], p[m]) if m.any() else np.nan,
                m.sum(),
            )
        )
    return out


def shuffle_predictions(pred: np.ndarray, tickers: np.ndarray, mode: str, rng) -> np.ndarray:
    """Permute predictions among test rows: within each ticker (rows of singleton tickers
    keep their own), or across rows of *different* tickers (derangement by ticker)."""
    out = pred.copy()
    if mode == "within":
        for t in np.unique(tickers):
            idx = np.where(tickers == t)[0]
            if len(idx) > 1:
                out[idx] = pred[idx[rng.permutation(len(idx))]]
        return out
    n = len(pred)
    perm = rng.permutation(n)
    for _ in range(200):  # repair rows that drew their own ticker by swapping with a random row
        same = np.where(tickers[perm] == tickers)[0]
        if len(same) == 0:
            break
        for i in same:
            j = int(rng.integers(n))
            perm[i], perm[j] = perm[j], perm[i]
    return pred[perm]


def prediction_shuffle(h, te, *, n_perm=N_PERMUTATIONS, seed=0) -> list[dict]:
    tickers = te["ticker"].to_numpy()
    counts = pd.Series(tickers).value_counts()
    m = pd.Series(tickers).map(counts).to_numpy() > 1  # rows a within-swap can move
    y, p = te["y_true"].to_numpy(float), te["y_pred"].to_numpy(float)
    if m.sum() < 2:
        return [
            _row(
                h, "prediction_shuffle", "n_shufflable", m.sum(), note="no ticker has ≥2 test calls"
            )
        ]
    rng = np.random.default_rng(seed)
    within = np.mean(
        [M.mse(y[m], shuffle_predictions(p, tickers, "within", rng)[m]) for _ in range(n_perm)]
    )
    glob = np.mean(
        [M.mse(y[m], shuffle_predictions(p, tickers, "global", rng)[m]) for _ in range(n_perm)]
    )
    real = M.mse(y[m], p[m])
    return [
        _row(h, "prediction_shuffle", "n_shufflable", m.sum(), len(te)),
        _row(h, "prediction_shuffle", "mse_real", real, m.sum()),
        _row(
            h, "prediction_shuffle", "mse_within_ticker_swap", within, m.sum(), f"mean of {n_perm}"
        ),
        _row(h, "prediction_shuffle", "mse_other_ticker_swap", glob, m.sum(), f"mean of {n_perm}"),
        _row(h, "prediction_shuffle", "within_swap_change_pct", 100 * (within / real - 1)),
        _row(h, "prediction_shuffle", "other_swap_change_pct", 100 * (glob / real - 1)),
    ]


def identity_share(h, te) -> list[dict]:
    counts = te["ticker"].map(te["ticker"].value_counts())
    g = te[counts > 1]
    out = []
    for col in ("y_pred", "y_true"):
        v = g[col].to_numpy(float)
        if len(v) < 2:
            out.append(_row(h, "identity_share", f"ticker_variance_share_{col}", np.nan, 0))
            continue
        total = float(np.var(v))
        between = float(np.var(g.groupby("ticker")[col].transform("mean").to_numpy(float)))
        share = between / total if total > 0 else np.nan
        out.append(
            _row(
                h,
                "identity_share",
                f"ticker_variance_share_{col}",
                share,
                len(g),
                "tickers with ≥2 test calls",
            )
        )
    return out


def significance(h, te, refs: dict[str, np.ndarray]) -> list[dict]:
    y, p = te["y_true"].to_numpy(float), te["y_pred"].to_numpy(float)
    order = np.argsort(te["date"].to_numpy(), kind="stable")
    rows = []
    for name, r in refs.items():
        dm = diebold_mariano((y - p)[order], (y - r)[order], h=1)
        rows.append(
            _row(h, "significance", f"dm_p_vs_{name}", dm.p_value, dm.n, "+ve stat ⇒ model worse")
        )
        rows.append(_row(h, "significance", f"dm_stat_vs_{name}", dm.statistic, dm.n))
    d = (y - p) ** 2 - (y - refs["ticker_only"]) ** 2
    point, lo, hi = cluster_bootstrap_ci(d, te["ticker"].to_numpy(), seed=0)
    rows.append(
        _row(
            h,
            "significance",
            "sqerr_diff_vs_ticker_only_mean",
            point,
            len(te),
            "model − ticker-only; <0 = model better",
        )
    )
    rows.append(
        _row(
            h,
            "significance",
            "sqerr_diff_vs_ticker_only_ci_lo",
            lo,
            len(te),
            "ticker-clustered 95% bootstrap",
        )
    )
    rows.append(_row(h, "significance", "sqerr_diff_vs_ticker_only_ci_hi", hi, len(te)))
    return rows


def audit(df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    rows = []
    for h, g in df.groupby("horizon"):
        fit = g[g["split"].isin(FIT_SPLITS)]
        te = g[(g["split"] == "test") & g["y_pred"].notna() & g["y_true"].notna()]
        rows += split_integrity(h, fit, te)
        fl, refs = floors(h, fit, te)
        rows += fl + seen_vs_unseen(h, fit, te)
        rows += prediction_shuffle(h, te, seed=seed) + identity_share(h, te)
        rows += significance(h, te, refs)
    return pd.DataFrame(rows)


# --- report ---------------------------------------------------------------------


def _get(rep, h, metric):
    s = rep[(rep["horizon"] == h) & (rep["metric"] == metric)]["value"]
    return float(s.iloc[0]) if len(s) else np.nan


def render_markdown(rep: pd.DataFrame, name: str) -> str:
    hs = sorted(rep["horizon"].unique())
    lines = [
        f"# Prediction audit — {name}",
        "",
        "Verdict rules are mechanical and stated inline; read every line against its n.",
        "",
    ]
    hdr = "| horizon | " + " | ".join(str(h) for h in hs) + " |"
    sep = "|---|" + "---|" * len(hs)

    def line(label, metric, fmt="{:.3f}"):
        vals = [_get(rep, h, metric) for h in hs]
        return (
            f"| {label} | " + " | ".join("—" if np.isnan(v) else fmt.format(v) for v in vals) + " |"
        )

    lines += [
        "## Split",
        hdr,
        sep,
        line(
            "test calls whose ticker is in train/val (%)", "test_calls_ticker_in_fit_pct", "{:.1f}"
        ),
        line(
            "gap, last train/val → first test call (business days)", "gap_business_days", "{:.0f}"
        ),
        line(
            "train/val target windows reaching the test period",
            "fit_windows_reaching_test",
            "{:.0f}",
        ),
        "",
    ]
    lines += ["## Floors (test MSE)", hdr, sep, line("model", "mse_model")]
    for k, lab in (
        ("persistence", "persistence (supplied)"),
        ("ticker_only", "ticker-only (per-ticker train mean)"),
        ("train_mean", "train mean"),
    ):
        if rep["metric"].eq(f"mse_{k}").any():
            lines.append(line(lab, f"mse_{k}"))
            lines.append(line(f"  R²_OOS vs {k}", f"r2_oos_vs_{k}"))
            lines.append(line(f"  DM p vs {k}", f"dm_p_vs_{k}"))
    lines += [
        "",
        "## Identity",
        hdr,
        sep,
        line("MSE, seen tickers", "mse_seen_ticker"),
        line("MSE, unseen tickers", "mse_unseen_ticker"),
        line("within-ticker prediction swap, ΔMSE (%)", "within_swap_change_pct", "{:+.1f}"),
        line("other-ticker prediction swap, ΔMSE (%)", "other_swap_change_pct", "{:+.1f}"),
        line("ticker share of prediction variance", "ticker_variance_share_y_pred", "{:.2f}"),
        line("ticker share of target variance", "ticker_variance_share_y_true", "{:.2f}"),
        "",
        "## Reading",
    ]
    for h in hs:
        notes = []
        seen = _get(rep, h, "test_calls_ticker_in_fit_pct")
        if seen >= 50:
            notes.append(
                f"{seen:.0f}% of test calls come from companies in training — the split does "
                "not test generalisation to new companies"
            )
        gap = _get(rep, h, "gap_business_days")
        if gap < h:
            notes.append(
                f"gap of {gap:.0f} business days < horizon {h}: training targets overlap "
                "the test period"
            )
        w, o = _get(rep, h, "within_swap_change_pct"), _get(rep, h, "other_swap_change_pct")
        if np.isnan(w):
            notes.append(
                "no company has two test calls, so the within-company controls cannot run "
                "(pool several test periods, or supply a longer test window)"
            )
        else:
            if w <= 5 and o > 10:
                notes.append(
                    "predictions are a function of the company: a same-company swap changes MSE by "
                    f"{w:+.1f}% while another company's costs {o:+.1f}%"
                    + (" (the call-specific part of the prediction hurts)" if w < -2 else "")
                )
            elif w > 5:
                notes.append(
                    f"predictions carry call-specific signal (same-company swap {w:+.1f}%)"
                )
            else:
                notes.append(
                    f"neither swap costs much (same company {w:+.1f}%, other {o:+.1f}%): the "
                    "differences between predictions carry little that matches the target"
                )
        p_t = _get(rep, h, "dm_p_vs_ticker_only")
        r2 = _get(rep, h, "r2_oos_vs_ticker_only")
        if not np.isnan(r2):
            beats = r2 > 0 and p_t < 0.05
            notes.append(
                f"{'beats' if beats else 'does not significantly beat'} the ticker-only model "
                f"(R²_OOS {r2:+.3f}, DM p {p_t:.3f})"
            )
        r2p = _get(rep, h, "r2_oos_vs_persistence")
        if not np.isnan(r2p):
            notes.append(
                f"R²_OOS vs persistence {r2p:+.3f} (DM p {_get(rep, h, 'dm_p_vs_persistence'):.3f})"
            )
        lines.append(f"- **τ={h}:** " + "; ".join(notes) + ".")
    return "\n".join(lines) + "\n"


def run_audit(
    path: str | Path, out_dir: str | Path, *, name: str | None = None, seed: int = 0
) -> Path:
    path, out = Path(path), Path(out_dir)
    df = load_frame(path)
    rep = audit(df, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "report.csv", index=False, lineterminator="\n")
    (out / "report.md").write_text(render_markdown(rep, name or path.stem), encoding="utf-8")
    return out
