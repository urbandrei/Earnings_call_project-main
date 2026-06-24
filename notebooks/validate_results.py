"""Validation harness (not pipeline code) — sanity-check that the results so far
are real, not junk.

Run:  .venv/Scripts/python.exe notebooks/validate_results.py

Produces:
  data/results/figures/*.png            — result + data + feature visualizations
  data/results/figures/target_handcheck.csv  — sample calls for manual checking
  stdout                                 — reconciliation + leakage report

The target reconciliation deliberately RE-IMPLEMENTS the realized-vol math from
scratch (numpy over the raw price parquets), independent of src/ecvol, so a bug
in targets.py would surface as a mismatch rather than agreeing with itself.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("data")
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
HORIZONS = [3, 7, 15, 30]


# --------------------------------------------------------------------------- #
# 1. INDEPENDENT TARGET RECONCILIATION                                         #
# --------------------------------------------------------------------------- #
def load_close(ticker: str) -> pd.DataFrame:
    p = ROOT / "prices" / f"{ticker}.parquet"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(p, columns=["date", "close"])
    df["date"] = df["date"].astype(str)
    return df.sort_values("date").reset_index(drop=True)


def rvol(returns: np.ndarray) -> float:
    """ln(sqrt(population variance)) of a return series — independent reimpl."""
    if returns.size < 2:
        return math.nan
    var = returns.var()  # numpy default ddof=0 -> population variance
    return math.log(math.sqrt(var)) if var > 0 else math.nan


def recompute_call(ticker: str, as_of: str, tau: int, cache: dict):
    """Independently recompute (v_pre, v_post, delta_v) for one (call, tau).

    Sessions are taken to be the price-parquet's own trading dates, so this
    does not reuse the project's NYSE calendar code at all.
    """
    if ticker not in cache:
        cache[ticker] = load_close(ticker)
    px = cache[ticker]
    if px.empty or as_of not in set(px["date"]):
        return (math.nan, math.nan, math.nan)
    i = px.index[px["date"] == as_of][0]
    closes = px["close"].to_numpy()
    # need closes[i-tau .. i+tau]
    if i - tau < 0 or i + tau >= len(closes):
        return (math.nan, math.nan, math.nan)
    pre_px = closes[i - tau : i + 1]  # tau returns ending day0
    post_px = closes[i : i + tau + 1]  # tau returns starting day0+1
    pre_ret = np.diff(pre_px) / pre_px[:-1]
    post_ret = np.diff(post_px) / post_px[:-1]
    v_pre, v_post = rvol(pre_ret), rvol(post_ret)
    delta = v_post - v_pre if not (math.isnan(v_pre) or math.isnan(v_post)) else math.nan
    return (v_pre, v_post, delta)


def reconcile_targets():
    print("=" * 70)
    print("1. TARGET RECONCILIATION  (independent numpy recompute vs stored)")
    print("=" * 70)
    tg = pd.read_parquet(ROOT / "fincall" / "targets.parquet")
    ok = tg[tg["status"] == "ok"].copy()
    print(
        f"FinCall targets: {len(tg)} rows, {len(ok)} ok, "
        f"{tg['call_id'].nunique()} calls, tickers={tg['ticker'].nunique()}"
    )

    cache: dict = {}
    recomp = ok.apply(
        lambda r: recompute_call(r["ticker"], r["as_of"], int(r["horizon"]), cache),
        axis=1,
        result_type="expand",
    )
    recomp.columns = ["v_pre_chk", "v_post_chk", "delta_chk"]
    # reset BOTH indices: ok.apply keeps ok's original (non-contiguous) index, so
    # concat(axis=1) against a renumbered ok would align by index and compare each
    # row to a different row's recompute. Reset both so positions line up.
    m = pd.concat([ok.reset_index(drop=True), recomp.reset_index(drop=True)], axis=1)
    for a, b in [("v_pre", "v_pre_chk"), ("v_post", "v_post_chk"), ("delta_v", "delta_chk")]:
        d = (m[a] - m[b]).abs()
        finite = d[np.isfinite(d)]
        print(
            f"  {a:9s}: max|Δ|={finite.max():.2e}  mean|Δ|={finite.mean():.2e}  "
            f"n_compared={finite.notna().sum()}  n_unmatched={d.isna().sum()}"
        )

    # sample hand-check CSV: a few calls, full price window shown
    sample_ids = ok["call_id"].drop_duplicates().head(8)
    rows = []
    for cid in sample_ids:
        sub = ok[ok["call_id"] == cid]
        tk, ao = sub.iloc[0]["ticker"], sub.iloc[0]["as_of"]
        px = cache.get(tk, load_close(tk))
        if px.empty or ao not in set(px["date"]):
            continue
        i = px.index[px["date"] == ao][0]
        for _, r in sub.iterrows():
            tau = int(r["horizon"])
            window = px.iloc[max(0, i - tau) : i + tau + 1][["date", "close"]]
            rows.append(
                {
                    "call_id": cid,
                    "ticker": tk,
                    "call_date": r["call_date"],
                    "day0_as_of": ao,
                    "horizon": tau,
                    "v_pre_stored": round(r["v_pre"], 6),
                    "v_post_stored": round(r["v_post"], 6),
                    "delta_v_stored": round(r["delta_v"], 6),
                    "v_pre_recomputed": round(recompute_call(tk, ao, tau, cache)[0], 6),
                    "v_post_recomputed": round(recompute_call(tk, ao, tau, cache)[1], 6),
                    "n_prices_in_window": len(window),
                    "first_price_date": window["date"].iloc[0],
                    "last_price_date": window["date"].iloc[-1],
                }
            )
    hc = pd.DataFrame(rows)
    hc.to_csv(FIG / "target_handcheck.csv", index=False)
    print(f"  -> wrote {FIG / 'target_handcheck.csv'} ({len(hc)} sample rows)")
    return tg, ok


# --------------------------------------------------------------------------- #
# 2. LEAKAGE CHECKS  (the most dangerous bug class)                            #
# --------------------------------------------------------------------------- #
def check_leakage():
    print("\n" + "=" * 70)
    print("2. LEAKAGE CHECKS  (temporal order + ticker-disjointness)")
    print("=" * 70)
    for ds in ["fincall", "maec"]:
        # temporal: max train as_of must be < min test as_of
        t = pd.read_csv(ROOT / "splits" / f"{ds}_temporal.csv")
        tr, te = t[t.split == "train"], t[t.split == "test"]
        gap_ok = tr["as_of"].max() < te["as_of"].min()
        print(
            f"  {ds} temporal: train≤{tr['as_of'].max()}  test≥{te['as_of'].min()}  "
            f"ordered={'PASS' if gap_ok else 'FAIL'}"
        )
        # ticker-disjoint: zero ticker overlap between train and test
        d = pd.read_csv(ROOT / "splits" / f"{ds}_ticker_disjoint.csv")
        dtr, dte = set(d[d.split == "train"].ticker), set(d[d.split == "test"].ticker)
        overlap = dtr & dte
        print(
            f"  {ds} ticker_disjoint: train_tk={len(dtr)} test_tk={len(dte)} "
            f"overlap={len(overlap)}  disjoint={'PASS' if not overlap else 'FAIL'}"
        )


# --------------------------------------------------------------------------- #
# 3. TARGET DISTRIBUTIONS                                                      #
# --------------------------------------------------------------------------- #
def plot_target_distributions(ok: pd.DataFrame):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for col, row in zip(["v_pre", "v_post", "delta_v"], range(3), strict=False):
        for j, tau in enumerate(HORIZONS):
            ax = axes[row][j]
            vals = ok[ok.horizon == tau][col].dropna()
            ax.hist(vals, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
            ax.axvline(vals.median(), color="crimson", lw=1.2, ls="--")
            ax.set_title(f"{col}  τ={tau}  (med={vals.median():.2f})", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle("FinCall target distributions (log realized-vol; red = median)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "target_distributions.png", dpi=130)
    plt.close(fig)
    print(f"  -> {FIG / 'target_distributions.png'}")


# --------------------------------------------------------------------------- #
# 4. RESULT HEATMAPS  (R2_OOS by model x horizon)                              #
# --------------------------------------------------------------------------- #
def plot_result_heatmaps():
    print("\n" + "=" * 70)
    print("3/4. RESULT VISUALIZATIONS")
    print("=" * 70)
    specs = [
        ("result_table_1.csv", "Stage-0/1 baselines"),
        ("result_table_2.csv", "Stage-2 text heads"),
        ("result_table_3.csv", "Stage-3 audio heads"),
    ]
    for fname, title in specs:
        df = pd.read_csv(ROOT / "results" / fname)
        df = (
            df[(df.segment == "test") & (df.target == "v") & (df.metric == "r2_oos")]
            if "metric" in df.columns
            else df[(df.segment == "test") & (df.target == "v")]
        )
        combos = df[["dataset", "split"]].drop_duplicates().values.tolist()
        n = len(combos)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 0.45 * df.model.nunique() + 2))
        if n == 1:
            axes = [axes]
        for ax, (ds, sp) in zip(axes, combos, strict=False):
            sel = df[(df.dataset == ds) & (df.split == sp)]
            piv = sel.pivot_table(index="model", columns="horizon", values="r2_oos")
            piv = piv.reindex(columns=HORIZONS)
            data = piv.to_numpy(dtype=float)
            im = ax.imshow(data, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(HORIZONS)), [f"τ={t}" for t in HORIZONS], fontsize=8)
            ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
            ax.set_title(f"{ds} / {sp}", fontsize=9)
            for yy in range(data.shape[0]):
                for xx in range(data.shape[1]):
                    v = data[yy, xx]
                    if np.isfinite(v):
                        ax.text(
                            xx,
                            yy,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="black" if abs(v) < 0.6 else "white",
                        )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{title} — R²_OOS vs persistence (test, level-v). "
            "Blue=better, Red=worse than persistence.",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = FIG / f"{fname.replace('.csv', '')}_r2_heatmap.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")


# --------------------------------------------------------------------------- #
# 5. TEMPORAL vs TICKER-DISJOINT GAP  (the identity story)                     #
# --------------------------------------------------------------------------- #
def plot_identity_gap():
    df = pd.read_csv(ROOT / "results" / "result_table_1.csv")
    df = (
        df[(df.segment == "test") & (df.target == "v") & (df.metric == "r2_oos")]
        if "metric" in df.columns
        else df[(df.segment == "test") & (df.target == "v")]
    )
    ds = "fincall"
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4 * len(HORIZONS), 4), sharey=True)
    for ax, tau in zip(axes, HORIZONS, strict=False):
        t = df[(df.dataset == ds) & (df.split == "temporal") & (df.horizon == tau)].set_index(
            "model"
        )["r2_oos"]
        d = df[
            (df.dataset == ds) & (df.split == "ticker_disjoint") & (df.horizon == tau)
        ].set_index("model")["r2_oos"]
        models = [m for m in t.index if m in d.index]
        x = np.arange(len(models))
        ax.bar(x - 0.2, t[models].values, 0.4, label="temporal", color="darkorange")
        ax.bar(x + 0.2, d[models].values, 0.4, label="ticker_disjoint", color="seagreen")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, models, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"τ={tau}", fontsize=10)
        if tau == HORIZONS[0]:
            ax.legend(fontsize=8)
            ax.set_ylabel("R²_OOS vs persistence")
    fig.suptitle(
        "FinCall: temporal vs ticker-disjoint R²_OOS (Stage-1). "
        "Large gaps ⇒ ticker-identity memorization.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "identity_gap.png", dpi=130)
    plt.close(fig)
    print(f"  -> {FIG / 'identity_gap.png'}")


# --------------------------------------------------------------------------- #
# 6. FEATURE SANITY                                                            #
# --------------------------------------------------------------------------- #
def vec_matrix(path: str, scope_filter=None):
    df = pd.read_parquet(ROOT / path)
    if scope_filter is not None and "scope" in df.columns:
        df = df[df["scope"] == scope_filter]
    return np.vstack(df["vector"].to_numpy())


def plot_feature_sanity():
    print("\n" + "=" * 70)
    print("5. FEATURE SANITY")
    print("=" * 70)
    panels = []
    # text embeddings (BGE-M3, full scope)
    try:
        X = vec_matrix("fincall/text_embeddings.parquet", scope_filter="full")
        panels.append(("BGE-M3 text emb", X))
    except Exception as e:
        print("  text_embeddings:", e)
    for name, p in [
        ("WavLM audio", "fincall/audio_wavlm.parquet"),
        ("emotion2vec+ audio", "fincall/audio_emotion2vec.parquet"),
    ]:
        try:
            panels.append((name, vec_matrix(p)))
        except Exception as e:
            print(f"  {p}:", e)

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, X) in zip(axes, panels, strict=False):
        norms = np.linalg.norm(X, axis=1)
        n_nan = int(np.isnan(X).sum())
        n_const = int((X.std(axis=0) == 0).sum())
        ax.hist(norms, bins=50, color="purple", edgecolor="white", linewidth=0.3)
        ax.set_title(
            f"{name}\n{X.shape[0]}×{X.shape[1]}  NaNs={n_nan}  const_dims={n_const}", fontsize=9
        )
        ax.set_xlabel("L2 norm per call", fontsize=8)
        print(
            f"  {name:22s} shape={X.shape}  NaNs={n_nan}  const_dims={n_const}  "
            f"norm[min={norms.min():.2f} max={norms.max():.2f}]"
        )
    fig.suptitle("Frozen feature sanity (per-call vector L2 norms)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "feature_sanity.png", dpi=130)
    plt.close(fig)
    print(f"  -> {FIG / 'feature_sanity.png'}")

    # FinBERT sentiment distribution
    try:
        fb = pd.read_parquet(ROOT / "fincall/text_finbert.parquet")
        fb = fb[fb["scope"] == "full"] if "full" in set(fb.get("scope", [])) else fb
        fig, ax = plt.subplots(figsize=(7, 4))
        for c, col in [
            ("p_positive", "seagreen"),
            ("p_negative", "crimson"),
            ("p_neutral", "gray"),
        ]:
            ax.hist(fb[c].dropna(), bins=40, alpha=0.5, label=c, color=col)
        ax.legend(fontsize=8)
        ax.set_title("FinBERT sentiment probabilities (FinCall, full scope)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "finbert_sentiment.png", dpi=130)
        plt.close(fig)
        print(f"  -> {FIG / 'finbert_sentiment.png'}")
        print(
            f"  FinBERT rows={len(fb)} mean p_pos={fb['p_positive'].mean():.3f} "
            f"p_neg={fb['p_negative'].mean():.3f} p_neu={fb['p_neutral'].mean():.3f}"
        )
    except Exception as e:
        print("  finbert:", e)


if __name__ == "__main__":
    _, ok = reconcile_targets()
    plot_target_distributions(ok)
    check_leakage()
    plot_result_heatmaps()
    plot_identity_gap()
    plot_feature_sanity()
    print("\nDONE. Figures + handcheck CSV in", FIG)
