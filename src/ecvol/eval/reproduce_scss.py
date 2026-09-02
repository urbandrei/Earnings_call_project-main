"""T6R.2 — Same-Company-Same-Signal (Yu, Liu & He, Findings of ACL 2025) on its own DEC data.

SCSS's contribution is two training-free baselines: **PEV(Mean)** — the mean of all
historical post-earnings volatilities available before the test quarter — and
**STPEV(Mean)** — the same restricted to the *same ticker* (the "volatility
signature"). DEC (`raw/ref/scss/DEC.csv`, 90 tickers × 20 quarters, 2019–2023)
ships the τ-day post-earnings log volatilities `lv{τ}_future_{τ}`, the
precomputed same-ticker means `stpev_mean_lv{τ}f{τ}`, and one rolling train/val/
test mask per quarter (`rolling_test_on_{year}_{quarter}_cate`; test = the 90
calls of that quarter). The paper reports MSE per quarter and horizon (Table 4);
we recompute PEV from the train+val rows of each mask and use the shipped STPEV.

Controls: STPEV is *by construction* same-ticker history, so under a
\tickerdisjoint{} evaluation it is undefined — that is the finding, recorded as a
reason code rather than a number. PEV(Mean) uses no ticker identity and is
unchanged. DEC's rolling design already embargoes by quarter (train ends before
the test quarter begins), so no separate temporal-embargo condition is needed.

Output: `results/result_table_6r_scss.csv` — (year, quarter, horizon, model, mse,
n, published_mse) rows for the reproduction plus a `condition` column, with the
`ticker_disjoint` rows for STPEV carrying `reason=undefined_without_same_ticker_history`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEC_REL = "raw/ref/scss/DEC.csv"
TAUS = (3, 7, 15, 30)
YEARS = (2021, 2022, 2023)
QUARTERS = ("first", "second", "third", "fourth")
# Table 4 (2023 block), quarterly average MSE over τ ∈ {3,7,15,30}, as published.
PUBLISHED_2023 = {
    "stpev_mean": {"first": 0.239, "second": 0.253, "third": 0.227, "fourth": 0.246},
    "pev_mean": {"first": 0.309, "second": 0.330, "third": 0.262, "fourth": 0.278},
}


def run_scss_reproduction(root: Path) -> pd.DataFrame:
    d = pd.read_csv(root / DEC_REL)
    rows = []
    for year in YEARS:
        for q in QUARTERS:
            col = f"rolling_test_on_{year}_{q}_cate"
            te = d[d[col] == "test"]
            tr = d[d[col].isin(["train", "val"])]
            per_tau = {"stpev_mean": [], "pev_mean": []}
            for tau in TAUS:
                y = te[f"lv{tau}_future_{tau}"].to_numpy(dtype=float)
                st = te[f"stpev_mean_lv{tau}f{tau}"].to_numpy(dtype=float)
                pev = float(tr[f"lv{tau}_future_{tau}"].mean())
                m = np.isfinite(y) & np.isfinite(st)
                mse_st = float(np.mean((y[m] - st[m]) ** 2))
                mse_pev = float(np.mean((y[np.isfinite(y)] - pev) ** 2))
                per_tau["stpev_mean"].append(mse_st)
                per_tau["pev_mean"].append(mse_pev)
                for model, mse, n in (
                    ("stpev_mean", mse_st, int(m.sum())),
                    ("pev_mean", mse_pev, int(np.isfinite(y).sum())),
                ):
                    rows.append(
                        {
                            "dataset": "dec",
                            "condition": "published_rolling",
                            "year": year,
                            "quarter": q,
                            "horizon": tau,
                            "model": model,
                            "n_test": n,
                            "mse": mse,
                            "published_mse": np.nan,
                            "reason": "",
                        }
                    )
            for model, mses in per_tau.items():
                rows.append(
                    {
                        "dataset": "dec",
                        "condition": "published_rolling",
                        "year": year,
                        "quarter": q,
                        "horizon": 0,  # average over τ, the paper's headline cell
                        "model": model,
                        "n_test": int(len(te)),
                        "mse": float(np.mean(mses)),
                        "published_mse": PUBLISHED_2023[model][q] if year == 2023 else np.nan,
                        "reason": "",
                    }
                )
    # the controlled condition: STPEV has no definition without same-ticker history
    for tau in (*TAUS, 0):
        rows.append(
            {
                "dataset": "dec",
                "condition": "ticker_disjoint",
                "year": 0,
                "quarter": "all",
                "horizon": tau,
                "model": "stpev_mean",
                "n_test": 0,
                "mse": np.nan,
                "published_mse": np.nan,
                "reason": "undefined_without_same_ticker_history",
            }
        )
    table = pd.DataFrame(rows)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_6r_scss.csv", index=False, lineterminator="\n")
    return table
