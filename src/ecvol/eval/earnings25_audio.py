"""T9.4 — the frozen Stage-3 audio ladder on Earnings25, per bitrate stratum.

DECISIONS 2026-09-02 §1: Earnings25's audio is webcast MP3 at 64 kbps (281 calls) or
≤24 kbps (232 calls), so it cannot test "clean audio"; it can test whether the
FinCall "audio is inert" result depends on bitrate, *within one quarter*. This
module re-runs `eval.stage3.evaluate_stage3` (same extractors, same ridge/MLP
heads, same covariates and DM references as T4.4) and the same-ticker/global
audio shuffle control on three cohorts — `all`, `64k`, `le24k` — and writes:

- `results/result_table_3_earnings25.csv` — Result-Table-3 rows + a `stratum` column;
- `results/audio_shuffle_earnings25.csv` — shuffle-control cells + `stratum`;
- `results/audio_strata_earnings25.csv` — the headline per-stratum summary
  (WavLM+past-vol ridge Δv, test R²_OOS real vs global-shuffle, n per stratum).

Only the **ticker-disjoint** split is used: a single-quarter corpus has ~105
sessions, so a 30-session-embargoed temporal split leaves 5 training calls
(see JOURNAL 2026-09-02). Every ticker appears once, so the *within-ticker*
shuffle is the identity — the informative control here is the global shuffle,
and the identity probe is undefined (chance = 1/n); neither is reported.
Strata come from `coverage/earnings25_inventory.csv` (ffprobe at ingestion).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecvol.eval import evaluate as E
from ecvol.eval.audio_eval import audio_shuffle_control
from ecvol.eval.stage3 import evaluate_stage3

DATASET = "earnings25"
SCHEMES = ("ticker_disjoint",)
STRATA = ("all", "64k", "le24k")
HEADLINE_MODEL = "ridge_wavlm_audio_pastvol"


def stratum_ids(root: Path) -> dict[str, set[str]]:
    """{stratum: call_ids} from the ingestion inventory; `all` = every probed call."""
    inv = pd.read_csv(root / "coverage" / f"{DATASET}_inventory.csv", dtype={"call_id": str})
    inv = inv[inv["stratum"].notna() & (inv["stratum"] != "")]
    out = {"all": set(inv["call_id"])}
    for s in STRATA[1:]:
        out[s] = set(inv.loc[inv["stratum"] == s, "call_id"])
    return out


def run_earnings25_audio(root: Path, *, seeds=E.DEFAULT_SEEDS) -> dict[str, pd.DataFrame]:
    ids = stratum_ids(root)
    table_rows: list[pd.DataFrame] = []
    shuffle_rows: list[pd.DataFrame] = []
    for stratum in STRATA:
        cohort = ids[stratum]
        rows = evaluate_stage3(root, DATASET, seeds=seeds, schemes=SCHEMES, call_ids=cohort)
        t = pd.DataFrame(rows)
        t.insert(1, "stratum", stratum)
        table_rows.append(t)
        sh = audio_shuffle_control(root, DATASET, schemes=SCHEMES, call_ids=cohort, write=False)
        sh.insert(0, "stratum", stratum)
        shuffle_rows.append(sh)

    table = (
        pd.concat(table_rows, ignore_index=True)
        .sort_values(["stratum", "split", "target", "horizon", "model", "segment"])
        .reset_index(drop=True)
    )
    shuffle = pd.concat(shuffle_rows, ignore_index=True).reset_index(drop=True)
    summary = _strata_summary(table, shuffle, ids)

    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"result_table_3_{DATASET}.csv", index=False, lineterminator="\n")
    shuffle.to_csv(out / f"audio_shuffle_{DATASET}.csv", index=False, lineterminator="\n")
    summary.to_csv(out / f"audio_strata_{DATASET}.csv", index=False, lineterminator="\n")
    return {"table": table, "shuffle": shuffle, "summary": summary}


def _strata_summary(table: pd.DataFrame, shuffle: pd.DataFrame, ids: dict) -> pd.DataFrame:
    """Headline per (stratum, horizon): WavLM+past-vol ridge Δv test R²_OOS, DM p vs HAR,
    and the real / global-shuffle R²_OOS of the shuffle control."""
    rows = []
    head = table[
        (table["model"] == HEADLINE_MODEL)
        & (table["target"] == "dv")
        & (table["segment"] == "test")
        & (table["split"] == "ticker_disjoint")
    ]
    for stratum in STRATA:
        for tau in E.HORIZONS:
            h = head[(head["stratum"] == stratum) & (head["horizon"] == tau)]
            s = shuffle[(shuffle["stratum"] == stratum) & (shuffle["horizon"] == tau)]
            real = s[s["condition"] == "real"]["r2_oos"]
            glob = s[s["condition"] == "global_shuffle"]["r2_oos"]
            rows.append(
                {
                    "stratum": stratum,
                    "n_calls": len(ids[stratum]),
                    "horizon": int(tau),
                    "n_test": int(h["n"].iloc[0]) if len(h) else 0,
                    "r2_oos_vs_persistence": float(h["r2_oos"].iloc[0]) if len(h) else float("nan"),
                    "dm_p_vs_har": float(h["dm_p_vs_har"].iloc[0]) if len(h) else float("nan"),
                    "shuffle_real_r2": float(real.iloc[0]) if len(real) else float("nan"),
                    "shuffle_global_r2": float(glob.iloc[0]) if len(glob) else float("nan"),
                }
            )
    return pd.DataFrame(rows)
