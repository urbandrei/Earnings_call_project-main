"""Figure stub for Result Table 1 (T2.3 subtask).

Loads the committed Stage-0/1 metrics and sketches the headline figures for the
paper. Kept as a runnable skeleton (no plotting dependency pinned yet — add
matplotlib to the lockfile when figures are finalized in Phase 8). Run as a
script or paste cells into a notebook.

Planned figures:
- R²_OOS vs persistence by model × horizon (FinCall temporal vs ticker-disjoint),
  showing the long-horizon regime-shift effect (DECISIONS 2026-06-18).
- MSE leaderboard per horizon (literature comparability).
- temporal-vs-disjoint gap per model (the identity-control story, DESIGN §7.3).
"""

from pathlib import Path

import pandas as pd

RESULTS = Path("data/results/result_table_1.csv")


def load() -> pd.DataFrame:
    return pd.read_csv(RESULTS)


def r2_by_model_horizon(
    df: pd.DataFrame, dataset: str, split: str, target: str = "v"
) -> pd.DataFrame:
    sel = df[
        (df["dataset"] == dataset)
        & (df["split"] == split)
        & (df["target"] == target)
        & (df["segment"] == "test")
    ]
    return sel.pivot_table(index="model", columns="horizon", values="r2_oos")


def main() -> None:
    df = load()
    print("FinCall temporal, level-v — R²_OOS by model × horizon:")
    print(r2_by_model_horizon(df, "fincall", "temporal").round(3))
    # TODO(Phase 8): matplotlib bar/line figures from these frames; save to data/results/figures/.


if __name__ == "__main__":
    main()
