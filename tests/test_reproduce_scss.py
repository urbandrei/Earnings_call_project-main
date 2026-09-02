"""T6R.2 SCSS: PEV/STPEV recomputation over DEC's rolling masks, and the disjoint reason code."""

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import reproduce_scss as S


def test_scss_reproduction_on_synthetic_dec(tmp_path: Path):
    root = tmp_path
    (root / S.DEC_REL).parent.mkdir(parents=True)
    rows = []
    masks = [f"rolling_test_on_{y}_{q}_cate" for y in S.YEARS for q in S.QUARTERS]
    for i in range(6):
        r = {"ticker": "A" if i % 2 else "B"}
        for tau in S.TAUS:
            r[f"lv{tau}_future_{tau}"] = float(i)
            r[f"stpev_mean_lv{tau}f{tau}"] = float(i) - 1.0  # off by one everywhere
        for m in masks:
            r[m] = "test" if i >= 4 else ("val" if i == 3 else "train")
        rows.append(r)
    pd.DataFrame(rows).to_csv(root / S.DEC_REL, index=False)

    t = S.run_scss_reproduction(root)
    cell = t[(t.year == 2023) & (t.quarter == "first") & (t.horizon == 3)].set_index("model")
    assert cell.loc["stpev_mean", "mse"] == 1.0  # (i − (i−1))² = 1
    # PEV = mean of train+val targets (0,1,2,3 → 1.5); test 4,5 → (2.5² + 3.5²)/2 = 9.25
    assert np.isclose(cell.loc["pev_mean", "mse"], 9.25)
    avg = t[(t.year == 2023) & (t.quarter == "first") & (t.horizon == 0)].set_index("model")
    assert avg.loc["stpev_mean", "published_mse"] == 0.239 and avg.loc["stpev_mean", "n_test"] == 2
    dj = t[t.condition == "ticker_disjoint"]
    assert (
        set(dj.model) == {"stpev_mean"}
        and (dj.reason == "undefined_without_same_ticker_history").all()
    )
    assert (root / "results" / "result_table_6r_scss.csv").is_file()
