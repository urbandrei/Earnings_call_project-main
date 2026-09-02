"""T9.4 Earnings25 audio ladder: stratum cohorts and the per-stratum headline summary."""

from pathlib import Path

import pandas as pd

from ecvol.eval import earnings25_audio as A


def test_stratum_ids_from_inventory(tmp_path: Path):
    cov = tmp_path / "coverage"
    cov.mkdir()
    pd.DataFrame(
        {
            "call_id": ["1", "2", "3", "4"],
            "stratum": ["64k", "le24k", "", "other"],
        }
    ).to_csv(cov / "earnings25_inventory.csv", index=False)
    ids = A.stratum_ids(tmp_path)
    assert ids["all"] == {"1", "2", "4"}  # unprobed call 3 is not in any cohort
    assert ids["64k"] == {"1"} and ids["le24k"] == {"2"}


def test_strata_summary_picks_headline_cells():
    table = pd.DataFrame(
        [
            {
                "stratum": s,
                "split": "ticker_disjoint",
                "target": "dv",
                "horizon": h,
                "model": A.HEADLINE_MODEL,
                "segment": "test",
                "n": 10,
                "r2_oos": 0.1 * i,
                "dm_p_vs_har": 0.5,
            }
            for i, (s, h) in enumerate((s, h) for s in A.STRATA for h in (3, 30))
        ]
    )
    shuffle = pd.DataFrame(
        [
            {"stratum": s, "split": "ticker_disjoint", "horizon": h, "condition": c, "r2_oos": v}
            for s in A.STRATA
            for h in (3, 30)
            for c, v in (("real", 0.2), ("global_shuffle", -0.1), ("within_shuffle", 0.2))
        ]
    )
    ids = {"all": set("abc"), "64k": set("ab"), "le24k": set("c")}
    out = A._strata_summary(table, shuffle, ids)
    assert len(out) == len(A.STRATA) * 4  # every stratum × every horizon, NaN when absent
    row = out[(out.stratum == "64k") & (out.horizon == 30)].iloc[0]
    assert row.n_calls == 2 and row.n_test == 10
    assert row.shuffle_real_r2 == 0.2 and row.shuffle_global_r2 == -0.1
    assert out[(out.stratum == "all") & (out.horizon == 7)].r2_oos_vs_persistence.isna().all()
