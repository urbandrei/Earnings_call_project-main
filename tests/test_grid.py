"""T5.2 Result Table 4: canonical-model consolidation, Holm correction, rendering."""

from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.eval import grid
from ecvol.eval import report as R


def _stage_csv(root: Path, name: str, model: str, *, dm=None):
    rows = []
    for split in ("temporal", "ticker_disjoint"):
        for target in ("v", "dv"):
            for h in (3, 7, 15, 30):
                for seg in ("val", "test"):
                    row = {
                        "dataset": "fincall",
                        "split": split,
                        "target": target,
                        "horizon": h,
                        "model": model,
                        "segment": seg,
                        "n": 100,
                        "mse": 0.4,
                        "mae": 0.3,
                        "r2_oos": 0.1,
                        "spearman_q": 0.2,
                    }
                    if dm is not None:
                        row["dm_p_vs_stage1"] = dm
                    rows.append(row)
    (root / "results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "results" / name, index=False)


def test_build_grid_consolidates_and_holm(tmp_path: Path):
    root = tmp_path / "data"
    # table1 has persistence/har/gbdt (no dm_p_vs_stage1); table2/3/4 have it
    rows1 = []
    for model in ("persistence", "har", "gbdt_tickerFE"):
        for split in ("temporal", "ticker_disjoint"):
            for target in ("v", "dv"):
                for h in (3, 7, 15, 30):
                    for seg in ("val", "test"):
                        rows1.append(
                            {
                                "dataset": "fincall",
                                "split": split,
                                "target": target,
                                "horizon": h,
                                "model": model,
                                "segment": seg,
                                "n": 100,
                                "mse": 0.4,
                                "mae": 0.3,
                                "r2_oos": 0.0,
                                "spearman_q": 0.1,
                                "dm_p_vs_persistence": np.nan,
                            }
                        )
    (root / "results").mkdir(parents=True)
    pd.DataFrame(rows1).to_csv(root / "results" / "result_table_1.csv", index=False)
    _stage_csv(root, "result_table_2.csv", "ridge_text_pastvol", dm=0.01)
    _stage_csv(root, "result_table_3.csv", "ridge_wavlm_egemaps_audio_pastvol", dm=0.2)
    _stage_csv(root, "result_table_4_fusion.csv", "stack_fusion_pastvol", dm=0.4)

    g = grid.build_grid(root)
    assert set(g["stage"]) == {
        "S0_persistence",
        "S0_HAR",
        "S1_GBDT",
        "S2_text",
        "S3_audio",
        "S4_fusion",
    }
    # Holm only on confirmatory stages, Δv test
    conf = g[
        (g.stage == "S2_text")
        & (g.target == "dv")
        & (g.segment == "test")
        & (g.split == "temporal")
    ]
    assert conf["holm_p_vs_stage1"].notna().all()
    # Holm-adjusted >= raw (4 horizons, all p=0.01 → adjusted up to 0.04)
    assert (conf["holm_p_vs_stage1"] >= conf["dm_p_vs_stage1"] - 1e-12).all()
    # S0/S1 get no Holm value
    s1 = g[(g.stage == "S1_GBDT") & (g.target == "dv") & (g.segment == "test")]
    assert s1["holm_p_vs_stage1"].isna().all()


def test_render_table4_holm_star():
    rows = []
    for stage in R.TABLE4_STAGE_ORDER:
        for h in R.HORIZONS:
            rows.append(
                {
                    "dataset": "fincall",
                    "split": "temporal",
                    "target": "dv",
                    "horizon": h,
                    "stage": stage,
                    "segment": "test",
                    "n": 100,
                    "r2_oos": 0.2,
                    "mse": 0.3,
                    "dm_p_vs_stage1": 0.01,
                    "holm_p_vs_stage1": 0.02 if stage == "S2_text" else 0.5,
                }
            )
    df = pd.DataFrame(rows)
    spec = R.TableSpec("fincall", "temporal", "dv", "r2_oos")
    header, body, _ = R.build_table4(df, spec)
    assert header[0] == "Stage"
    labels = [r[0] for r in body]
    assert labels == [R.TABLE4_STAGE_LABELS[s] for s in R.TABLE4_STAGE_ORDER]
    text_row = body[labels.index("Text (BGE+vol)")]
    assert text_row[1].endswith("*")  # Holm-significant
    audio_row = body[labels.index("Audio (WavLM+eG+vol)")]
    assert not audio_row[1].endswith("*")
    md = R.render_markdown4(df, [spec])
    assert "Result Table 4" in md and "Holm" in md


REPO_RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def test_committed_table4_matches_fresh_render():
    """Committed Table-4 .md/.tex must equal a fresh render of the committed CSV (DESIGN §7)."""
    if not (REPO_RESULTS / "result_table_4.csv").is_file():
        return
    df = pd.read_csv(REPO_RESULTS / "result_table_4.csv")
    assert R.render_markdown4(df, R.TABLE_4_SPECS) == (
        REPO_RESULTS / "result_table_4.md"
    ).read_text(encoding="utf-8")
    assert R.render_latex4(df, R.TABLE_4_SPECS) == (REPO_RESULTS / "result_table_4.tex").read_text(
        encoding="utf-8"
    )
