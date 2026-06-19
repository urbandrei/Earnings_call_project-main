"""T2.3 reporting: table specs, rendering, byte-identical regeneration.

The acceptance test (DESIGN §7 / TASKS T2.3) is byte-identical regeneration —
checked both on synthetic data and on the *committed* Result Table 1 artifacts.
"""

from pathlib import Path

import pandas as pd

from ecvol.eval import report as R


def _results_frame() -> pd.DataFrame:
    rows = []
    for model in R.MODEL_ORDER:
        for h in R.HORIZONS:
            rows.append(
                {
                    "dataset": "fincall",
                    "split": "temporal",
                    "target": "v",
                    "horizon": h,
                    "model": model,
                    "segment": "test",
                    "n": 100,
                    "mse": 0.5,
                    "mae": 0.4,
                    "r2_oos": 0.0 if model == "persistence" else 0.3,
                    "spearman_q": 0.5,
                    "gbdt_mse_std": float("nan"),
                    "dm_p_vs_persistence": float("nan") if model == "persistence" else 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_build_table_order_and_shape():
    df = _results_frame()
    spec = R.TableSpec("fincall", "temporal", "v", "r2_oos")
    header, body, n = R.build_table(df, spec)
    assert header == ["Model", "τ=3", "τ=7", "τ=15", "τ=30"]
    assert [r[0] for r in body] == [R.MODEL_LABELS[m] for m in R.MODEL_ORDER]
    assert n == 100


def test_cell_dm_significance_star():
    df = _results_frame()
    spec = R.TableSpec("fincall", "temporal", "v", "r2_oos")
    _, body, _ = R.build_table(df, spec)
    # persistence: no star; others: significant (p=0.01<0.05) → "*"
    assert body[0][1] == "0.000"  # persistence
    assert body[1][1].endswith("*")  # ewma


def test_missing_cell_is_em_dash():
    df = _results_frame()
    df = df[df["horizon"] != 30]  # drop τ=30
    spec = R.TableSpec("fincall", "temporal", "v", "r2_oos")
    _, body, _ = R.build_table(df, spec)
    assert all(r[-1] == "—" for r in body)  # τ=30 column all missing


def test_renders_both_formats(tmp_path: Path):
    root = tmp_path / "data"
    (root / "results").mkdir(parents=True)
    _results_frame().to_csv(root / "results" / "result_table_1.csv", index=False)
    md_path, tex_path = R.write_reports(
        root, specs=[R.TableSpec("fincall", "temporal", "v", "r2_oos")]
    )
    md, tex = md_path.read_text(encoding="utf-8"), tex_path.read_text(encoding="utf-8")
    assert "| Model |" in md and "HAR-RV" in md
    assert r"\begin{tabular}" in tex and r"\toprule" in tex and "HAR-RV" in tex


def test_write_reports_deterministic(tmp_path: Path):
    root = tmp_path / "data"
    (root / "results").mkdir(parents=True)
    _results_frame().to_csv(root / "results" / "result_table_1.csv", index=False)
    R.write_reports(root)
    md1 = (root / "results" / "result_table_1.md").read_bytes()
    tex1 = (root / "results" / "result_table_1.tex").read_bytes()
    R.write_reports(root)
    assert (root / "results" / "result_table_1.md").read_bytes() == md1
    assert (root / "results" / "result_table_1.tex").read_bytes() == tex1


# --- committed artifacts stay in sync (CI-guarded; DESIGN §7 byte-identical) --

REPO_RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def test_committed_reports_match_fresh_render():
    """The committed .md/.tex must equal a fresh render of the committed CSV."""
    if not (REPO_RESULTS / "result_table_1.csv").is_file():
        return
    df = pd.read_csv(REPO_RESULTS / "result_table_1.csv")
    assert R.render_markdown(df, R.TABLE_1_SPECS) == (REPO_RESULTS / "result_table_1.md").read_text(
        encoding="utf-8"
    )
    assert R.render_latex(df, R.TABLE_1_SPECS) == (REPO_RESULTS / "result_table_1.tex").read_text(
        encoding="utf-8"
    )
