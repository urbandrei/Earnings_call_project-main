"""T3.3 Stage-2 content heads: heads, feature assembly, Result-Table-2 orchestration, render.

All CPU/sklearn (no torch). Validates the head logic, the design-matrix assembly (incl.
zero-fill of missing scopes/roles), the end-to-end table-2 schema with the three DM columns,
and the Table-2 rendering (DM-vs-Stage-1 star).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.eval import report as R
from ecvol.eval import stage2
from ecvol.features.text import assemble
from ecvol.models import heads

# --- heads -------------------------------------------------------------------


def test_ridge_recovers_linear_and_selects_alpha():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    w = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
    y = X @ w + 0.01 * rng.normal(size=200)
    tr, val, te = slice(0, 120), slice(120, 160), slice(160, 200)
    pv, pte, alpha = heads.ridge_fit_predict(X[tr], y[tr], X[val], y[val], X[te])
    assert pte.shape == (40,)
    assert np.mean((y[te] - pte) ** 2) < 0.1  # near-perfect linear recovery
    assert alpha in heads.RIDGE_ALPHAS


def test_mlp_deterministic_per_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 6))
    y = (X[:, 0] ** 2 + X[:, 1]).astype(float)
    a1, b1 = heads.mlp_fit_predict(X[:100], y[:100], X[100:125], X[125:], seed=0)
    a2, b2 = heads.mlp_fit_predict(X[:100], y[:100], X[100:125], X[125:], seed=0)
    assert np.allclose(b1, b2)  # same seed → identical
    assert b1.shape == (25,)


def test_pca_reduce_shapes_and_train_fit():
    rng = np.random.default_rng(2)
    e = rng.normal(size=(80, 50))
    a, b, c = heads.pca_reduce(e[:50], e[50:65], e[65:], dim=16)
    assert a.shape == (50, 16) and b.shape == (15, 16) and c.shape == (15, 16)


# --- feature assembly --------------------------------------------------------


def _write_text_features(root: Path):
    d = root / "fincall"

    def emb_row(cid, scope):
        return {
            "call_id": cid,
            "source": "fincall",
            "scope": scope,
            "n_chunks": 1,
            "vector": list(np.full(assemble.EMB_DIM, float(cid))),
        }

    emb = [emb_row(1, "prepared_remarks"), emb_row(1, "qa"), emb_row(2, "prepared_remarks")]
    pq.write_table(pa.Table.from_pylist(emb), d / "text_embeddings.parquet")
    fb = [
        {
            "call_id": 1,
            "source": "fincall",
            "scope": "qa",
            "role": "analyst",
            "n_chunks": 1,
            "p_positive": 0.7,
            "p_negative": 0.1,
            "p_neutral": 0.2,
            "net": 0.6,
        },
    ]
    pq.write_table(pa.Table.from_pylist(fb), d / "text_finbert.parquet")
    sf = [
        {
            "call_id": 1,
            "source": "fincall",
            "scope": "prepared_remarks",
            "n_turns": 3,
            "n_chunks": 3,
            "n_words": 100,
            "n_chars": 500,
            "numeric_density": 0.2,
            "question_marks": 0,
            "words_per_turn": 33.3,
        },
    ]
    pq.write_table(pa.Table.from_pylist(sf), d / "text_surface.parquet")


def test_build_text_matrix_shapes_and_zero_fill(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fincall").mkdir(parents=True)
    _write_text_features(root)
    df, emb_cols, other_cols = assemble.build_text_matrix(root, "fincall")
    assert len(emb_cols) == assemble.EMB_DIM * 2  # prepared + qa
    assert df.shape[0] == 2  # two calls
    # call 2 has no qa scope → its qa half is zero-filled
    row2 = df[df["call_id"] == "2"].iloc[0]
    assert row2[f"e{assemble.EMB_DIM}"] == 0.0  # first qa-block column
    # finbert/surface present for call 1, zero for call 2
    assert df[df["call_id"] == "1"].iloc[0]["fb_qa_analyst_net"] == 0.6
    assert df[df["call_id"] == "2"].iloc[0]["fb_qa_analyst_net"] == 0.0


# --- end-to-end orchestration ------------------------------------------------


def _eval_frame(n_calls=80):
    rng = np.random.default_rng(3)
    rows = []
    quarters = ["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"]
    for c in range(n_calls):
        v_pre = float(rng.normal(-4, 0.5))
        for h in (3, 7, 15, 30):
            v_post = v_pre + 0.1 * h / 30 + float(rng.normal(0, 0.3))
            rows.append(
                {
                    "call_id": str(c),
                    "ticker": f"T{c % 10}",
                    "horizon": h,
                    "as_of": quarters[c % 4],
                    "v_pre": v_pre,
                    "v_post": v_post,
                    "delta_v": v_post - v_pre,
                    "rv_daily": abs(v_pre),
                    "rv_weekly": abs(v_pre),
                    "rv_monthly": abs(v_pre),
                    "n_turns": 20,
                    "n_chars": 5000,
                }
            )
    return pd.DataFrame(rows)


def _text_df(n_calls=80):
    rng = np.random.default_rng(4)
    emb_cols = ["e0", "e1", "e2"]
    other_cols = ["fb_qa_all_net", "sf_full_n_words"]
    data = {"call_id": [str(c) for c in range(n_calls)]}
    for col in emb_cols + other_cols:
        data[col] = rng.normal(size=n_calls)
    return pd.DataFrame(data), emb_cols, other_cols


def _splits_csv(path: Path, n_calls=80):
    rows = []
    for c in range(n_calls):
        seg = "train" if c < 56 else ("val" if c < 68 else "test")
        rows.append({"call_id": str(c), "split": seg})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_stage2_end_to_end_schema(tmp_path: Path):
    splits = tmp_path / "splits"
    splits.mkdir()
    _splits_csv(splits / "fincall_temporal.csv")
    df = _eval_frame()
    # Inject a NaN past-vol covariate (a real MAEC edge) to exercise the train-median impute
    # and the writeable-array copy in _predict.
    df.loc[df["call_id"] == "0", "rv_monthly"] = np.nan
    text_df, emb_cols, other_cols = _text_df()
    rows = stage2.evaluate_stage2_dataset(
        df, text_df, emb_cols, other_cols, "fincall", splits, seeds=(0, 1)
    )
    t = pd.DataFrame(rows)
    assert len(t) > 0
    assert {"dm_p_vs_persistence", "dm_p_vs_har", "dm_p_vs_stage1", "seed_std"} <= set(t.columns)
    assert {"ridge_text", "mlp_text_pastvol", "ridge_pastvol"} <= set(t["model"])
    # ridge has no seed_std; mlp does
    assert t[t.model == "ridge_text"]["seed_std"].isna().all()
    assert t[t.model == "mlp_text"]["seed_std"].notna().any()


# --- Result Table 2 rendering ------------------------------------------------


def test_render_table2_star_vs_stage1():
    rows = []
    for model in R.TABLE2_MODEL_ORDER:
        for h in R.HORIZONS:
            rows.append(
                {
                    "dataset": "fincall",
                    "split": "temporal",
                    "target": "dv",
                    "horizon": h,
                    "model": model,
                    "segment": "test",
                    "n": 100,
                    "mse": 0.4,
                    "mae": 0.3,
                    "r2_oos": 0.2,
                    "spearman_q": 0.4,
                    "seed_std": 0.01,
                    "dm_p_vs_persistence": 0.2,
                    "dm_p_vs_har": 0.2,
                    "dm_p_vs_stage1": 0.01 if model == "ridge_text" else 0.5,
                }
            )
    df = pd.DataFrame(rows)
    spec = R.TableSpec("fincall", "temporal", "dv", "r2_oos")
    _, body, n = R.build_table2(df, spec)
    labels = [r[0] for r in body]
    assert labels == [R.TABLE2_MODEL_LABELS[m] for m in R.TABLE2_MODEL_ORDER]
    ridge_text_row = body[labels.index("Ridge (text)")]
    assert ridge_text_row[1].endswith("*")  # DM-significant vs Stage-1
    mlp_text_row = body[labels.index("MLP (text)")]
    assert not mlp_text_row[1].endswith("*")
    md = R.render_markdown2(df, [spec])
    assert "Result Table 2" in md and "vs Stage-1" in md


REPO_RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def test_committed_table2_matches_fresh_render():
    """The committed Table-2 .md/.tex must equal a fresh render of the committed CSV (DESIGN §7)."""
    if not (REPO_RESULTS / "result_table_2.csv").is_file():
        return
    df = pd.read_csv(REPO_RESULTS / "result_table_2.csv")
    assert R.render_markdown2(df, R.TABLE_2_SPECS) == (
        REPO_RESULTS / "result_table_2.md"
    ).read_text(encoding="utf-8")
    assert R.render_latex2(df, R.TABLE_2_SPECS) == (REPO_RESULTS / "result_table_2.tex").read_text(
        encoding="utf-8"
    )
