"""T4.4 Stage-3 audio: feature assembly, end-to-end schema, probe/gender, Table-3 render."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.eval import audio_eval, stage3
from ecvol.eval import report as R
from ecvol.features.audio import assemble


def _write_audio(root: Path, n=80, dim=8):
    rng = np.random.default_rng(0)
    d = root / "fincall"
    d.mkdir(parents=True, exist_ok=True)
    ids = [str(c) for c in range(n)]
    # eGeMAPS: scalar cols incl. the F0 proxy
    eg = {"call_id": ids, assemble.EGEMAPS_PARQUET and "x": 0}  # placeholder, replaced below
    eg = pd.DataFrame({"call_id": ids})
    eg["F0semitoneFrom27.5Hz_sma3nz_amean"] = rng.normal(31, 3, n)
    eg["loudness"] = rng.normal(0.5, 0.1, n)
    pq.write_table(pa.Table.from_pandas(eg, preserve_index=False), d / "audio_egemaps.parquet")
    for fn in ("audio_wavlm.parquet", "audio_emotion2vec.parquet"):
        vecs = [list(rng.normal(size=dim)) for _ in range(n)]
        t = pa.table(
            {
                "call_id": pa.array([int(i) for i in ids], pa.int64()),
                "n_windows": pa.array([5] * n, pa.int64()),
                "vector": pa.array(vecs, pa.list_(pa.float64())),
            }
        )
        pq.write_table(t, d / fn)


def test_load_audio_blocks(tmp_path: Path):
    root = tmp_path / "data"
    _write_audio(root)
    df, blocks = assemble.load_audio_blocks(root)
    assert set(blocks) == {"egemaps", "wavlm", "emotion2vec"}
    assert len(blocks["wavlm"]) == 8 and len(blocks["emotion2vec"]) == 8
    assert "F0semitoneFrom27.5Hz_sma3nz_amean" in blocks["egemaps"]
    assert len(df) == 80


def test_table3_label():
    assert R._table3_label("ridge_wavlm_audio") == "ridge WavLM"
    assert R._table3_label("mlp_wavlm_egemaps_audio_pastvol") == "mlp WavLM+eGeMAPS +vol"
    assert R._table3_label("ridge_wavlm_text_audio_pastvol") == "ridge WavLM+text +vol"


def test_render_table3_star(tmp_path: Path):
    rows = []
    for model in R.TABLE3_MODEL_ORDER:
        for h in R.HORIZONS:
            rows.append(
                {
                    "dataset": "fincall",
                    "split": "temporal",
                    "target": "dv",
                    "horizon": h,
                    "model": model,
                    "segment": "test",
                    "n": 50,
                    "mse": 0.4,
                    "mae": 0.3,
                    "r2_oos": 0.1,
                    "spearman_q": 0.2,
                    "seed_std": 0.0,
                    "dm_p_vs_persistence": 0.2,
                    "dm_p_vs_har": 0.2,
                    "dm_p_vs_stage1": 0.01 if model.startswith("ridge_wavlm_audio") else 0.5,
                    "dm_p_vs_stage2": 0.3,
                }
            )
    df = pd.DataFrame(rows)
    spec = R.TableSpec("fincall", "temporal", "dv", "r2_oos")
    _, body, _ = R.build_table3(df, spec)
    labels = [r[0] for r in body]
    assert "ridge WavLM" in labels
    row = body[labels.index("ridge WavLM")]
    assert row[1].endswith("*")  # DM-significant vs Stage-1
    md = R.render_markdown3(df, [spec])
    assert "Result Table 3" in md and "WavLM" in md


# --- end-to-end (synthetic) --------------------------------------------------


def _eval_frame(n=80):
    rng = np.random.default_rng(3)
    rows = []
    q = ["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"]
    for c in range(n):
        v = float(rng.normal(-4, 0.5))
        for h in (3, 7, 15, 30):
            rows.append(
                {
                    "call_id": str(c),
                    "ticker": f"T{c % 10}",
                    "horizon": h,
                    "as_of": q[c % 4],
                    "v_pre": v,
                    "v_post": v + float(rng.normal(0, 0.3)),
                    "delta_v": float(rng.normal(0, 0.3)),
                    "rv_daily": abs(v),
                    "rv_weekly": abs(v),
                    "rv_monthly": abs(v),
                    "n_turns": 20,
                    "n_chars": 5000,
                }
            )
    return pd.DataFrame(rows)


def test_stage3_end_to_end(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    _write_audio(root, n=80)
    # minimal text matrix so the fusion preview + stage2 ref resolve
    (root / "fincall" / "calls.parquet")  # not needed; build_text_matrix patched
    monkeypatch.setattr(
        stage3,
        "build_text_matrix",
        lambda r, ds: (
            pd.DataFrame(
                {"call_id": [str(c) for c in range(80)], "te0": np.zeros(80), "tf": np.zeros(80)}
            ),
            ["te0"],
            ["tf"],
        ),
    )
    monkeypatch.setattr(stage3.E, "load_eval_frame", lambda r, ds: _eval_frame())
    splits = root / "splits"
    splits.mkdir()
    pd.DataFrame(
        [
            {"call_id": str(c), "split": "train" if c < 56 else ("val" if c < 68 else "test")}
            for c in range(80)
        ]
    ).to_csv(splits / "fincall_temporal.csv", index=False)
    rows = stage3.evaluate_stage3(root, "fincall", seeds=(0,))
    t = pd.DataFrame(rows)
    assert len(t) > 0
    assert {"dm_p_vs_stage1", "dm_p_vs_stage2"} <= set(t.columns)
    assert any(m.startswith("ridge_wavlm_") for m in t["model"])


def test_probe_and_gender(tmp_path: Path):
    root = tmp_path / "data"
    _write_audio(root, n=60)
    # calls.parquet with tickers (repeat so each has >=2 calls)
    ids = [str(c) for c in range(60)]
    calls = pd.DataFrame({"call_id": ids, "ticker": [f"T{c % 12}" for c in range(60)]})
    pq.write_table(
        pa.Table.from_pandas(calls, preserve_index=False), root / "fincall" / "calls.parquet"
    )
    probe = audio_eval.identity_probe(root)
    assert set(probe["embedding"]) == {"wavlm", "emotion2vec"}
    assert (probe["probe_accuracy"] >= 0).all()


def test_audio_shuffle_control(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    _write_audio(root, n=80)
    monkeypatch.setattr(audio_eval.E, "load_eval_frame", lambda r, ds: _eval_frame())
    splits = root / "splits"
    splits.mkdir()
    for scheme in ("temporal", "ticker_disjoint"):
        pd.DataFrame(
            [
                {"call_id": str(c), "split": "train" if c < 56 else ("val" if c < 68 else "test")}
                for c in range(80)
            ]
        ).to_csv(splits / f"fincall_{scheme}.csv", index=False)
    sh = audio_eval.audio_shuffle_control(root)
    assert set(sh["condition"]) == {"real", "within_shuffle", "global_shuffle"}
    assert (sh["split"].isin(["temporal", "ticker_disjoint"])).all()
