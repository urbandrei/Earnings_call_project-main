"""T5.1 fusion: gated-fusion block reduction, stacking meta-learner, end-to-end schema."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.eval import stage4
from ecvol.models import fusion

# --- gated fusion block reduction --------------------------------------------


def test_gated_fuse_blocks_shapes_and_norm():
    rng = np.random.default_rng(0)
    b1 = (rng.normal(size=(60, 50)), rng.normal(size=(15, 50)), rng.normal(size=(15, 50)))
    b2 = (rng.normal(size=(60, 8)), rng.normal(size=(15, 8)), rng.normal(size=(15, 8)))
    tr = [b1[0], b2[0]]
    va = [b1[1], b2[1]]
    te = [b1[2], b2[2]]
    Xtr, Xva, Xte = fusion.gated_fuse_blocks(tr, va, te, dim=16)
    # block1 → 16 dims (PCA capped), block2 → 8 dims (rank) = 24
    assert Xtr.shape == (60, 24) and Xte.shape == (15, 24)
    # each modality sub-block is L2-normalized per row
    assert np.allclose(np.linalg.norm(Xtr[:, :16], axis=1), 1.0, atol=1e-6)


def test_gated_fuse_skips_empty_block():
    rng = np.random.default_rng(1)
    tr = [rng.normal(size=(40, 5)), np.zeros((40, 0))]
    va = [rng.normal(size=(10, 5)), np.zeros((10, 0))]
    te = [rng.normal(size=(10, 5)), np.zeros((10, 0))]
    Xtr, _, _ = fusion.gated_fuse_blocks(tr, va, te, dim=4)
    assert Xtr.shape == (40, 4)  # empty block skipped


# --- stacking meta-learner ---------------------------------------------------


def test_stack_recovers_better_base():
    rng = np.random.default_rng(2)
    n = 200
    y = rng.normal(size=n)
    good = y + 0.05 * rng.normal(size=n)  # base 0 ~ perfect
    bad = rng.normal(size=n)  # base 1 ~ noise
    base = np.column_stack([good, bad])
    va, te = slice(0, 140), slice(140, 200)
    pv, pte = fusion.stack_fit_predict(base[va], base[te], y[va])
    # meta-learner should track y via the good base
    assert np.corrcoef(pte, y[te])[0, 1] > 0.9


def test_gated_fusion_fit_predict_runs():
    rng = np.random.default_rng(3)
    blocks = lambda n: [rng.normal(size=(n, 20)), rng.normal(size=(n, 8))]  # noqa: E731
    btr, bva, bte = blocks(80), blocks(20), blocks(20)
    dense = (rng.normal(size=(80, 3)), rng.normal(size=(20, 3)), rng.normal(size=(20, 3)))
    y = rng.normal(size=80)
    pv, pte = fusion.gated_fusion_fit_predict(btr, bva, bte, dense, y, seed=0)
    assert pv.shape == (20,) and pte.shape == (20,) and np.isfinite(pte).all()


# --- end-to-end (synthetic) --------------------------------------------------


def _write_audio(root: Path, n=80, dim=8):
    rng = np.random.default_rng(0)
    d = root / "fincall"
    d.mkdir(parents=True, exist_ok=True)
    ids = [str(c) for c in range(n)]
    eg = pd.DataFrame({"call_id": ids})
    eg["F0semitoneFrom27.5Hz_sma3nz_amean"] = rng.normal(31, 3, n)
    eg["loudness"] = rng.normal(0.5, 0.1, n)
    pq.write_table(pa.Table.from_pandas(eg, preserve_index=False), d / "audio_egemaps.parquet")
    for fn in ("audio_wavlm.parquet", "audio_emotion2vec.parquet"):
        vecs = [list(rng.normal(size=dim)) for _ in range(n)]
        pq.write_table(
            pa.table(
                {
                    "call_id": pa.array([int(i) for i in ids], pa.int64()),
                    "n_windows": pa.array([5] * n, pa.int64()),
                    "vector": pa.array(vecs, pa.list_(pa.float64())),
                }
            ),
            d / fn,
        )


def _eval_frame(n=80):
    rng = np.random.default_rng(3)
    rows, q = [], ["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"]
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


def test_stage4_end_to_end(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    _write_audio(root, n=80)
    monkeypatch.setattr(
        stage4,
        "build_text_matrix",
        lambda r, ds: (
            pd.DataFrame(
                {
                    "call_id": [str(c) for c in range(80)],
                    "te0": np.zeros(80),
                    "te1": np.zeros(80),
                    "tf": np.zeros(80),
                }
            ),
            ["te0", "te1"],
            ["tf"],
        ),
    )
    monkeypatch.setattr(stage4.E, "load_eval_frame", lambda r, ds: _eval_frame())
    splits = root / "splits"
    splits.mkdir()
    pd.DataFrame(
        [
            {"call_id": str(c), "split": "train" if c < 56 else ("val" if c < 68 else "test")}
            for c in range(80)
        ]
    ).to_csv(splits / "fincall_temporal.csv", index=False)
    rows = stage4.evaluate_stage4(root, "fincall", seeds=(0,))
    t = pd.DataFrame(rows)
    assert len(t) > 0
    assert {"gated_fusion", "gated_fusion_pastvol", "stack_fusion", "stack_fusion_pastvol"} <= set(
        t["model"]
    )
    assert {"dm_p_vs_stage1", "dm_p_vs_stage2", "dm_p_vs_stage3"} <= set(t.columns)
