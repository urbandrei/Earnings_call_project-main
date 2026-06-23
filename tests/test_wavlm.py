"""T4.3 WavLM: deterministic vector parquet (CI-safe) + a guarded embed smoke (downloads WavLM)."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from ecvol.features.audio import wavlm as W

# --- pure: vector parquet (no torch) -----------------------------------------


def test_write_vector_parquet_sorted_deterministic(tmp_path: Path):
    rows = [
        {"call_id": 2, "n_windows": 3, "vector": [0.1, 0.2]},
        {"call_id": 1, "n_windows": 1, "vector": [0.3, 0.4]},
    ]
    out = tmp_path / "w.parquet"
    W.write_vector_parquet(rows, out)
    t = pq.read_table(out)
    assert t.column("call_id").to_pylist() == [1, 2]  # sorted
    assert t.column("vector").to_pylist()[0] == [0.3, 0.4]
    b1 = out.read_bytes()
    W.write_vector_parquet(rows, out)
    assert out.read_bytes() == b1  # byte-identical


# --- guarded: real WavLM embed (downloads microsoft/wavlm-large) -------------


@pytest.mark.skipif(
    not all(
        __import__("importlib").util.find_spec(m) for m in ("torch", "transformers", "soundfile")
    ),
    reason="gpu/audio stack not installed",
)
def test_embed_file_synthetic(tmp_path: Path):
    import soundfile as sf

    sr = W.SR
    t = np.linspace(0, 35, sr * 35, endpoint=False)  # 35 s → 2 windows (30 s + 5 s)
    p = tmp_path / "tone.flac"
    sf.write(p, (0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr)
    assert W._n_windows(p) == 2
    fe, model = W.load_model(device="cpu")
    vec = W.embed_file(fe, model, str(p), device="cpu")
    assert vec.shape == (W.WAVLM_DIM,) and np.isfinite(vec).all()
