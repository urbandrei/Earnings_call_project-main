"""T4.3 emotion2vec+: constants + a guarded real-embed smoke (downloads emotion2vec_plus_large)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ecvol.features.audio import emotion2vec as E

_HAVE = all(importlib.util.find_spec(m) for m in ("funasr", "torch", "torchaudio", "soundfile"))


def test_constants():
    assert E.E2V_DIM == 1024
    assert "emotion2vec" in E.E2V_MODEL


@pytest.mark.skipif(not _HAVE, reason="funasr/gpu stack not installed")
def test_embed_file_synthetic(tmp_path: Path):
    import soundfile as sf

    from ecvol.features.audio.wavlm import SR

    sr = SR
    t = np.linspace(0, 35, sr * 35, endpoint=False)  # 35 s → 2 windows
    p = tmp_path / "tone.flac"
    sf.write(p, (0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr)
    model = E.load_model(device="cpu")
    vec = E.embed_file(model, str(p))
    assert vec.shape == (E.E2V_DIM,) and np.isfinite(vec).all()
