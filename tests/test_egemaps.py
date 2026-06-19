"""T4.2 eGeMAPS: deterministic parquet + summary (CI-safe) and openSMILE extraction (guarded)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from ecvol.features.audio import egemaps as eg

# --- pure: parquet + summary (no openSMILE) ----------------------------------


def test_write_egemaps_parquet_sorted_and_deterministic(tmp_path: Path):
    feats = ["F0_mean", "loud_mean"]
    df = pd.DataFrame(
        [
            {"call_id": 2, "F0_mean": 30.0, "loud_mean": 1.0},
            {"call_id": 1, "F0_mean": 25.0, "loud_mean": 0.5},
        ]
    )
    out = tmp_path / "eg.parquet"
    eg.write_egemaps_parquet(df, out, feats)
    t = pq.read_table(out)
    assert t.column("call_id").to_pylist() == [1, 2]  # sorted
    assert t.column("F0_mean").to_pylist() == [25.0, 30.0]
    b1 = out.read_bytes()
    eg.write_egemaps_parquet(df, out, feats)
    assert out.read_bytes() == b1  # byte-identical


def test_summarize():
    df = pd.DataFrame({"call_id": [1, 2], "x": [1.0, 3.0]})
    rows = dict(eg.summarize(df, ["x"]))
    assert rows["n_calls"] == 2
    assert rows["x__mean"] == 2.0


def test_pending_skips_done():
    assert eg._pending([1, 2, 3, 4], {2, 4}) == [1, 3]
    assert eg._pending([1, 2], {1, 2}) == []  # all cached → resume is a no-op


# --- openSMILE extraction (guarded) ------------------------------------------


def test_egemaps_extract_synthetic(tmp_path: Path):
    opensmile = pytest.importorskip("opensmile")
    sf = pytest.importorskip("soundfile")
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    wav = 0.3 * np.sin(2 * np.pi * 150 * t)  # 150 Hz tone
    p = tmp_path / "tone.flac"
    sf.write(p, wav.astype(np.float32), sr)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    row = smile.process_file(str(p)).iloc[0]
    assert len(row) == 88  # eGeMAPSv02 functionals
    # a voiced 150 Hz tone → finite F0 functionals present
    f0 = [c for c in row.index if c.startswith("F0semitone")]
    assert f0 and np.isfinite(row[f0[0]])
