"""T4.1 audio QC: stderr parsing (CI-safe) + ffmpeg integration (synthetic known-good signals).

`parse_qc`/`_reason` are pure and run everywhere; the ffmpeg-backed tests generate deterministic
signals (silence, clipped tone) and assert the QC metrics — the synthetic half of the
'validated on known-good reference' acceptance (the real Earnings-21 sample is fetched separately).
"""

from pathlib import Path

import pytest

from ecvol.features.audio import qc

_ASTATS = """[Parsed_astats_0 @ 0x1] Channel: 1
[Parsed_astats_0 @ 0x1] Peak level dB: {pk:.6f}
[Parsed_astats_0 @ 0x1] RMS level dB: -20.000000
[Parsed_astats_0 @ 0x1] Peak count: {pc}
[Parsed_astats_0 @ 0x1] Number of samples: {ns}
[Parsed_astats_0 @ 0x1] Overall
[Parsed_astats_0 @ 0x1] Peak level dB: {pk:.6f}
[Parsed_astats_0 @ 0x1] RMS level dB: -20.000000
[Parsed_astats_0 @ 0x1] Peak count: {pc}
[Parsed_astats_0 @ 0x1] Number of samples: {ns}
"""


# --- pure parsing (no ffmpeg) ------------------------------------------------


def test_parse_qc_clipping():
    stderr = _ASTATS.format(pk=-0.05, pc=50, ns=1000)
    out = qc.parse_qc(stderr, duration=60.0)
    assert out["peak_dbfs"] == -0.05
    assert abs(out["clip_ratio"] - 0.05) < 1e-9  # peak >= -0.1 dBFS → clipping counted
    assert out["silence_ratio"] == 0.0


def test_parse_qc_no_clipping_below_threshold():
    stderr = _ASTATS.format(pk=-3.0, pc=50, ns=1000)
    out = qc.parse_qc(stderr, duration=60.0)
    assert out["clip_ratio"] == 0.0  # peak below -0.1 dBFS → not clipping


def test_parse_qc_silence_ratio():
    stderr = _ASTATS.format(pk=-3.0, pc=0, ns=1000)
    stderr += "[silencedetect @ 0x2] silence_end: 30 | silence_duration: 18.0\n"
    stderr += "[silencedetect @ 0x2] silence_end: 60 | silence_duration: 12.0\n"
    out = qc.parse_qc(stderr, duration=60.0)
    assert abs(out["silence_ratio"] - 0.5) < 1e-9  # (18+12)/60


def test_parse_qc_pure_silence_inf_peak():
    out = qc.parse_qc(_ASTATS.format(pk=float("-inf"), pc=0, ns=1000), duration=10.0)
    assert out["peak_dbfs"] == float("-inf") and out["clip_ratio"] == 0.0


def test_reason_codes():
    assert qc._reason(False, {"clip_ratio": 0, "silence_ratio": 0}) == "decode_error"
    assert qc._reason(True, {"clip_ratio": 0.5, "silence_ratio": 0}) == "high_clipping"
    assert qc._reason(True, {"clip_ratio": 0, "silence_ratio": 0.9}) == "mostly_silent"
    assert qc._reason(True, {"clip_ratio": 0, "silence_ratio": 0.1}) == ""


# --- ffmpeg integration (synthetic known-good) -------------------------------

needs_ffmpeg = pytest.mark.skipif(not qc.have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")


def _gen(args, out: Path):
    import subprocess

    subprocess.run(["ffmpeg", "-y", *args, str(out)], capture_output=True, check=True)


@needs_ffmpeg
def test_qc_one_silence(tmp_path: Path):
    src = tmp_path / "sil.wav"
    _gen(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "3"], src)
    row = qc.qc_one(1, src, tmp_path / "out" / "1.flac")
    assert row["decode_ok"] and row["sample_rate"] == 44100
    assert row["silence_ratio"] > 0.9 and row["reason"] == "mostly_silent"
    assert (tmp_path / "out" / "1.flac").is_file()


@needs_ffmpeg
def test_qc_one_clipped_tone_and_resample(tmp_path: Path):
    src = tmp_path / "clip.wav"
    # loud sine → clips at 0 dBFS
    _gen(["-f", "lavfi", "-i", "sine=frequency=440:r=44100:d=3", "-af", "volume=20dB"], src)
    dst = tmp_path / "out" / "2.flac"
    row = qc.qc_one(2, src, dst)
    assert row["decode_ok"] and row["peak_dbfs"] >= -0.1 and row["clip_ratio"] > 0
    # resampled store is 16 kHz mono
    meta = qc.ffprobe(dst)
    assert meta["sample_rate"] == 16000 and meta["channels"] == 1


@needs_ffmpeg
def test_qc_one_missing_file(tmp_path: Path):
    row = qc.qc_one(3, tmp_path / "nope.mp3", tmp_path / "out" / "3.flac")
    assert not row["decode_ok"] and row["reason"] == "missing"
