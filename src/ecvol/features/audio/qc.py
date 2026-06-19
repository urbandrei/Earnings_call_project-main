"""Audio QC + 16 kHz mono resampled store (T4.1, FinCall-only — MAEC ships no audio).

One ffmpeg pass per call both (a) measures QC on the decoded source — peak/RMS, clipping,
silence ratio, decode success — via the `astats` + `silencedetect` filters (stderr), and
(b) writes a 16 kHz mono FLAC to the resampled store reused by eGeMAPS / WavLM / emotion2vec.
QC uses only the installed ffmpeg/ffprobe (no Python audio deps); corrupt files are flagged
with reason codes, never dropped. Output: committed `data/coverage/fincall_audio_qc.csv`
(per-call) + summary; store under `data/raw/audio_16k/fincall/` (gitignored, manifest-tracked).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

CLIP_DBFS = -0.1  # peak at/above this counts as clipping-prone
HIGH_CLIP_RATIO = 1e-3  # >0.1% of samples at peak → flag
MOSTLY_SILENT = 0.5  # >50% silence → flag
SILENCE_NOISE_DB = -30  # silencedetect threshold
SILENCE_MIN_DUR = 0.5  # seconds

_PEAK = re.compile(r"Peak level dB:\s*(-?inf|-?\d+\.?\d*)")
_RMS = re.compile(r"RMS level dB:\s*(-?inf|-?\d+\.?\d*)")
_PEAKCNT = re.compile(r"Peak count:\s*(\d+)")
_NSAMP = re.compile(r"Number of samples:\s*(\d+)")
_SILDUR = re.compile(r"silence_duration:\s*(\d+\.?\d*)")


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ffprobe(path: Path) -> dict:
    """Container/stream metadata (duration, sample_rate, channels, codec, bit_rate)."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return {}
    data = json.loads(out.stdout or "{}")
    astream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})
    return {
        "codec": astream.get("codec_name", ""),
        "sample_rate": int(astream.get("sample_rate", 0) or 0),
        "channels": int(astream.get("channels", 0) or 0),
        "bit_rate": int(fmt.get("bit_rate", 0) or 0),
        "duration_sec": float(fmt.get("duration", 0.0) or 0.0),
    }


def _qc_resample(src: Path, dst: Path) -> tuple[int, str]:
    """One pass: astats+silencedetect on the source, write 16 kHz mono FLAC (stripped metadata)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    af = (
        f"astats=metadata=1:reset=0,"
        f"silencedetect=noise={SILENCE_NOISE_DB}dB:duration={SILENCE_MIN_DUR}"
    )
    out = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            af,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-map_metadata",
            "-1",
            "-sample_fmt",
            "s16",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    return out.returncode, out.stderr


def parse_qc(stderr: str, duration: float) -> dict:
    """Peak/RMS/clip-ratio/silence-ratio from ffmpeg astats+silencedetect stderr."""
    # astats prints per-channel then an Overall block; take the last (Overall) values.
    peaks = _PEAK.findall(stderr)
    rmss = _RMS.findall(stderr)
    peak_db = float(peaks[-1]) if peaks and peaks[-1] != "-inf" else float("-inf")
    rms_db = float(rmss[-1]) if rmss and rmss[-1] != "-inf" else float("-inf")
    pc = _PEAKCNT.findall(stderr)
    ns = _NSAMP.findall(stderr)
    clip_ratio = 0.0
    if peak_db >= CLIP_DBFS and pc and ns and int(ns[-1]) > 0:
        clip_ratio = int(pc[-1]) / int(ns[-1])
    sil = sum(float(x) for x in _SILDUR.findall(stderr))
    silence_ratio = (sil / duration) if duration > 0 else 0.0
    return {
        "peak_dbfs": peak_db,
        "rms_dbfs": rms_db,
        "clip_ratio": min(clip_ratio, 1.0),
        "silence_ratio": min(silence_ratio, 1.0),
    }


def _reason(decode_ok: bool, qc: dict) -> str:
    if not decode_ok:
        return "decode_error"
    if qc["clip_ratio"] > HIGH_CLIP_RATIO:
        return "high_clipping"
    if qc["silence_ratio"] > MOSTLY_SILENT:
        return "mostly_silent"
    return ""


def qc_one(call_id, src: Path, dst: Path) -> dict:
    """QC + resample one file → a report row (never raises; flags failures)."""
    if not src.is_file():
        return {
            "call_id": call_id,
            "decode_ok": False,
            "reason": "missing",
            "codec": "",
            "sample_rate": 0,
            "channels": 0,
            "bit_rate": 0,
            "duration_sec": 0.0,
            "peak_dbfs": float("nan"),
            "rms_dbfs": float("nan"),
            "clip_ratio": float("nan"),
            "silence_ratio": float("nan"),
        }
    meta = ffprobe(src)
    rc, stderr = _qc_resample(src, dst)
    decode_ok = rc == 0 and dst.is_file()
    qc = parse_qc(stderr, meta.get("duration_sec", 0.0))
    return {
        "call_id": call_id,
        "decode_ok": decode_ok,
        "codec": meta.get("codec", ""),
        "sample_rate": meta.get("sample_rate", 0),
        "channels": meta.get("channels", 0),
        "bit_rate": meta.get("bit_rate", 0),
        "duration_sec": meta.get("duration_sec", 0.0),
        **qc,
        "reason": _reason(decode_ok, qc),
    }


@dataclass
class QCSummary:
    n: int
    decoded: int
    flagged: dict[str, int]
    store_dir: str


def build_qc(root: Path, *, limit: int | None = None, workers: int = 8) -> QCSummary:
    """QC + resample every FinCall call with audio; write report + manifest (T4.1)."""
    import pandas as pd

    from ecvol.data.calls import write_metric_csv

    calls = pd.read_parquet(
        root / "fincall" / "calls.parquet", columns=["call_id", "audio_path", "audio_exists"]
    )
    calls = calls[calls["audio_exists"]].reset_index(drop=True)
    if limit is not None:
        calls = calls.head(limit)
    store = root / "raw" / "audio_16k" / "fincall"

    def work(row):
        src = root / row.audio_path
        dst = store / f"{row.call_id}.flac"
        return qc_one(row.call_id, src, dst)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, calls.itertuples(index=False)))
    rows.sort(key=lambda r: r["call_id"])
    df = pd.DataFrame(rows)

    cov = root / "coverage"
    cov.mkdir(parents=True, exist_ok=True)
    df.to_csv(cov / "fincall_audio_qc.csv", index=False, lineterminator="\n")
    flagged = {r: int((df["reason"] == r).sum()) for r in df["reason"].unique() if r}
    summary_rows = [
        ("n", len(df)),
        ("decoded", int(df["decode_ok"].sum())),
        ("median_duration_sec", float(df["duration_sec"].median())),
        ("median_silence_ratio", float(df["silence_ratio"].median())),
        ("median_peak_dbfs", float(df["peak_dbfs"].replace(float("-inf"), float("nan")).median())),
    ] + [(f"flag:{k}", v) for k, v in sorted(flagged.items())]
    # 16 kHz store is a deterministic regenerable cache (mp3 + fixed ffmpeg args), like the
    # T3.2 embedding cache — gitignored, not per-file manifested; the QC CSV is the audited report.
    n_flac = sum(1 for _ in store.glob("*.flac")) if store.exists() else 0
    summary_rows.append(("resampled_flac_files", n_flac))
    write_metric_csv(summary_rows, cov / "fincall_audio_qc_summary.csv")
    return QCSummary(len(df), int(df["decode_ok"].sum()), flagged, str(store))


EARNINGS21_REPO = "Revai/earnings21"  # Rev.com, CC-BY-SA-4.0 (DESIGN [D5]) — QC reference only


def validate_earnings21(root: Path, *, n: int = 3) -> list[dict]:
    """Fetch the first `n` Earnings-21 wavs (known-good reference) and QC them (T4.1 acceptance).

    Writes `data/coverage/earnings21_qc_validation.csv`. Requires huggingface_hub (gpu group).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    wavs = sorted(
        f for f in list_repo_files(EARNINGS21_REPO, repo_type="dataset") if f.endswith(".wav")
    )
    ref = root / "raw" / "ref" / "earnings21"
    ref.mkdir(parents=True, exist_ok=True)
    rows = []
    for fn in wavs[:n]:
        local = hf_hub_download(EARNINGS21_REPO, fn, repo_type="dataset", local_dir=str(ref))
        row = qc_one(Path(fn).stem, Path(local), ref / f"{Path(fn).stem}.16k.flac")
        rows.append(row)
    cov = root / "coverage"
    cov.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        cov / "earnings21_qc_validation.csv", index=False, lineterminator="\n"
    )
    return rows
