"""T6R.3 H3/W3 — the 27 Praat features of Qin & Yang (2019) per EC sentence clip.

HTML (Yang et al. 2020) and Sawhney et al. (2020) both take sentence-level Praat features
"in line with" MDRM (Qin & Yang 2019), whose Table lists 27 voice-report statistics. Neither
repository ships an extractor or the feature files, so they are recomputed here with
`praat-parselmouth` from the released `CEO/<speaker>_<turn>_<sentence>.mp3` clips, in the
order Praat's Voice Report prints them:

pitch (mean, sd, min, max) · intensity (mean, sd, min, max) · pulses (number of pulses,
number of periods, mean period, sd period) · voicing (fraction of unvoiced frames, number of
voice breaks, degree of voice breaks) · jitter (local, local absolute, rap, ppq5, ddp) ·
shimmer (local, local dB, apq3, apq5, apq11, dda) · harmonicity (mean HNR).

Clips are aligned to `TextSequence.txt` lines by position after sorting the clip names by
(turn, sentence) — the alignment Qin & Yang describe as noisy. Failed clips (too short for
a pitch track, decode errors) yield NaN rows, imputed downstream with the training mean.
Output: `data/ec/cache/praat27_sentences.parquet` (call_id, sent_idx, 27 columns).
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "pitch_mean", "pitch_sd", "pitch_min", "pitch_max",
    "intensity_mean", "intensity_sd", "intensity_min", "intensity_max",
    "n_pulses", "n_periods", "period_mean", "period_sd",
    "unvoiced_fraction", "n_voice_breaks", "degree_voice_breaks",
    "jitter_local", "jitter_local_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp",
    "shimmer_local", "shimmer_local_db", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
    "shimmer_dda", "hnr_mean",
)  # fmt: skip
CACHE_FILE = "praat27_sentences.parquet"
CLIP_RE = re.compile(r"_(\d+)_(\d+)\.mp3$")
PITCH_FLOOR, PITCH_CEIL = 75.0, 600.0


def clip_order(name: str) -> tuple[int, int]:
    m = CLIP_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (10**9, 10**9)


def voice_report_features(path: str) -> list[float]:
    """The 27 statistics for one clip; NaNs where Praat cannot compute them."""
    import parselmouth
    import soundfile as sf
    from parselmouth.praat import call

    try:
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        snd = parselmouth.Sound(data.mean(axis=1), sr)
        if snd.get_total_duration() < 0.1:
            return [float("nan")] * len(FEATURES)
        pitch = call(snd, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEIL)
        pp = call([snd, pitch], "To PointProcess (cc)")
        report = call(
            [snd, pitch, pp],
            "Voice report",
            0.0,
            0.0,
            PITCH_FLOOR,
            PITCH_CEIL,
            1.3,
            1.6,
            0.03,
            0.45,
        )
        intensity = call(snd, "To Intensity", PITCH_FLOOR, 0.0, "yes")
        vals = {
            "pitch_mean": call(pitch, "Get mean", 0, 0, "Hertz"),
            "pitch_sd": call(pitch, "Get standard deviation", 0, 0, "Hertz"),
            "pitch_min": call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"),
            "pitch_max": call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"),
            "intensity_mean": call(intensity, "Get mean", 0, 0, "energy"),
            "intensity_sd": call(intensity, "Get standard deviation", 0, 0),
            "intensity_min": call(intensity, "Get minimum", 0, 0, "Parabolic"),
            "intensity_max": call(intensity, "Get maximum", 0, 0, "Parabolic"),
            "n_pulses": _report(report, "Number of pulses"),
            "n_periods": _report(report, "Number of periods"),
            "period_mean": _report(report, "Mean period"),
            "period_sd": _report(report, "Standard deviation of period"),
            "unvoiced_fraction": _report(report, "Fraction of locally unvoiced frames"),
            "n_voice_breaks": _report(report, "Number of voice breaks"),
            "degree_voice_breaks": _report(report, "Degree of voice breaks"),
            "jitter_local": _report(report, "Jitter (local)"),
            "jitter_local_abs": _report(report, "Jitter (local, absolute)"),
            "jitter_rap": _report(report, "Jitter (rap)"),
            "jitter_ppq5": _report(report, "Jitter (ppq5)"),
            "jitter_ddp": _report(report, "Jitter (ddp)"),
            "shimmer_local": _report(report, "Shimmer (local)"),
            "shimmer_local_db": _report(report, "Shimmer (local, dB)"),
            "shimmer_apq3": _report(report, "Shimmer (apq3)"),
            "shimmer_apq5": _report(report, "Shimmer (apq5)"),
            "shimmer_apq11": _report(report, "Shimmer (apq11)"),
            "shimmer_dda": _report(report, "Shimmer (dda)"),
            "hnr_mean": _report(report, "Mean harmonics-to-noise ratio"),
        }
        return [float(vals[k]) for k in FEATURES]
    except Exception:  # noqa: BLE001 — a failed clip is a NaN row, by design
        return [float("nan")] * len(FEATURES)


def _report(report: str, label: str) -> float:
    """Parse one line of Praat's Voice Report text (`label: value unit` or `value%`)."""
    for line in report.splitlines():
        s = line.strip()
        if s.startswith(label + ":"):
            token = s.split(":", 1)[1].strip().split()[0].rstrip("%")
            try:
                return float(token)
            except ValueError:
                return float("nan")
    return float("nan")


def ec_clips(root: Path) -> list[tuple[str, int, str]]:
    """(call_id, sent_idx, clip path) for every EC call, clips ordered by (turn, sentence)."""
    base = root / "raw/ec/extracted/ACL19_Release"
    out = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        clips = sorted((folder / "CEO").glob("*.mp3"), key=lambda p: clip_order(p.name))
        out.extend((folder.name, i, str(c)) for i, c in enumerate(clips))
    return out


def extract_ec_praat(root: Path, *, workers: int = 8, log=print) -> Path:
    """All EC clips → cache parquet. Idempotent per call (resumes from the cache)."""
    cache = root / "ec" / "cache" / CACHE_FILE
    done: set[str] = set()
    frames = []
    if cache.is_file():
        prev = pd.read_parquet(cache)
        frames.append(prev)
        done = set(prev["call_id"].unique())
    todo = [c for c in ec_clips(root) if c[0] not in done]
    log(f"  {len(todo)} clips to extract ({len(done)} calls cached)")
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            feats = list(ex.map(voice_report_features, [c[2] for c in todo], chunksize=64))
        new = pd.DataFrame(feats, columns=list(FEATURES))
        new.insert(0, "sent_idx", [c[1] for c in todo])
        new.insert(0, "call_id", [c[0] for c in todo])
        frames.append(new)
    table = pd.concat(frames, ignore_index=True).sort_values(["call_id", "sent_idx"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache, index=False)
    return cache


def sentence_audio(root: Path, n_sentences: dict[str, int]) -> dict[str, np.ndarray]:
    """Per call: [n_sentences, 27] aligned by position to the transcript lines; missing
    clips → NaN rows (imputed by the caller); extra clips dropped."""
    t = pd.read_parquet(root / "ec" / "cache" / CACHE_FILE)
    out = {}
    for cid, g in t.groupby("call_id", sort=False):
        n = n_sentences.get(str(cid))
        if n is None:
            continue
        a = np.full((n, len(FEATURES)), np.nan, dtype=np.float32)
        g = g[g["sent_idx"] < n]
        a[g["sent_idx"].to_numpy()] = g[list(FEATURES)].to_numpy(dtype=np.float32)
        out[str(cid)] = a
    return out
