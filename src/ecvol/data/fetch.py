"""Dataset acquisition: local, checksummed mirrors of FinCall-Surprise and MAEC (T1.1).

Sources are pinned to upstream commit SHAs so a re-fetch is reproducible. Bulk
payloads (FinCall-Surprise audio/slides) come from Google Drive and land under
`data/raw/` (a junction onto a bigger disk — see DECISIONS.md 2026-06-12).
Every mirrored file gets a manifest entry in `data/manifests/<dataset>.json`;
`ecvol data verify` checks them from then on.

MAEC's 59 GB high-level MFCC archive is link-rotted upstream (404) and is NOT
part of the mirror — documented in DECISIONS.md 2026-06-12.
"""

import json
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import requests

from ecvol.data.manifests import ManifestEntry, make_entry, write_manifest

# --- FinCall-Surprise (D1) ---------------------------------------------------

FINCALL_LICENSE = "Apache-2.0"
FINCALL_REPO = "Tizzzzy/FinCall-Surprise"
FINCALL_COMMIT = "36bb82d2298fe61de9d127c14ebda2c7be454136"  # main @ 2026-02-18
FINCALL_REPO_FILES = (
    "transcripts_2019.json",
    "transcripts_2020.json",
    "transcripts_2021.json",
    "LICENSE",
)
FINCALL_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1gdoRW2jhHQzabyzuCdJMGECUhw5eZ2-6"
FINCALL_DRIVE_FILES = {  # name -> Drive file id (folder listing, 2026-06-12)
    "mp3_2019.zip": "1e2CcQZDZVETs6Y1hLgXK-AB9Pj1YgX0b",  # ~18 GB
    "mp3_2020.zip": "1WQlt3txTrUmdTb4t6v1523_-bUVaqS66",  # ~14 GB
    "mp3_2021.zip": "1BKABLqdC8RQrpRRhvsDf6vyfFKw0Ojzd",  # ~20 GB
    "ppt_2019.zip": "1y1x8QwyuC4OIWVljtrbr_tp-OrE65W03",  # ~1.7 GB
    "ppt_2020.zip": "1eWxpGThGfSNIy6mO32M8EFn-mGJcNoGX",  # ~1.4 GB
    "ppt_2021.zip": "1HmqGi0PgJsMxQoYnuTKPBFXX4BAmKo5A",  # ~2.3 GB
}

# --- MAEC (D2) ----------------------------------------------------------------

MAEC_LICENSE = "CC-BY-SA-4.0"
MAEC_REPO = (
    "Earnings-Call-Dataset/"
    "MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction"
)
MAEC_COMMIT = "65a109f5b1a8cb4c96e8337b749ce3db41f2c210"  # master @ 2022-01-11


def _download(url: str, dest: Path, headers: dict[str, str] | None = None) -> None:
    """Stream `url` to `dest` atomically (temp file + rename). Skips if `dest` exists."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
        resp.raise_for_status()
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    part.replace(dest)


def _download_drive(file_id: str, dest: Path) -> None:
    """Fetch one Google Drive file via gdown (confirm tokens, resumable). Idempotent."""
    if dest.exists():
        return
    import gdown  # slow import; keep it off the CLI's critical path

    dest.parent.mkdir(parents=True, exist_ok=True)
    out = gdown.download(id=file_id, output=str(dest), resume=True)
    if out is None:
        raise RuntimeError(
            f"gdown failed for Drive id {file_id} -> {dest.name} "
            "(common cause: Drive download quota; retry later)"
        )


def _extract_zip(archive: Path, dest_dir: Path) -> None:
    """Extract `archive` into `dest_dir` atomically (temp dir + rename). Idempotent."""
    if dest_dir.is_dir():
        return
    partial = dest_dir.with_name(dest_dir.name + ".partial")
    if partial.is_dir():
        shutil.rmtree(partial)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(partial)
    # Zips made from a folder usually wrap everything in one top-level dir; unwrap it.
    entries = list(partial.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        entries[0].replace(dest_dir)
        partial.rmdir()
    else:
        partial.replace(dest_dir)


def _extract_tarball(archive: Path, dest_dir: Path) -> None:
    """Extract a GitHub codeload tarball, stripping its single top-level dir. Idempotent."""
    if dest_dir.is_dir():
        return
    partial = dest_dir.with_name(dest_dir.name + ".partial")
    if partial.is_dir():
        shutil.rmtree(partial)
    with tarfile.open(archive) as tf:
        tf.extractall(partial, filter="data")
    (top,) = list(partial.iterdir())  # codeload tarballs have exactly one root dir
    top.replace(dest_dir)
    partial.rmdir()


def _walk_files(root: Path) -> Iterator[Path]:
    yield from (p for p in sorted(root.rglob("*")) if p.is_file())


def _entries_for_tree(tree: Path, data_root: Path, source_url: str, license: str):
    for file in _walk_files(tree):
        yield make_entry(file, data_root, source_url=source_url, license=license)


def fetch_fincall(data_root: Path, *, skip_drive: bool = False) -> Path:
    """Mirror FinCall-Surprise into `data_root/raw/fincall/`; write its manifest."""
    raw = data_root / "raw" / "fincall"
    entries: list[ManifestEntry] = []

    for name in FINCALL_REPO_FILES:
        url = f"https://raw.githubusercontent.com/{FINCALL_REPO}/{FINCALL_COMMIT}/{name}"
        _download(url, raw / name)
        entries.append(make_entry(raw / name, data_root, source_url=url, license=FINCALL_LICENSE))

    if not skip_drive:
        for name, file_id in FINCALL_DRIVE_FILES.items():
            archive, extracted = raw / name, raw / Path(name).stem
            _download_drive(file_id, archive)
            url = f"https://drive.google.com/uc?id={file_id} (folder {FINCALL_DRIVE_FOLDER})"
            entries.append(make_entry(archive, data_root, source_url=url, license=FINCALL_LICENSE))
            _extract_zip(archive, extracted)
            entries.extend(_entries_for_tree(extracted, data_root, url, FINCALL_LICENSE))

    manifest = data_root / "manifests" / "fincall.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(entries, manifest)
    return manifest


def fetch_maec(data_root: Path) -> Path:
    """Mirror the MAEC repo (transcripts + low-level features) into `data_root/raw/maec/`."""
    raw = data_root / "raw" / "maec"
    url = f"https://codeload.github.com/{MAEC_REPO}/tar.gz/{MAEC_COMMIT}"
    archive = raw / f"maec-{MAEC_COMMIT[:8]}.tar.gz"
    _download(url, archive)
    _extract_tarball(archive, raw / "repo")

    entries = [make_entry(archive, data_root, source_url=url, license=MAEC_LICENSE)]
    entries.extend(_entries_for_tree(raw / "repo", data_root, url, MAEC_LICENSE))
    manifest = data_root / "manifests" / "maec.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(entries, manifest)
    return manifest


def count_fincall_calls(raw_fincall: Path) -> dict[str, int]:
    """Calls per transcript year-file, plus mirrored mp3/ppt file counts."""
    counts: dict[str, int] = {}
    for f in sorted(raw_fincall.glob("transcripts_*.json")):
        counts[f.stem] = len(json.loads(f.read_text(encoding="utf-8")))
    counts["calls_total"] = sum(counts.values())
    for kind in ("mp3", "ppt"):
        dirs = sorted(raw_fincall.glob(f"{kind}_*"))
        if dirs:
            counts[f"{kind}_files"] = sum(1 for d in dirs if d.is_dir() for _ in _walk_files(d))
    return counts


def count_maec_calls(raw_maec: Path) -> dict[str, int]:
    """Call folders (YYYYMMDD_TICKER) in the mirrored MAEC_Dataset."""
    dataset = raw_maec / "repo" / "MAEC_Dataset"
    folders = [d for d in dataset.iterdir() if d.is_dir()] if dataset.is_dir() else []
    return {
        "calls_total": len(folders),
        "with_text": sum(1 for d in folders if (d / "text.txt").is_file()),
        "with_features": sum(1 for d in folders if (d / "features.csv").is_file()),
    }


def spotcheck_audio(raw_root: Path, n: int, seed: int) -> list[str]:
    """Decode `n` seeded-random audio files end-to-end with ffmpeg.

    Returns a list of problems; empty means all sampled files decode cleanly.
    """
    import random

    files = [p for p in _walk_files(raw_root) if p.suffix.lower() == ".mp3"]
    if not files:
        return [f"no .mp3 files found under {raw_root}"]
    sample = random.Random(seed).sample(files, min(n, len(files)))
    problems = []
    for file in sample:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(file), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or proc.stderr.strip():
            detail = proc.stderr.strip().splitlines()[:1]
            problems.append(f"{file.name}: decode error {detail}")
    return problems
