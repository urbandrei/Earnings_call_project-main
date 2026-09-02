"""T8.1 — released derived-feature archives (DESIGN §12; LICENSE-DATA.md).

One zip per corpus holding exactly the files LICENSE-DATA.md says we release: targets
(both conventions, both anchors), splits, per-call pooled features, the identity and
timing tables, and the data manifests that let a reader check every member's SHA-256
against this repository. Raw transcripts, audio, prices and chunk-level text are never
members. Archives are byte-deterministic (fixed member timestamps, sorted members,
DEFLATE level 9) so `manifests/release.json` — committed — pins their SHA-256 and a
re-run on another machine can be compared byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

RELEASE_PREFIX = "ecvol-bench"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)  # zip epoch: archives do not depend on build time

# corpus → (licence of the released derivatives, data-root-relative globs)
RELEASED: dict[str, tuple[str, tuple[str, ...]]] = {
    "fincall": (
        "Apache-2.0",
        (
            "fincall/targets*.parquet",
            "fincall/text_*.parquet",
            "fincall/audio_*.parquet",
            "identity/fincall_identity.csv",
            "splits/fincall_*.csv",
            "coverage/fincall_timing.csv",
            "manifests/fincall_*.json",
        ),
    ),
    "maec": (
        "CC-BY-SA-4.0",
        (
            "maec/targets*.parquet",
            "maec/text_*.parquet",
            "splits/maec_*.csv",
            "coverage/maec_timing.csv",
            "manifests/maec_*.json",
        ),
    ),
    "earnings25": (
        "CC-BY-4.0",
        (
            "earnings25/targets*.parquet",
            "earnings25/text_*.parquet",
            "earnings25/audio_*.parquet",
            "splits/earnings25_*.csv",
            "coverage/earnings25_inventory.csv",
            "coverage/earnings25_timing.csv",
            "coverage/earnings25_call_times.csv",
            "coverage/earnings25_ingest_report.csv",
            "manifests/earnings25_*.json",
        ),
    ),
    "ec": (
        "terms pending (LICENSE-DATA.md open item 1); pooled per-call features only",
        (
            "ec/targets*.parquet",
            "ec/text_*.parquet",
            "splits/ec_*.csv",
            "manifests/ec_*.json",
        ),
    ),
}
DOCS = ("LICENSE-DATA.md", "REPRODUCE.md")  # repo-root files copied into every archive


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def members(root: Path, corpus: str) -> list[Path]:
    """Data-root-relative member paths for `corpus`, sorted, existing files only."""
    _, globs = RELEASED[corpus]
    found = {p.relative_to(root) for g in globs for p in root.glob(g) if p.is_file()}
    return sorted(found, key=lambda p: p.as_posix())


def _add(z: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)


def build_archive(root: Path, corpus: str, out_dir: Path, tag: str, repo_root: Path) -> dict:
    """Write `<out_dir>/ecvol-bench_<corpus>_<tag>.zip`; return its manifest record."""
    licence, _ = RELEASED[corpus]
    rels = members(root, corpus)
    if not rels:
        raise FileNotFoundError(f"{corpus}: no releasable files under {root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{RELEASE_PREFIX}_{corpus}_{tag}.zip"
    recorded = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in rels:
            data = (root / rel).read_bytes()
            _add(z, rel.as_posix(), data)
            recorded.append({"path": rel.as_posix(), "sha256": _sha256(data), "bytes": len(data)})
        for doc in DOCS:
            src = repo_root / doc
            if src.is_file():
                _add(z, doc, src.read_bytes())
    blob = archive.read_bytes()
    return {
        "corpus": corpus,
        "archive": archive.name,
        "license": licence,
        "sha256": _sha256(blob),
        "bytes": len(blob),
        "members": recorded,
    }


def build_release(
    root: Path, out_dir: Path, tag: str, *, corpora=None, repo_root: Path | None = None
) -> Path:
    """Build every corpus archive and write the committed `manifests/release.json`."""
    repo_root = repo_root if repo_root is not None else root.parent
    records = [
        build_archive(root, c, out_dir, tag, repo_root) for c in (corpora or tuple(RELEASED))
    ]
    manifest = root / "manifests" / "release.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"tag": tag, "archives": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
