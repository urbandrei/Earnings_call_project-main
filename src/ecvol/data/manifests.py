"""Data-file manifests: record and verify provenance of external files (T0.3).

Every external data file gets `{path, source_url, retrieved_at, sha256, license}`
in `data/manifests/*.json` (DESIGN.md §8.2). Payloads are gitignored; manifests
are committed, so checksums travel with the repo and `ecvol data verify` can
detect missing or corrupted files on any machine.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_CHUNK_SIZE = 1 << 20  # 1 MiB


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)  # POSIX-style, relative to the data root
    source_url: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=1)  # ISO-8601 UTC
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str = Field(min_length=1)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def make_entry(
    file: str | Path,
    root: str | Path,
    source_url: str,
    license: str,
    retrieved_at: str | None = None,
) -> ManifestEntry:
    """Hash an existing file under `root` and build its manifest entry."""
    file = Path(file)
    if retrieved_at is None:
        retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    return ManifestEntry(
        path=file.relative_to(root).as_posix(),
        source_url=source_url,
        retrieved_at=retrieved_at,
        sha256=sha256_file(file),
        license=license,
    )


def write_manifest(entries: list[ManifestEntry], path: str | Path) -> None:
    """Write entries as deterministic JSON (sorted by path, stable key order)."""
    ordered = sorted(entries, key=lambda e: e.path)
    payload = json.dumps([e.model_dump() for e in ordered], indent=2, sort_keys=True)
    Path(path).write_text(payload + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ManifestEntry.model_validate(item) for item in raw]


def verify_manifest(manifest_path: str | Path, root: str | Path) -> list[str]:
    """Check every entry's file exists under `root` with a matching SHA-256.

    Returns a list of human-readable problems; empty means the manifest is clean.
    """
    problems = []
    for entry in load_manifest(manifest_path):
        file = Path(root) / entry.path
        if not file.is_file():
            problems.append(f"{entry.path}: missing (expected under {root})")
            continue
        actual = sha256_file(file)
        if actual != entry.sha256:
            problems.append(
                f"{entry.path}: checksum mismatch (manifest {entry.sha256[:12]}…, "
                f"file {actual[:12]}…)"
            )
    return problems
