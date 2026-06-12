"""Run artifacts: every run writes artifacts/runs/<run_id>/ (T0.3).

Per DESIGN.md §8.2 a run artifact contains the resolved config (config.yaml),
run metadata with config hash, git SHA, seed list, and environment fingerprint
(run.json), and a metrics parquet. Metrics writing is deterministic: identical
metric rows produce a bit-identical metrics.parquet, which is what makes the
byte-identical regeneration tests possible.

Run-ID scheme: `<UTC timestamp>-<config-hash prefix>` — sortable by start
time, and the hash prefix ties the directory to its exact resolved config.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.config import ExperimentConfig, config_hash, dump_config


def new_run_id(cfg: ExperimentConfig, now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(UTC)
    return f"{now:%Y%m%dT%H%M%SZ}-{config_hash(cfg)[:8]}"


def git_info() -> dict[str, Any] | None:
    """Current commit SHA and dirty flag, or None outside a git repo."""
    try:
        sha = _run_capture(["git", "rev-parse", "HEAD"])
        status = _run_capture(["git", "status", "--porcelain"])
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"sha": sha, "dirty": bool(status)}


def env_fingerprint(lockfile: str | Path = "uv.lock") -> dict[str, Any]:
    """Environment identity: interpreter, platform, lockfile hash, GPU/driver."""
    lockfile = Path(lockfile)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "lockfile_sha256": (
            hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.is_file() else None
        ),
        "gpus": _nvidia_gpus(),
    }


def write_run(
    cfg: ExperimentConfig,
    metrics: list[dict[str, Any]],
    artifacts_dir: str | Path = "artifacts",
    run_id: str | None = None,
) -> Path:
    """Write a complete run artifact directory and return its path.

    Refuses to overwrite an existing run directory — reruns get new IDs;
    bit-identical regeneration is asserted on the metrics bytes, not by
    rewriting in place.
    """
    _validate_metrics(metrics)  # before any filesystem writes — no partial run dirs
    if run_id is None:
        run_id = new_run_id(cfg)
    run_dir = Path(artifacts_dir) / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    (run_dir / "config.yaml").write_text(dump_config(cfg), encoding="utf-8")
    write_metrics(metrics, run_dir / "metrics.parquet")
    meta = {
        "run_id": run_id,
        "config_hash": config_hash(cfg),
        "git": git_info(),
        "seeds": cfg.seeds,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "env": env_fingerprint(),
    }
    (run_dir / "run.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def write_metrics(metrics: list[dict[str, Any]], path: str | Path) -> None:
    """Write metric rows as parquet, deterministically (same rows = same bytes)."""
    _validate_metrics(metrics)
    table = pa.Table.from_pylist(metrics)
    pq.write_table(table, path, compression="none", store_schema=True)


def _validate_metrics(metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        raise ValueError("refusing to write an empty metrics table")
    columns = list(metrics[0])
    for i, row in enumerate(metrics):
        if list(row) != columns:
            raise ValueError(
                f"metrics row {i} columns {list(row)} differ from row 0 columns {columns}"
            )


def read_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(Path(run_dir) / "metrics.parquet").to_pylist()


def _run_capture(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=30
    ).stdout.strip()


def _nvidia_gpus() -> list[str] | None:
    """`name, driver_version` per GPU via nvidia-smi; None if unavailable."""
    try:
        out = _run_capture(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]
