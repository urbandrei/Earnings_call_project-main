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
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from ecvol.config import (
    CommandConfig,
    ConfigError,
    ExperimentConfig,
    config_hash,
    dump_config,
    load_command_config,
)


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


# --- Command-level run manifests (T9.3; DECISIONS 2026-08-26) -----------------
# The result-producing CLI commands are option-driven and each writes several
# data/results/*.csv files, so their provenance record is one run directory per
# invocation: run.json holds the resolved CommandConfig and its hash, git SHA,
# environment fingerprint, and the SHA-256 of every output file (paths relative
# to the data root). `ecvol report` refuses to render a result CSV whose current
# bytes match no manifest; CI asserts the same for every committed CSV.

COMMAND_OUTPUTS: dict[str, tuple[str, ...]] = {
    "evaluate": ("results/result_table_1.csv",),
    "evaluate-text": ("results/result_table_2.csv",),
    "controls": ("results/result_controls.csv", "results/controls_probe.csv"),
    "evaluate-audio": (
        "results/result_table_3.csv",
        "results/audio_probe.csv",
        "results/audio_gender.csv",
        "results/audio_shuffle.csv",
    ),
    "evaluate-fusion": ("results/result_table_4_fusion.csv",),
    "timing-sensitivity": ("results/timing_sensitivity.csv",),
    "audit-substrate": ("results/result_table_5r.csv", "results/result_table_5r_labels.csv"),
    "reproduce-html": ("results/result_table_6r.csv",),
    "reproduce-scss": ("results/result_table_6r_scss.csv",),
    "audit-code": ("results/code_availability.csv",),
    "evaluate-audio-earnings25": (
        "results/result_table_3_earnings25.csv",
        "results/audio_shuffle_earnings25.csv",
        "results/audio_strata_earnings25.csv",
    ),
    "grid": ("results/result_table_4.csv", "results/result_table_4_peryear.csv"),
}


class ProvenanceError(RuntimeError):
    """A result file's bytes match no committed run manifest."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_command_config(
    command: str,
    config_path: str | Path | None = None,
    seeds: str | None = None,
    configs_dir: str | Path = "configs",
) -> CommandConfig:
    """The committed `configs/<command>.yaml`, with an optional `--seeds` override."""
    path = Path(config_path) if config_path is not None else Path(configs_dir) / f"{command}.yaml"
    cfg = load_command_config(path, command)
    if seeds is not None:
        parsed = [int(s) for s in seeds.split(",") if s.strip()]
        try:
            cfg = CommandConfig.model_validate({**cfg.model_dump(), "seeds": parsed})
        except ValidationError as exc:
            raise ConfigError(f"--seeds {seeds!r}: {exc.errors()[0]['msg']}") from exc
    return cfg


def write_command_run(
    cfg: CommandConfig,
    root: str | Path,
    outputs: Iterable[str] | None = None,
    artifacts_dir: str | Path = "artifacts",
    provenance: str = "run",
    now: datetime | None = None,
) -> Path:
    """Write artifacts/runs/<run_id>/run.json for one command invocation.

    `outputs` are data-root-relative paths (default: COMMAND_OUTPUTS[command]);
    every one must exist. `provenance` is "run" for a live invocation and
    "backfill" for a manifest written after the fact over pre-existing files.
    """
    root = Path(root)
    if outputs is None:
        outputs = COMMAND_OUTPUTS[cfg.command]
    recorded: dict[str, dict[str, Any]] = {}
    for rel in outputs:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"{cfg.command}: expected output missing: {path}")
        recorded[Path(rel).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if now is None:
        now = datetime.now(UTC)
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{cfg.command}-{config_hash(cfg)[:8]}"
    run_dir = Path(artifacts_dir) / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "command": cfg.command,
        "provenance": provenance,
        "config": cfg.model_dump(mode="json"),
        "config_hash": config_hash(cfg),
        "seeds": cfg.seeds,
        "git": git_info(),
        "created_at": now.isoformat(timespec="seconds"),
        "env": env_fingerprint(),
        "outputs": recorded,
    }
    (run_dir / "run.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def load_run_manifests(artifacts_dir: str | Path = "artifacts") -> list[dict[str, Any]]:
    runs = Path(artifacts_dir) / "runs"
    if not runs.is_dir():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs.glob("*/run.json"))]


def find_manifest(
    rel: str, sha256: str, artifacts_dir: str | Path = "artifacts"
) -> dict[str, Any] | None:
    """The newest manifest recording `rel` with exactly these bytes, or None."""
    for meta in reversed(load_run_manifests(artifacts_dir)):
        entry = meta.get("outputs", {}).get(rel)
        if entry is not None and entry["sha256"] == sha256:
            return meta
    return None


def verify_outputs(
    root: str | Path, rels: Iterable[str], artifacts_dir: str | Path = "artifacts"
) -> dict[str, dict[str, Any]]:
    """Map each data-root-relative result file to its matching manifest, or raise."""
    root = Path(root)
    matched: dict[str, dict[str, Any]] = {}
    unmatched: list[str] = []
    for rel in rels:
        rel = Path(rel).as_posix()
        meta = find_manifest(rel, sha256_file(root / rel), artifacts_dir)
        if meta is None:
            unmatched.append(rel)
        else:
            matched[rel] = meta
    if unmatched:
        raise ProvenanceError(
            "no run manifest under "
            f"{Path(artifacts_dir) / 'runs'} matches the current bytes of: "
            + ", ".join(unmatched)
            + " (re-run the producing command, or `ecvol runs backfill` for pre-manifest files)"
        )
    return matched
