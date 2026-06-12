"""Run-artifact writer, run-ID scheme, env fingerprint (T0.3 acceptance)."""

import json
import random
from datetime import UTC, datetime

import pytest

from ecvol.config import ExperimentConfig, config_hash
from ecvol.tracking import (
    env_fingerprint,
    git_info,
    new_run_id,
    read_metrics,
    write_run,
)

CFG = ExperimentConfig.model_validate(
    {
        "name": "tracking-test",
        "seeds": [0, 1],
        "data": {"dataset": "fincall_surprise"},
        "target": {"variant": "delta", "horizons": [3, 30]},
        "split": {"scheme": "temporal", "embargo_trading_days": 30},
        "model": {"name": "har_rv"},
    }
)


def _cpu_only_metrics(cfg: ExperimentConfig) -> list[dict]:
    """Stand-in for a real pipeline run: deterministic given config + seeds."""
    rows = []
    for seed in cfg.seeds:
        rng = random.Random(seed)
        for tau in cfg.target.horizons:
            rows.append({"seed": seed, "horizon": tau, "mse": rng.random(), "mae": rng.random()})
    return rows


def test_rerun_identical_config_reproduces_metrics_bit_identically(tmp_path):
    run1 = write_run(CFG, _cpu_only_metrics(CFG), tmp_path / "a")
    run2 = write_run(CFG, _cpu_only_metrics(CFG), tmp_path / "b")
    assert (run1 / "metrics.parquet").read_bytes() == (run2 / "metrics.parquet").read_bytes()
    assert (run1 / "config.yaml").read_bytes() == (run2 / "config.yaml").read_bytes()


def test_run_artifact_contents(tmp_path):
    metrics = _cpu_only_metrics(CFG)
    run_dir = write_run(CFG, metrics, tmp_path)

    assert read_metrics(run_dir) == metrics
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == run_dir.name
    assert meta["config_hash"] == config_hash(CFG)
    assert meta["seeds"] == CFG.seeds
    assert set(meta["env"]) == {"python", "platform", "lockfile_sha256", "gpus"}
    # this test runs inside the project repo, so git info must be present
    assert meta["git"] is not None and len(meta["git"]["sha"]) == 40


def test_run_id_scheme():
    ts = datetime(2026, 6, 12, 15, 30, 0, tzinfo=UTC)
    assert new_run_id(CFG, now=ts) == f"20260612T153000Z-{config_hash(CFG)[:8]}"


def test_write_run_refuses_overwrite(tmp_path):
    run_dir = write_run(CFG, _cpu_only_metrics(CFG), tmp_path)
    with pytest.raises(FileExistsError):
        write_run(CFG, _cpu_only_metrics(CFG), tmp_path, run_id=run_dir.name)


def test_write_run_rejects_ragged_and_empty_metrics(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_run(CFG, [], tmp_path)
    ragged = [{"seed": 0, "mse": 0.1}, {"seed": 1, "rmse": 0.2}]
    with pytest.raises(ValueError, match="row 1"):
        write_run(CFG, ragged, tmp_path)


def test_env_fingerprint_lockfile_hash(tmp_path):
    fp = env_fingerprint()  # cwd is the project root under pytest
    assert fp["lockfile_sha256"] is not None and len(fp["lockfile_sha256"]) == 64
    assert env_fingerprint(lockfile=tmp_path / "absent.lock")["lockfile_sha256"] is None


def test_git_info_in_repo():
    info = git_info()
    assert info is not None
    assert len(info["sha"]) == 40
    assert isinstance(info["dirty"], bool)
