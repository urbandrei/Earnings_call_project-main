"""Command-run manifests and provenance verification (T9.3 acceptance).

Every result-producing CLI command writes artifacts/runs/<run_id>/run.json with the
SHA-256 of each result CSV it produced; `ecvol report` and CI refuse a result file
whose bytes match no manifest (DECISIONS 2026-08-26 §4 + the T9.3 design call).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from ecvol.config import CommandConfig, ConfigError, load_command_config
from ecvol.tracking import (
    COMMAND_OUTPUTS,
    ProvenanceError,
    find_manifest,
    resolve_command_config,
    sha256_file,
    verify_outputs,
    write_command_run,
)

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=UTC)


def _root_with(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "data"
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    return root


def test_write_command_run_records_outputs(tmp_path):
    root = _root_with(tmp_path, {"results/result_table_1.csv": b"a,b\n1,2\n"})
    cfg = CommandConfig(command="evaluate", seeds=[0, 1])
    run_dir = write_command_run(cfg, root, artifacts_dir=tmp_path / "art", now=NOW)

    assert run_dir.name.startswith("20260826T120000Z-evaluate-")
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["command"] == "evaluate" and meta["provenance"] == "run"
    assert meta["seeds"] == [0, 1] and meta["config"] == {"command": "evaluate", "seeds": [0, 1]}
    entry = meta["outputs"]["results/result_table_1.csv"]
    assert entry["sha256"] == sha256_file(root / "results/result_table_1.csv")
    assert entry["bytes"] == 8
    assert meta["git"] is not None and len(meta["git"]["sha"]) == 40
    assert set(meta["env"]) == {"python", "platform", "lockfile_sha256", "gpus"}


def test_write_command_run_requires_outputs_to_exist(tmp_path):
    root = _root_with(tmp_path, {})
    with pytest.raises(FileNotFoundError):
        write_command_run(CommandConfig(command="grid"), root, artifacts_dir=tmp_path / "art")


def test_verify_outputs_matches_then_detects_modified_bytes(tmp_path):
    root = _root_with(
        tmp_path,
        {"results/result_table_4.csv": b"x\n", "results/result_table_4_peryear.csv": b"y\n"},
    )
    art = tmp_path / "art"
    write_command_run(CommandConfig(command="grid"), root, artifacts_dir=art, now=NOW)

    matched = verify_outputs(root, COMMAND_OUTPUTS["grid"], art)
    assert set(matched) == set(COMMAND_OUTPUTS["grid"])

    (root / "results/result_table_4.csv").write_bytes(b"x-modified\n")
    with pytest.raises(ProvenanceError, match="result_table_4.csv"):
        verify_outputs(root, COMMAND_OUTPUTS["grid"], art)
    assert find_manifest("results/result_table_4.csv", "0" * 64, art) is None


def test_verify_prefers_newest_manifest(tmp_path):
    root = _root_with(tmp_path, {"results/result_table_1.csv": b"v1\n"})
    art = tmp_path / "art"
    cfg = CommandConfig(command="evaluate")
    write_command_run(cfg, root, artifacts_dir=art, now=NOW)
    (root / "results/result_table_1.csv").write_bytes(b"v2\n")
    later = write_command_run(cfg, root, artifacts_dir=art, now=NOW.replace(hour=13))
    assert (
        verify_outputs(root, COMMAND_OUTPUTS["evaluate"], art)["results/result_table_1.csv"][
            "run_id"
        ]
        == later.name
    )


def test_resolve_command_config_default_path_and_seed_override(tmp_path):
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    (cfgs / "controls.yaml").write_text(
        yaml.safe_dump({"command": "controls", "seeds": [1, 2, 3]}), encoding="utf-8"
    )
    assert resolve_command_config("controls", configs_dir=cfgs).seeds == [1, 2, 3]
    assert resolve_command_config("controls", seeds="7, 8", configs_dir=cfgs).seeds == [7, 8]
    with pytest.raises(ConfigError, match="not 'grid'"):
        resolve_command_config("grid", config_path=cfgs / "controls.yaml")
    with pytest.raises(ConfigError, match="not found"):
        resolve_command_config("grid", configs_dir=cfgs)
    with pytest.raises(ConfigError):
        resolve_command_config("controls", seeds="1,1", configs_dir=cfgs)


def test_command_config_rejects_unknown_command_and_duplicate_seeds():
    with pytest.raises(ValueError):
        CommandConfig(command="train")
    with pytest.raises(ValueError):
        CommandConfig(command="evaluate", seeds=[0, 0])


# --- committed artifacts stay in sync (CI-guarded; DESIGN §8.2 / §12) ---------


def test_every_command_has_a_committed_config():
    for command in COMMAND_OUTPUTS:
        cfg = load_command_config(REPO / "configs" / f"{command}.yaml", command)
        assert cfg.command == command


def test_committed_results_have_matching_manifests():
    """Every committed result CSV must match a committed run manifest byte-for-byte."""
    root = REPO / "data"
    present = [rel for outs in COMMAND_OUTPUTS.values() for rel in outs if (root / rel).is_file()]
    if not present:
        return
    matched = verify_outputs(root, present, REPO / "artifacts")
    assert set(matched) == set(present)
