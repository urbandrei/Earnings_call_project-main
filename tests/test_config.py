"""Config schema, loader, and hash tests (T0.2 acceptance)."""

from pathlib import Path

import pytest
import yaml

from ecvol.config import (
    ConfigError,
    ExperimentConfig,
    config_hash,
    dump_config,
    load_config,
)

EXAMPLE = Path(__file__).parents[1] / "configs" / "example.yaml"


def valid_dict() -> dict:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert cfg.name == "example-gbdt-fincall-temporal"
    assert len(cfg.seeds) == 5


def test_bad_horizon_fails_with_actionable_message(tmp_path):
    data = valid_dict()
    data["target"]["horizons"] = [3, 9]
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_yaml(tmp_path, data))
    message = str(excinfo.value)
    assert "target.horizons" in message  # where
    assert "9" in message  # what
    assert "[3, 7, 15, 30]" in message  # the fix


def test_missing_seed_list_fails_with_actionable_message(tmp_path):
    data = valid_dict()
    del data["seeds"]
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_yaml(tmp_path, data))
    message = str(excinfo.value)
    assert "seeds" in message
    assert "required" in message.lower()


def test_unknown_key_rejected(tmp_path):
    data = valid_dict()
    data["sedes"] = [1, 2, 3]  # typo of seeds
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_yaml(tmp_path, data))
    assert "sedes" in str(excinfo.value)


def test_duplicate_seeds_rejected(tmp_path):
    data = valid_dict()
    data["seeds"] = [101, 101]
    with pytest.raises(ConfigError):
        load_config(write_yaml(tmp_path, data))


def test_embargo_shorter_than_longest_horizon_rejected(tmp_path):
    data = valid_dict()
    data["split"]["embargo_trading_days"] = 7
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_yaml(tmp_path, data))
    assert "embargo" in str(excinfo.value)


def test_missing_file_and_non_mapping_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "list.yaml"
    bad.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_round_trip_load_resolve_dump_is_stable(tmp_path):
    cfg = load_config(EXAMPLE)
    dumped = dump_config(cfg)
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text(dumped, encoding="utf-8")
    cfg2 = load_config(resolved)
    assert cfg2 == cfg
    assert dump_config(cfg2) == dumped


def test_hash_canonicalizes_key_order_and_tracks_content():
    data = valid_dict()
    reordered = dict(reversed(list(data.items())))
    h_original = config_hash(ExperimentConfig.model_validate(data))
    h_reordered = config_hash(ExperimentConfig.model_validate(reordered))
    assert h_original == h_reordered
    assert len(h_original) == 64  # sha256 hex

    data["seeds"] = [101]
    assert config_hash(ExperimentConfig.model_validate(data)) != h_original
