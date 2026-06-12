"""Load, dump, and hash experiment configs (T0.2).

Loading resolves defaults via pydantic; dumping produces deterministic YAML
(keys sorted) so load -> resolve -> dump round-trips are stable. The config
hash is SHA-256 over canonicalized JSON of the resolved config and keys run
artifacts (DESIGN.md §8.2).
"""

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from ecvol.config.schema import ExperimentConfig


class ConfigError(ValueError):
    """Invalid or unreadable experiment config."""


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: invalid config:\n{_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def dump_config(cfg: ExperimentConfig) -> str:
    """Resolved config as deterministic YAML (defaults filled, keys sorted)."""
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True)


def config_hash(cfg: ExperimentConfig) -> str:
    """SHA-256 hex digest of the canonicalized resolved config."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
