"""Pydantic config schemas, loader, and hash (T0.2)."""

from ecvol.config.load import ConfigError, config_hash, dump_config, load_config
from ecvol.config.schema import (
    ALLOWED_HORIZONS,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    FeatureSpec,
    ModelConfig,
    SplitConfig,
    TargetConfig,
)

__all__ = [
    "ALLOWED_HORIZONS",
    "ConfigError",
    "DataConfig",
    "EvalConfig",
    "ExperimentConfig",
    "FeatureSpec",
    "ModelConfig",
    "SplitConfig",
    "TargetConfig",
    "config_hash",
    "dump_config",
    "load_config",
]
