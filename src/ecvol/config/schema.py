"""Pydantic schemas for experiment configs (T0.2).

One experiment = one YAML file under configs/, validated into ExperimentConfig.
Field constraints encode DESIGN.md contracts: horizons and target variants from
§5.3, split schemes and the embargo rule from §5.4, seeds-in-configs from §8.2.
Unknown keys are rejected everywhere (extra="forbid") so typos fail loudly.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_HORIZONS = (3, 7, 15, 30)  # trading days, DESIGN.md §5.3


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    dataset: Literal["fincall_surprise", "maec", "legacy_earningscall", "fresh_2026"]


class TargetConfig(StrictModel):
    variant: Literal["level", "delta", "har_residual"]
    horizons: list[int] = Field(min_length=1)

    @field_validator("horizons")
    @classmethod
    def _allowed_and_unique(cls, v: list[int]) -> list[int]:
        bad = sorted({h for h in v if h not in ALLOWED_HORIZONS})
        if bad:
            raise ValueError(
                f"horizons {bad} are not valid; allowed values are "
                f"{list(ALLOWED_HORIZONS)} trading days (DESIGN.md §5.3)"
            )
        if len(set(v)) != len(v):
            raise ValueError(f"horizons must be unique, got {v}")
        return sorted(v)


class SplitConfig(StrictModel):
    scheme: Literal["temporal", "ticker_disjoint", "combined"]
    embargo_trading_days: int = Field(default=30, ge=0)


class FeatureSpec(StrictModel):
    kind: Literal["text", "audio", "llm"]
    extractor: str = Field(min_length=1)
    model_id: str | None = None  # HF repo id, e.g. "BAAI/bge-large-en-v1.5"
    revision: str | None = None  # exact HF commit, pinned at use time (DESIGN.md §12)
    options: dict[str, Any] = {}


class ModelConfig(StrictModel):
    name: str = Field(min_length=1)
    include_past_vol_covariates: bool = True  # §6 invariant; ablations set False
    params: dict[str, Any] = {}


class EvalConfig(StrictModel):
    metrics: list[Literal["mse", "mae", "r2_oos", "spearman_quarterly"]] = [
        "mse",
        "mae",
        "r2_oos",
        "spearman_quarterly",
    ]
    bootstrap_resamples: int = Field(default=1000, ge=1)


class ExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    data: DataConfig
    target: TargetConfig
    split: SplitConfig
    features: list[FeatureSpec] = []
    model: ModelConfig
    eval: EvalConfig = EvalConfig()

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError(f"seeds must be unique, got {v}")
        return v

    @model_validator(mode="after")
    def _embargo_covers_longest_horizon(self) -> "ExperimentConfig":
        if self.split.scheme in ("temporal", "combined"):
            longest = max(self.target.horizons)
            if self.split.embargo_trading_days < longest:
                raise ValueError(
                    f"split.embargo_trading_days={self.split.embargo_trading_days} is shorter "
                    f"than the longest target horizon ({longest}); a target window would cross "
                    f"the split boundary (DESIGN.md §5.4) — set it to at least {longest}"
                )
        return self
