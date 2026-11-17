"""Settings, loadable from a TOML file. Every field has a working default."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    command: str = "claude"
    # Aliases resolve to the latest model of each family, so the desk does not
    # need an update every time a model ships. Pin full IDs here for reproducibility.
    fast_model: str = "sonnet"  # analysts and debaters
    deep_model: str = "opus"  # risk manager and portfolio manager
    fallback_model: str | None = "sonnet"
    timeout_s: float = 300.0
    max_retries: int = 2
    max_concurrency: int = 4
    cache_dir: Path | None = Path(".desk_cache/llm")


class RiskLimits(BaseModel):
    max_position_pct: float = Field(20.0, gt=0, le=100)
    min_conviction: float = Field(0.55, ge=0, le=1)
    max_stop_distance_pct: float = Field(15.0, gt=0, le=90)
    default_stop_distance_pct: float = Field(8.0, gt=0, le=90)


class DeskSettings(BaseModel):
    llm: LLMSettings = LLMSettings()
    risk: RiskLimits = RiskLimits()
    debate_rounds: int = Field(1, ge=1, le=5)
    lookback_days: int = Field(400, ge=60)
    web_search: bool = True  # live mode only: lets the news analyst search the web
    language: str = "English"
    results_dir: Path = Path("results")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DeskSettings":
        if path is None:
            return cls()
        with open(path, "rb") as fh:
            return cls.model_validate(tomllib.load(fh))
