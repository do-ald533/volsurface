"""Typed application settings.

Values come from, in increasing priority: field defaults, a ``.env`` file in the working
directory, ``VOLSURFACE_*`` environment variables (nested fields use ``__``, for example
``VOLSURFACE_IBKR__PORT=4001``) and finally an explicit TOML file passed to
:meth:`Settings.load`.
"""

import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from volsurface.domain.errors import ConfigurationError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class IbkrSettings(BaseModel):
    """Connection and subscription limits for Interactive Brokers (TWS or IB Gateway)."""

    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=7497, ge=1, le=65535)
    client_id: int = Field(default=35, ge=0)
    max_lines: int = Field(default=80, ge=5, le=10_000)
    expiries: int = Field(default=6, ge=1, le=24)
    band: float = Field(default=0.08, gt=0.0, lt=1.0)
    min_days: int = Field(default=3, ge=0)
    max_days: int = Field(default=120, ge=1)
    market_data_type: int = Field(default=3, ge=1, le=4)
    connect_timeout_s: float = Field(default=10.0, gt=0.0)
    rate: float = Field(default=0.04, ge=-0.05, le=0.5)
    dividend_yield: float = Field(default=0.012, ge=0.0, le=0.5)


class PipelineSettings(BaseModel):
    """Quote filtering, pricing and fitting parameters."""

    model_config = ConfigDict(frozen=True)

    max_days_to_expiry: int = Field(default=120, ge=1)
    min_time_value: float = Field(default=0.05, ge=0.0)
    tree_steps: int = Field(default=200, ge=3, le=5000)
    gex_tree_steps: int = Field(default=100, ge=3, le=5000)
    min_points_per_expiry: int = Field(default=6, ge=5)
    flip_grid_low: float = Field(default=0.90, gt=0.0, lt=1.0)
    flip_grid_high: float = Field(default=1.10, gt=1.0)
    flip_grid_points: int = Field(default=81, ge=3)
    near_flip_pct: float = Field(default=0.5, gt=0.0)
    min_gex: float = Field(default=0.05, ge=0.0)
    iv_jump_points: float = Field(default=2.0, gt=0.0)
    compute_gex: bool = True

    @field_validator("flip_grid_points")
    @classmethod
    def _odd_grid(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("flip_grid_points must be odd so that spot is a grid point")
        return value


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_prefix="VOLSURFACE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    log_level: LogLevel = "INFO"
    log_json: bool = False
    backend: str = "numba-cpu"
    ui_hz: float = Field(default=5.0, gt=0.0, le=60.0)
    ibkr: IbkrSettings = IbkrSettings()
    pipeline: PipelineSettings = PipelineSettings()

    @classmethod
    def load(cls, toml_path: Path | None = None) -> Self:
        """Build settings; values in ``toml_path`` take priority over the environment."""
        values: dict[str, Any] = {}
        if toml_path is not None:
            if not toml_path.is_file():
                raise ConfigurationError(f"settings file not found: {toml_path}")
            try:
                values = tomllib.loads(toml_path.read_text())
            except tomllib.TOMLDecodeError as exc:
                raise ConfigurationError(f"invalid TOML in {toml_path}: {exc}") from exc
        try:
            return cls(**values)
        except ValidationError as exc:
            raise ConfigurationError(f"invalid settings: {exc}") from exc
