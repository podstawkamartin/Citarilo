"""Typed configuration for local Citarilo runs."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

type ReasoningEffort = Literal["none", "low", "medium", "high"]


class TokenPrice(BaseModel):
    """USD per one million input or output tokens."""

    input_usd_per_million: Decimal = Field(ge=0)
    output_usd_per_million: Decimal = Field(default=Decimal("0"), ge=0)


class Settings(BaseSettings):
    """Load settings from environment variables and an optional local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    generation_model: str = "gpt-6-luna"
    verification_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = Field(default=3072, gt=0)
    embedding_batch_size: int = Field(default=2048, gt=0, le=2048)
    top_k: int = Field(default=5, gt=0)
    generation_reasoning_effort: ReasoningEffort = "none"
    verification_reasoning_effort: ReasoningEffort = "low"
    call_timeout_seconds: float = Field(default=30.0, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)

    price_table_version: str = "2026-11-06"
    price_table: dict[str, TokenPrice] = Field(
        default_factory=lambda: {
            "gpt-6-luna": TokenPrice(
                input_usd_per_million=Decimal("0.10"),
                output_usd_per_million=Decimal("0.50"),
            ),
            "text-embedding-3-large": TokenPrice(input_usd_per_million=Decimal("0.13")),
            "gpt-6.1-terra": TokenPrice(
                input_usd_per_million=Decimal("1.00"),
                output_usd_per_million=Decimal("5.00"),
            ),
        }
    )
    budget_cap_usd: Decimal = Field(default=Decimal("3.0"), gt=0, le=3)

    data_dir: Path = Path("data/trailsupply")
    runtime_dir: Path = Path(".runtime")
    web_dir: Path = Path("frontend/dist")
    api_host: Literal["127.0.0.1"] = "127.0.0.1"
    api_port: int = Field(default=5174, ge=5174, le=5199)

    @property
    def index_path(self) -> Path:
        return self.runtime_dir / "index.json"

    @property
    def index_manifest_path(self) -> Path:
        return self.runtime_dir / "index.manifest.json"

    @property
    def runs_dir(self) -> Path:
        return self.runtime_dir / "runs"

    @property
    def spend_db_path(self) -> Path:
        return self.runtime_dir / "spend.sqlite"
