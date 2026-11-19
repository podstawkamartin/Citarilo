"""Typed configuration for local Citarilo runs."""

import hashlib
import json
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

type ReasoningEffort = Literal["none", "low", "medium", "high"]


def make_private_dir(path: Path) -> Path:
    """Create (or tighten) a directory only its owner can enter.

    Runtime state includes the spend ledger, run records with questions and answers,
    and the Qdrant storage, which must not be writable by other local users.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def validate_set_name(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Invalid document set name")
    return name


_MAX_SET_TITLE = 200


def read_set_title(data_dir: Path) -> str:
    """Human title for a document set.

    An optional ``set.json`` may contain ``{"title": "..."}``. A missing or
    unusable file falls back to the directory name. The title is display text,
    never a path.
    """
    fallback = data_dir.name
    try:
        payload = json.loads((data_dir / "set.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    title = payload.get("title")
    if not isinstance(title, str):
        return fallback
    title = " ".join(title.split())
    if not title or len(title) > _MAX_SET_TITLE:
        return fallback
    return title


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
    top_k: int = Field(default=8, gt=0)
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
    # Comma-separated corpus directories served together. Empty keeps data_dir alone,
    # so an existing DATA_DIR setup is unchanged. Ingest, ask and eval still use data_dir.
    data_dirs: str = ""
    runtime_dir: Path = Path(".runtime")
    web_dir: Path = Path("frontend/dist")
    api_host: Literal["127.0.0.1"] = "127.0.0.1"
    api_port: int = Field(default=5177, ge=5174, le=5199)

    def corpus_dirs(self) -> tuple[Path, ...]:
        """Directories the server loads. One entry unless DATA_DIRS lists more."""
        raw = self.data_dirs.strip()
        if not raw:
            return (self.data_dir,)
        directories = tuple(Path(part.strip()) for part in raw.split(",") if part.strip())
        if not directories:
            raise ValueError("DATA_DIRS does not list a directory")
        return directories

    def for_data_dir(self, data_dir: Path) -> "Settings":
        """Same process settings pointed at one corpus directory."""
        if data_dir == self.data_dir:
            return self
        return self.model_copy(update={"data_dir": data_dir})

    @property
    def index_dir(self) -> Path:
        resolved = self.data_dir.resolve()
        slug = re.sub(r"[^a-z0-9]+", "-", resolved.name.lower()).strip("-") or "corpus"
        digest = hashlib.sha256(os.fsencode(resolved)).hexdigest()[:8]
        return self.runtime_dir / "index" / f"{slug}-{digest}"

    @property
    def qdrant_path(self) -> Path:
        return self.index_dir / "qdrant"

    @property
    def index_lock_path(self) -> Path:
        return self.runtime_dir / "index.lock"

    @property
    def index_manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def runs_dir(self) -> Path:
        return self.runtime_dir / "runs"

    @property
    def feedback_dir(self) -> Path:
        return self.runtime_dir / "feedback"

    @property
    def spend_db_path(self) -> Path:
        return self.runtime_dir / "spend.sqlite"
