"""Budgeted embeddings and an explicitly built, corpus-bound vector index."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from citarilo.budget import BudgetLedger, actual_cost, max_cost
from citarilo.config import Settings
from citarilo.corpus import CHUNKING_VERSION, Corpus, section_documents

_MAX_INPUT_TOKENS = 8192
_MAX_BATCH_TOKENS = 300_000


class MissingAPIKeyError(ValueError):
    """A live embedding request requires an OpenAI key."""


class EmbeddingResponseError(RuntimeError):
    """The provider response cannot be used safely."""


class IndexMissingError(FileNotFoundError):
    """The index or its manifest is absent; explicit ingest is required."""


class IndexMismatchError(ValueError):
    """The persisted index is stale, incompatible, or corrupt."""


class _EmbeddingsEndpoint(Protocol):
    def create(
        self,
        *,
        input: list[str],
        model: str,
        dimensions: int,
        encoding_format: Literal["float"],
        timeout: float,
    ) -> CreateEmbeddingResponse: ...


class EmbeddingsClient(Protocol):
    """The SDK surface needed by the adapter; injectable without credentials."""

    @property
    def embeddings(self) -> _EmbeddingsEndpoint: ...


class EmbeddingUsage(BaseModel):
    """Provider-reported usage for one request, linked to its ledger entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reservation_id: str
    prompt_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)
    cost_usd: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    duration_ms: float = Field(default=0, ge=0, allow_inf_nan=False)


class IndexManifest(BaseModel):
    """Compatibility and provenance of a published vector store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    embedding_model: str
    dimensions: int = Field(gt=0, strict=True)
    chunking_version: str
    corpus_hash: str
    section_count: int = Field(gt=0, strict=True)
    created_at: datetime
    usage: list[EmbeddingUsage]
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _batches(texts: list[str], batch_size: int) -> Iterator[list[str]]:
    # Bytes bound tokens without relying on a tokenizer. Keep sections intact.
    bounds = [len(text.encode("utf-8")) for text in texts]
    if any(not 0 < bound <= _MAX_INPUT_TOKENS for bound in bounds):
        raise ValueError("Embedding inputs must contain 1–8192 UTF-8 bytes")
    batch: list[str] = []
    tokens = 0
    for text, bound in zip(texts, bounds, strict=True):
        if batch and (len(batch) == batch_size or tokens + bound > _MAX_BATCH_TOKENS):
            yield batch
            batch, tokens = [], 0
        batch.append(text)
        tokens += bound
    if batch:
        yield batch


class UsageTrackingEmbeddings(Embeddings):
    """Preserve SDK usage and reserve each batch before dispatching it.

    SDK construction and secret access are lazy. Injected clients must provide
    a single-attempt endpoint; the default SDK client disables automatic retries.
    """

    def __init__(
        self,
        settings: Settings,
        ledger: BudgetLedger,
        *,
        group_id: str | None = None,
        client: EmbeddingsClient | None = None,
        operation: str = "embedding",
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.group_id = group_id
        self.operation = operation
        self.usage: list[EmbeddingUsage] = []
        self._client = client

    def _get_client(self) -> EmbeddingsClient:
        if self._client is None:
            secret = self.settings.openai_api_key
            if secret is None or not (key := secret.get_secret_value()).strip():
                raise MissingAPIKeyError("Set OPENAI_API_KEY in the environment or .env file.")
            self._client = OpenAI(
                api_key=key,
                timeout=self.settings.call_timeout_seconds,
                max_retries=0,
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed consecutive batches, retaining usage even for malformed vectors."""
        vectors: list[list[float]] = []
        for batch in _batches(texts, self.settings.embedding_batch_size):
            bound = sum(len(text.encode("utf-8")) for text in batch)
            cost = max_cost(
                self.settings.embedding_model, bound, 0, price_table=self.settings.price_table
            )
            with self.ledger.reserve(
                self.operation, self.settings.embedding_model, cost, group_id=self.group_id
            ) as reservation:
                try:
                    endpoint = self._get_client().embeddings
                except Exception:
                    reservation.cancel()  # Client setup failed before dispatch.
                    raise
                reservation.mark_sent()
                started = perf_counter()
                response = endpoint.create(
                    input=batch,
                    model=self.settings.embedding_model,
                    dimensions=self.settings.embedding_dimensions,
                    encoding_format="float",
                    timeout=self.settings.call_timeout_seconds,
                )
                try:
                    usage = EmbeddingUsage(
                        reservation_id=reservation.id,
                        prompt_tokens=response.usage.prompt_tokens,
                        total_tokens=response.usage.total_tokens,
                        cost_usd=actual_cost(
                            self.settings.embedding_model,
                            response.usage.prompt_tokens,
                            0,
                            price_table=self.settings.price_table,
                        ),
                        duration_ms=(perf_counter() - started) * 1000,
                    )
                    if usage.total_tokens < usage.prompt_tokens:
                        raise ValueError("Inconsistent token usage")
                except (AttributeError, TypeError, ValueError) as exc:
                    raise EmbeddingResponseError("Missing or invalid embedding usage") from exc
                self.usage.append(usage)
                reservation.settle(usage.prompt_tokens, 0)
                items = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in items] != list(range(len(batch))):
                    raise EmbeddingResponseError("Embedding response indices do not match inputs")
                for item in items:
                    if not _valid_vector(item.embedding, self.settings.embedding_dimensions):
                        raise EmbeddingResponseError("Invalid embedding dimensions or values")
                    vectors.append(item.embedding)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed the unmodified question through the same budgeted path."""
        return self.embed_documents([text])[0]


def _valid_vector(vector: object, dimensions: int) -> bool:
    return (
        isinstance(vector, list)
        and len(vector) == dimensions
        and all(
            isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
            for value in vector
        )
        and any(value != 0 for value in vector)
    )


@dataclass(frozen=True)
class IngestEstimate:
    """Conservative token and cost bounds, including per-batch ledger rounding."""

    sections: int
    upper_bound_tokens: int
    requests: int
    worst_case_usd: Decimal


def estimate_ingest(corpus: Corpus, settings: Settings) -> IngestEstimate:
    """Estimate ingestion without keys, network access, or a ledger."""
    texts = [document.page_content for document in section_documents(corpus)]
    bounds = [
        sum(len(text.encode("utf-8")) for text in batch)
        for batch in _batches(texts, settings.embedding_batch_size)
    ]
    return IngestEstimate(
        sections=len(texts),
        upper_bound_tokens=sum(bounds),
        requests=len(bounds),
        worst_case_usd=sum(
            (
                max_cost(settings.embedding_model, bound, 0, price_table=settings.price_table)
                for bound in bounds
            ),
            Decimal(0),
        ),
    )


def build_index(
    corpus: Corpus,
    settings: Settings,
    ledger: BudgetLedger,
    *,
    group_id: str | None = None,
    client: EmbeddingsClient | None = None,
) -> IndexManifest:
    """Embed sections and atomically publish the store, then its manifest."""
    embeddings = UsageTrackingEmbeddings(
        settings, ledger, group_id=group_id, client=client, operation="ingest"
    )
    store = InMemoryVectorStore(embeddings)
    documents = section_documents(corpus)
    store.add_documents(documents, ids=[document.metadata["section_key"] for document in documents])
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".index-", dir=settings.runtime_dir) as directory:
        dump_path = Path(directory) / "index.json"
        manifest_path = Path(directory) / "manifest.json"
        store.dump(str(dump_path))
        manifest = IndexManifest(
            embedding_model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            chunking_version=CHUNKING_VERSION,
            corpus_hash=corpus.corpus_hash,
            section_count=len(documents),
            created_at=datetime.now(UTC),
            usage=embeddings.usage,
            index_sha256=hashlib.sha256(dump_path.read_bytes()).hexdigest(),
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(dump_path, settings.index_path)
        os.replace(manifest_path, settings.index_manifest_path)
    return manifest


def load_index(corpus: Corpus, settings: Settings, embeddings: Embeddings) -> InMemoryVectorStore:
    """Load only compatible, complete data; never call the embedding provider."""
    try:
        manifest = IndexManifest.model_validate_json(settings.index_manifest_path.read_bytes())
        raw = settings.index_path.read_bytes()
    except FileNotFoundError as exc:
        raise IndexMissingError("Index missing; run citarilo ingest explicitly.") from exc
    except (ValidationError, OSError) as exc:
        raise IndexMismatchError("Cannot read index or manifest; run citarilo ingest.") from exc
    expected = {
        "embedding_model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
        "chunking_version": CHUNKING_VERSION,
        "corpus_hash": corpus.corpus_hash,
        "section_count": sum(1 for _ in corpus.sections()),
        "index_sha256": hashlib.sha256(raw).hexdigest(),
    }
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise IndexMismatchError(f"Index {field} mismatch; run citarilo ingest.")
    try:
        # Load the exact bytes whose checksum was verified, avoiding a second read
        # racing with another ingest. Dump contents are only JSON primitives.
        store = InMemoryVectorStore(embeddings)
        store.store = json.loads(raw)
        documents = section_documents(corpus)
        if set(store.store) != {document.metadata["section_key"] for document in documents}:
            raise ValueError("Section IDs do not match the corpus")
        for document in documents:
            key = document.metadata["section_key"]
            record = store.store[key]
            if (
                record["id"] != key
                or record["text"] != document.page_content
                or record["metadata"] != document.metadata
                or not _valid_vector(record["vector"], manifest.dimensions)
            ):
                raise ValueError("Section contents or vector dimensions do not match")
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        raise IndexMismatchError("Invalid vector store; run citarilo ingest.") from exc
    return store


def search(store: InMemoryVectorStore, question: str, k: int) -> list[tuple[str, float, Document]]:
    """Search a loaded store; usage remains on its caller-owned embedding adapter."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return [
        (document.metadata["section_key"], float(score), document)
        for document, score in store.similarity_search_with_score(question, k=k)
    ]


def retrieve(
    question: str,
    corpus: Corpus,
    settings: Settings,
    ledger: BudgetLedger,
    *,
    group_id: str | None = None,
    k: int | None = None,
    client: EmbeddingsClient | None = None,
) -> list[tuple[str, float, Document]]:
    """Return ranked section keys, cosine scores and citable documents."""
    limit = settings.top_k if k is None else k
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("k must be a positive integer")
    embeddings = UsageTrackingEmbeddings(
        settings, ledger, group_id=group_id, client=client, operation="retrieve"
    )
    store = load_index(corpus, settings, embeddings)
    return search(store, question, limit)
