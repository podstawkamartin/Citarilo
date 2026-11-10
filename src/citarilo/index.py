"""Budgeted embeddings and an explicitly built, corpus-bound vector index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal, Protocol
from uuid import UUID, uuid5

import httpx2
import portalocker
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from qdrant_client import QdrantClient, models

from citarilo.budget import BudgetLedger, actual_cost, max_cost
from citarilo.config import Settings, make_private_dir
from citarilo.corpus import CHUNKING_VERSION, Corpus, section_documents

_MAX_INPUT_TOKENS = 8192
_MAX_BATCH_TOKENS = 300_000
_COLLECTION = "sections"
_POINT_NAMESPACE = UUID("97fac41b-d50e-4cb8-839a-35d38be8a357")
_SCROLL_PAGE_SIZE = 256


class MissingAPIKeyError(ValueError):
    """A live embedding request requires an OpenAI key."""


class EmbeddingResponseError(RuntimeError):
    """The provider response cannot be used safely."""


class IndexMissingError(FileNotFoundError):
    """The index or its manifest is absent; explicit ingest is required."""


class IndexMismatchError(ValueError):
    """The persisted index is stale, incompatible, or corrupt."""


class IndexLockedError(RuntimeError):
    """Another local client owns the index directory."""

    def __init__(self) -> None:
        super().__init__(
            "The local index is in use by another Citarilo process (for example "
            "`citarilo serve`). Stop it and try again."
        )


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

    schema_version: Literal[2] = 2
    embedding_model: str
    dimensions: int = Field(gt=0, strict=True)
    chunking_version: str
    corpus_hash: str
    section_count: int = Field(gt=0, strict=True)
    created_at: datetime
    usage: list[EmbeddingUsage]
    points_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
        http_client: httpx2.Client | None = None,
        operation: str = "embedding",
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.group_id = group_id
        self.operation = operation
        self.usage: list[EmbeddingUsage] = []
        self._client = client
        self._http_client = http_client

    def _get_client(self) -> EmbeddingsClient:
        if self._client is None:
            secret = self.settings.openai_api_key
            if secret is None or not (key := secret.get_secret_value()).strip():
                raise MissingAPIKeyError("Set OPENAI_API_KEY in the environment or .env file.")
            self._client = OpenAI(
                api_key=key,
                timeout=self.settings.call_timeout_seconds,
                max_retries=0,
                http_client=self._http_client,
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


def _normalize_vector(vector: list[float], dimensions: int) -> list[float]:
    """Normalize validated embeddings without overflow for large finite components."""
    if not _valid_vector(vector, dimensions):
        raise EmbeddingResponseError("Invalid embedding dimensions or values")
    scale = max(abs(value) for value in vector)
    scaled = [value / scale for value in vector]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]


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
    scroll_page_size: int = _SCROLL_PAGE_SIZE,
) -> IndexManifest:
    """Acquire the local lock before embedding and publish validated points atomically."""
    app_lock = _acquire_index_lock(settings)
    qdrant = None
    try:
        embeddings = UsageTrackingEmbeddings(
            settings, ledger, group_id=group_id, client=client, operation="ingest"
        )
        documents = section_documents(corpus)
        vectors = [
            _normalize_vector(vector, settings.embedding_dimensions)
            for vector in embeddings.embed_documents(
                [document.page_content for document in documents]
            )
        ]
        # Never open or deserialize old storage: a successful embedding can repair it.
        path = settings.qdrant_path
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
        qdrant = _open_client(settings)
        qdrant.create_collection(
            _COLLECTION,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimensions, distance=models.Distance.COSINE
            ),
        )
        qdrant.upsert(
            _COLLECTION,
            points=[
                models.PointStruct(
                    id=_point_id(document.metadata["section_key"]),
                    vector=vector,
                    payload={"page_content": document.page_content, "metadata": document.metadata},
                )
                for document, vector in zip(documents, vectors, strict=True)
            ],
            wait=True,
        )
        # Local Qdrant 1.20 persists input vectors but normalizes its live copy.
        # Reopen immediately so the digest covers the same bytes load_index reads.
        qdrant.close()
        qdrant = None
        qdrant = _open_client(settings)
        points = _validated_points(
            qdrant, corpus, settings.embedding_dimensions, page_size=scroll_page_size
        )
        expected_vectors = {
            document.metadata["section_key"]: vector
            for document, vector in zip(documents, vectors, strict=True)
        }
        for point in points:
            expected = expected_vectors[point.payload["metadata"]["section_key"]]
            # Allow float32 storage rounding while detecting another build's vectors.
            if not all(
                math.isclose(actual, wanted, rel_tol=1e-6, abs_tol=1e-8)
                for actual, wanted in zip(point.vector, expected, strict=True)
            ):
                raise IndexMismatchError(
                    "Persisted vectors differ from this build; run citarilo ingest."
                )
        manifest = IndexManifest(
            embedding_model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            chunking_version=CHUNKING_VERSION,
            corpus_hash=corpus.corpus_hash,
            section_count=len(documents),
            created_at=datetime.now(UTC),
            usage=embeddings.usage,
            points_sha256=_points_sha256(points),
        )
        with TemporaryDirectory(prefix=".index-", dir=settings.runtime_dir) as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
            os.replace(manifest_path, settings.index_manifest_path)
        return manifest
    finally:
        try:
            if qdrant is not None:
                qdrant.close()
        finally:
            app_lock.release()


def _acquire_index_lock(settings: Settings) -> portalocker.Lock:
    """Serialize Citarilo owners across the entire Qdrant close/reopen interval."""
    make_private_dir(settings.runtime_dir)
    lock = portalocker.Lock(
        str(settings.index_lock_path),
        mode="a",
        timeout=0,
        flags=portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING,
    )
    try:
        lock.acquire()
    except portalocker.exceptions.LockException as exc:
        lock.release()
        raise IndexLockedError() from exc
    except BaseException:
        lock.release()
        raise
    return lock


def _open_client(settings: Settings) -> QdrantClient:
    # Retain the instance if __init__ fails so partially initialized resources
    # can be closed immediately, rather than waiting for garbage collection.
    client = QdrantClient.__new__(QdrantClient)
    try:
        client.__init__(path=str(settings.qdrant_path))
        return client
    except BaseException as exc:
        with suppress(Exception):
            client.close()
        if isinstance(exc, RuntimeError) and "already accessed by another instance" in str(exc):
            raise IndexLockedError() from exc
        if not isinstance(exc, Exception):
            raise
        raise IndexMismatchError("Cannot open local index; run citarilo ingest.") from exc


def _point_id(section_key: str) -> str:
    return str(uuid5(_POINT_NAMESPACE, section_key))


def _validated_points(
    client: QdrantClient, corpus: Corpus, dimensions: int, *, page_size: int
) -> list[models.Record]:
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("Scroll page size must be a positive integer")
    if not client.collection_exists(_COLLECTION):
        raise IndexMissingError("Index collection missing; run citarilo ingest explicitly.")
    info = client.get_collection(_COLLECTION)
    config = info.config.params.vectors
    if (
        not isinstance(config, models.VectorParams)
        or config.size != dimensions
        or config.distance != models.Distance.COSINE
    ):
        raise ValueError("Invalid collection dimensions or distance")
    documents = {doc.metadata["section_key"]: doc for doc in section_documents(corpus)}
    if client.count(_COLLECTION, exact=True).count != len(documents):
        raise ValueError("Point count does not match the corpus")
    points = []
    offset = None
    seen = set()
    while True:
        page, offset = client.scroll(
            _COLLECTION, limit=page_size, offset=offset, with_vectors=True, with_payload=True
        )
        for point in page:
            payload = point.payload
            if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
                raise ValueError("Invalid point payload")
            key = payload["metadata"].get("section_key")
            if not isinstance(key, str) or key in seen or key not in documents:
                raise ValueError("Section keys do not match the corpus")
            document = documents[key]
            if (
                point.id != _point_id(key)
                or payload != {"page_content": document.page_content, "metadata": document.metadata}
                or not _valid_vector(point.vector, dimensions)
            ):
                raise ValueError("Section contents or vector dimensions do not match")
            seen.add(key)
            points.append(point)
        if offset is None:
            break
    if seen != set(documents):
        raise ValueError("Section keys do not match the corpus")
    return points


def _points_sha256(points: list[models.Record]) -> str:
    """Hash validated Qdrant readback, independent of scroll order and JSON formatting."""
    digest = hashlib.sha256()
    for point in sorted(points, key=lambda p: p.payload["metadata"]["section_key"]):
        payload = point.payload
        for value in (
            payload["metadata"]["section_key"],
            payload["page_content"],
            payload["metadata"],
        ):
            encoded = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        digest.update(struct.pack(f"<{len(point.vector)}f", *point.vector))
    return digest.hexdigest()


class SectionIndex:
    """Own one local Qdrant client; alternate query adapters borrow the same store."""

    def __init__(
        self,
        client: QdrantClient,
        embeddings: Embeddings,
        dimensions: int,
        *,
        app_lock: portalocker.Lock,
    ) -> None:
        self._client = client
        self._app_lock = app_lock
        self.embeddings = embeddings
        self.dimensions = dimensions
        self._closed = False

    def close(self) -> None:
        """Release the directory lock exactly once, including across context copies."""
        if not self._closed:
            self._closed = True
            try:
                self._client.close()
            finally:
                self._app_lock.release()

    def __enter__(self) -> SectionIndex:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def search(
        self, question: str, k: int, *, embeddings: Embeddings | None = None
    ) -> list[tuple[str, float, Document]]:
        """Return cosine-ranked sections, optionally tracking usage on a caller adapter."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if self._closed:
            raise RuntimeError("The section index is closed")
        # Evals override the adapter for per-run usage tracking without another client.
        vector = (self.embeddings if embeddings is None else embeddings).embed_query(question)
        if not _valid_vector(vector, self.dimensions):
            raise EmbeddingResponseError("Invalid embedding dimensions or values")
        points = self._client.query_points(
            _COLLECTION, query=vector, limit=k, with_payload=True
        ).points
        return [
            (
                point.payload["metadata"]["section_key"],
                float(point.score),
                Document(
                    page_content=point.payload["page_content"], metadata=point.payload["metadata"]
                ),
            )
            for point in points
        ]


def load_index(
    corpus: Corpus,
    settings: Settings,
    embeddings: Embeddings,
    *,
    scroll_page_size: int = _SCROLL_PAGE_SIZE,
) -> SectionIndex:
    """Check the manifest before taking the lock; validate all points without embedding."""
    try:
        manifest = IndexManifest.model_validate_json(settings.index_manifest_path.read_bytes())
    except FileNotFoundError as exc:
        raise IndexMissingError("Index missing; run citarilo ingest explicitly.") from exc
    except (ValidationError, OSError) as exc:
        raise IndexMismatchError("Cannot read index manifest; run citarilo ingest.") from exc
    expected = {
        "embedding_model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
        "chunking_version": CHUNKING_VERSION,
        "corpus_hash": corpus.corpus_hash,
        "section_count": sum(1 for _ in corpus.sections()),
    }
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise IndexMismatchError(f"Index {field} mismatch; run citarilo ingest.")
    app_lock = _acquire_index_lock(settings)
    client = None
    success = False
    try:
        client = _open_client(settings)
        try:
            points = _validated_points(
                client, corpus, manifest.dimensions, page_size=scroll_page_size
            )
            if _points_sha256(points) != manifest.points_sha256:
                raise IndexMismatchError("Index points_sha256 mismatch; run citarilo ingest.")
        except (IndexMissingError, IndexMismatchError):
            raise
        except Exception as exc:
            raise IndexMismatchError("Invalid vector store; run citarilo ingest.") from exc
        store = SectionIndex(client, embeddings, manifest.dimensions, app_lock=app_lock)
        success = True
        return store
    finally:
        if not success:
            try:
                if client is not None:
                    client.close()
            finally:
                app_lock.release()


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
    with load_index(corpus, settings, embeddings) as store:
        return store.search(question, limit)
