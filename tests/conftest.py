"""Isolate every test from local credentials and real provider clients."""

from unittest.mock import Mock

import httpx2
import pytest
from langchain_openai import ChatOpenAI
from openai import DefaultHttpxClient
from qdrant_client import QdrantClient, models

import citarilo.index as index_module
import citarilo.pipeline as pipeline_module
from citarilo.config import Settings
from citarilo.index import SectionIndex, _point_id


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    def forbidden(*args, **kwargs):
        pytest.fail("Tests must inject a fake OpenAI client")

    monkeypatch.setattr(index_module, "OpenAI", forbidden)
    monkeypatch.setattr(ChatOpenAI, "__init__", forbidden)
    monkeypatch.setattr(pipeline_module, "ChatOpenAI", forbidden)
    monkeypatch.setattr(pipeline_module, "_UsageValidatedOpenAI", forbidden)


@pytest.fixture
def tracked_http_client(monkeypatch):
    def forbidden(request):
        pytest.fail("Lifecycle tests must not dispatch HTTP requests")

    with DefaultHttpxClient(transport=httpx2.MockTransport(forbidden)) as client:
        monkeypatch.setattr(client, "close", Mock(wraps=client.close))
        yield client


@pytest.fixture
def section_index_factory(tmp_path, request):
    """Disk-backed fixture indexes with isolated locks and no usage reservations."""
    count = 0

    def create(documents, embeddings, dimensions=8):
        nonlocal count
        path = tmp_path / f"fixture-index-{count}"
        count += 1
        settings = Settings(_env_file=None, runtime_dir=path)
        app_lock = index_module._acquire_index_lock(settings)
        client = QdrantClient(path=str(settings.qdrant_path))
        store = SectionIndex(client, embeddings, dimensions, app_lock=app_lock)
        request.addfinalizer(store.close)
        client.create_collection(
            "sections",
            vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
        )
        if documents:
            from langchain_core.embeddings import DeterministicFakeEmbedding

            vectors = DeterministicFakeEmbedding(size=dimensions).embed_documents(
                [doc.page_content for doc in documents]
            )
            client.upsert(
                "sections",
                [
                    models.PointStruct(
                        id=_point_id(doc.metadata["section_key"]),
                        vector=vector,
                        payload={"page_content": doc.page_content, "metadata": doc.metadata},
                    )
                    for doc, vector in zip(documents, vectors, strict=True)
                ],
                wait=True,
            )
        return store

    return create
