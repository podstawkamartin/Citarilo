"""Isolate every test from local credentials and real provider clients."""

from unittest.mock import Mock

import httpx2
import pytest
from langchain_openai import ChatOpenAI
from openai import DefaultHttpxClient

import citarilo.index as index_module
import citarilo.pipeline as pipeline_module
from citarilo.config import Settings


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
