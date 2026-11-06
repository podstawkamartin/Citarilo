"""Isolate every test from local credentials and real provider clients."""

import pytest

import citarilo.index as index_module
from citarilo.config import Settings


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    def forbidden(*args, **kwargs):
        pytest.fail("Tests must inject a fake OpenAI client")

    monkeypatch.setattr(index_module, "OpenAI", forbidden)
