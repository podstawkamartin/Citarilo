"""HTTP contracts and guards with in-process clients and deterministic providers."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from citarilo import __version__, api, pipeline
from citarilo.budget import BudgetLedger
from citarilo.cli import main
from citarilo.config import Settings
from citarilo.corpus import load_corpus, section_documents
from citarilo.index import (
    IndexLockedError,
    IndexMismatchError,
    IndexMissingError,
    MissingAPIKeyError,
    UsageTrackingEmbeddings,
    build_index,
)
from citarilo.pipeline import AnswerContext, PipelineError
from citarilo.records import AskResponse, read_record

SECRET = "fake-sensitive-api-key"
QUESTION = "What is the return period?"


def test_lifespan_closes_owned_context_on_shutdown(rig, tracked_http_client):
    context = replace(rig.context, http_client=tracked_http_client)
    app = api.create_app(settings=rig.settings, context_factory=lambda _: context)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert not tracked_http_client.is_closed
    assert tracked_http_client.is_closed
    context.close()
    tracked_http_client.close.assert_called_once_with()


@pytest.fixture
def rig(tmp_path: Path, section_index_factory):
    data = tmp_path / "data"
    data.mkdir()
    (data / "returns.md").write_text(
        "# Vrácení zboží\n\n## Lhůta {#return-window}\n"
        "Zboží můžete vrátit do 30 dnů.\nPoplatek je 79 Kč.\n",
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        openai_api_key=SecretStr(SECRET),
        data_dir=data,
        runtime_dir=tmp_path / "runtime",
        embedding_dimensions=8,
    )
    corpus = load_corpus(data)
    ledger = BudgetLedger(
        settings.spend_db_path,
        settings.budget_cap_usd,
        settings.price_table_version,
        price_table=settings.price_table,
    )
    fake = DeterministicFakeEmbedding(size=8)
    calls = []

    def embed(**kwargs):
        calls.append("embedding")
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=vector)
                for i, vector in enumerate(fake.embed_documents(kwargs["input"]))
            ],
            usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
        )

    client = SimpleNamespace(embeddings=SimpleNamespace(create=embed))
    embeddings = UsageTrackingEmbeddings(settings, ledger, client=client)
    store = section_index_factory(section_documents(corpus), embeddings)
    results = {
        "generate": {
            "status": "answered",
            "answer_language": "en",
            "claims": [
                {
                    "text": "You can return goods within 30 days.",
                    "citations": ["returns.return-window.1"],
                }
            ],
        },
        "verify": {
            "verdicts": [{"claim_index": 0, "verdict": "supported", "reason": "Explicit deadline."}]
        },
    }

    def invoke(node, messages):
        calls.append(node)
        result = results[node]
        if isinstance(result, Exception):
            raise result
        return {
            "parsed": result,
            "parsing_error": None,
            "raw": AIMessage(
                content=json.dumps(result),
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            ),
        }

    context = AnswerContext(
        settings=settings,
        corpus=corpus,
        ledger=ledger,
        embeddings=embeddings,
        store=store,
        generator=RunnableLambda(lambda messages: invoke("generate", messages)),
        verifier=RunnableLambda(lambda messages: invoke("verify", messages)),
    )
    return SimpleNamespace(
        context=context, settings=settings, results=results, calls=calls, client=client
    )


@pytest.fixture
def app(rig):
    return api.create_app(settings=rig.settings, context_factory=lambda settings: rig.context)


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def assert_headers(response, *, ask=True):
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["content-security-policy"] == (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    )
    assert "access-control-allow-origin" not in response.headers
    if ask:
        assert response.headers["cache-control"] == "no-store"
    assert SECRET not in response.text
    assert "Traceback" not in response.text
    assert "/runtime" not in response.text


def assert_error(response, status, code, *, ask=True):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str)
    assert body["error"]["message"]
    assert_headers(response, ask=ask)


def test_health_ready_and_context_built_once(rig):
    factory_calls = []

    def factory(settings):
        factory_calls.append(settings)
        return rig.context

    app = api.create_app(settings=rig.settings, context_factory=factory)
    assert factory_calls == []
    with TestClient(app) as client:
        for _ in range(2):
            response = client.get("/api/health")
            assert response.status_code == 200
            assert response.json() == {
                "status": "ok",
                "corpus_hash": rig.context.corpus.corpus_hash,
                "document_count": 1,
                "sentence_count": 2,
                "index_ready": True,
                "key_configured": True,
                "remaining_budget_usd": "3.000000",
                "version": __version__,
            }
            assert_headers(response, ask=False)
        assert rig.calls == []
    assert factory_calls == [rig.settings]


@pytest.mark.parametrize("language", ["en", "cs"])
def test_ask_answered_is_public_live_projection(client, rig, language):
    rig.results["generate"]["answer_language"] = language
    response = client.post("/api/ask", json={"question": f"  {QUESTION}  "})
    assert response.status_code == 200
    body = response.json()
    public = AskResponse.model_validate(body)
    assert public.mode == "live"
    assert public.status == "answered"
    assert public.answer_language == language
    assert public.refusal_code is None
    assert public.claims[0].citations == ("returns.return-window.1",)
    assert public.sources[0].doc_id == "returns"
    assert public.metrics.input_tokens > 0
    assert isinstance(body["metrics"]["cost_usd"], str)
    record = read_record(next(rig.settings.runs_dir.glob("*.json")))
    assert record.origin == "api"
    assert record.question == QUESTION
    assert body == record.to_response().model_dump(mode="json")
    assert set(body) == {
        "run_id",
        "mode",
        "status",
        "answer_language",
        "refusal_code",
        "claims",
        "sources",
        "metrics",
    }
    assert_headers(response)


def test_ask_refused_is_200_with_no_unverified_text(client, rig):
    rig.results["verify"]["verdicts"][0]["verdict"] = "unsupported"
    response = client.post("/api/ask", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["refusal_code"] == "unsupported"
    assert body["answer_language"] is None
    assert body["claims"] == body["sources"] == []
    assert body["mode"] == "live"
    assert "within 30 days" not in response.text
    assert_headers(response)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": "ab"},
        {"question": "x" * 501},
        {"question": "  a  "},
        {"question": "   "},
        {"question": 123},
        {"question": None},
        {"question": [SECRET]},
        {"question": QUESTION, "extra": SECRET},
        [SECRET],
    ],
)
def test_validation_is_safe_and_does_not_run_pipeline(client, rig, body):
    assert_error(client.post("/api/ask", json=body), 422, "invalid_request")
    assert rig.calls == []
    assert not rig.settings.runs_dir.exists()


def test_malformed_json_and_content_type_use_safe_envelope(client):
    for body, content_type in [(b'{"question":', "application/json"), (SECRET, "text/plain")]:
        assert_error(
            client.post("/api/ask", content=body, headers={"Content-Type": content_type}),
            422,
            "invalid_request",
        )


@pytest.mark.parametrize("length", [3, 500])
def test_question_length_boundaries(client, length):
    assert client.post("/api/ask", json={"question": "x" * length}).status_code == 200


def test_oversized_invalid_json_is_rejected_before_json_parsing(client, rig, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Oversized body must be rejected before JSON parsing")

    monkeypatch.setattr(api.Request, "json", forbidden)
    response = client.post(
        "/api/ask",
        content=SECRET.encode() + b"{" * 4096,
        headers={"Content-Type": "application/json", "Content-Length": "1"},
    )
    assert_error(response, 413, "payload_too_large")
    assert rig.calls == []


def test_exact_4k_body_is_accepted(client):
    body = json.dumps({"question": QUESTION}).encode()
    response = client.post(
        "/api/ask",
        content=body + b" " * (4096 - len(body)),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200


def test_chunked_body_without_content_length_is_bounded_before_parsing(app, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Oversized body must not be parsed")

    monkeypatch.setattr(api.Request, "json", forbidden)

    async def request():
        messages = iter(
            [
                {"type": "http.request", "body": b"{" * 2048, "more_body": True},
                {"type": "http.request", "body": b"{" * 2049, "more_body": True},
            ]
        )
        sent = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        async with app.router.lifespan_context(app):
            await app(
                {"type": "http", "method": "POST", "path": "/api/ask", "headers": []},
                receive,
                send,
            )
        assert sent[0]["status"] == 413
        assert json.loads(sent[1]["body"])["error"]["code"] == "payload_too_large"

    asyncio.run(request())


@pytest.mark.parametrize("failure", [IndexMissingError, IndexMismatchError])
def test_index_unavailable_startup_still_serves_health_and_documents(rig, failure):
    def factory(settings):
        raise failure(SECRET)

    app = api.create_app(settings=rig.settings, context_factory=factory)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "not_ready"
        assert health.json()["index_ready"] is False
        assert health.json()["key_configured"] is True
        assert health.json()["corpus_hash"] == rig.context.corpus.corpus_hash
        assert health.json()["document_count"] == 1
        assert health.json()["sentence_count"] == 2
        assert health.json()["remaining_budget_usd"] == "3.000000"
        assert_headers(health, ask=False)
        assert_error(client.post("/api/ask", json={"question": QUESTION}), 503, "index_unavailable")
        assert client.get("/api/documents/returns").status_code == 200
    assert rig.calls == []


@pytest.mark.parametrize("with_index", [False, True])
def test_default_factory_missing_key_and_independent_index_readiness(rig, with_index):
    if with_index:
        build_index(rig.context.corpus, rig.settings, rig.context.ledger, client=rig.client)
    rig.calls.clear()
    rig.settings.openai_api_key = None
    with TestClient(api.create_app(settings=rig.settings)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "not_ready"
        assert health.json()["key_configured"] is False
        assert health.json()["index_ready"] is with_index
        assert_headers(health, ask=False)
        assert_error(
            client.post("/api/ask", json={"question": QUESTION}), 503, "configuration_error"
        )
        assert client.get("/api/documents/returns").status_code == 200
    assert rig.calls == []


def test_default_factory_missing_index_sends_no_provider_requests(rig):
    with TestClient(api.create_app(settings=rig.settings)) as client:
        assert_error(client.post("/api/ask", json={"question": QUESTION}), 503, "index_unavailable")
    assert rig.calls == []


def test_default_factory_rejects_stale_index_without_provider_requests(rig):
    build_index(rig.context.corpus, rig.settings, rig.context.ledger, client=rig.client)
    rig.calls.clear()
    path = rig.settings.data_dir / "returns.md"
    path.write_text(path.read_text(encoding="utf-8") + "Další podmínka.\n", encoding="utf-8")
    with TestClient(api.create_app(settings=rig.settings)) as client:
        health = client.get("/api/health")
        assert health.json()["status"] == "not_ready"
        assert health.json()["index_ready"] is False
        assert_error(client.post("/api/ask", json={"question": QUESTION}), 503, "index_unavailable")
    assert rig.calls == []


def test_invalid_settings_do_not_prevent_startup_or_echo_values(rig, monkeypatch):
    monkeypatch.setenv("API_PORT", SECRET)
    with TestClient(api.create_app()) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "not_ready"
        assert health.json()["corpus_hash"] is None
        assert health.json()["document_count"] == 0
        assert health.json()["sentence_count"] == 0
        assert health.json()["remaining_budget_usd"] == "0"
        assert_headers(health, ask=False)
        assert_error(client.post("/api/ask", json={"question": SECRET}), 503, "configuration_error")


def test_exhausted_budget_is_503_without_provider_dispatch(client, rig):
    rig.context.ledger.reserve("test", rig.settings.generation_model, Decimal("3"))
    health = client.get("/api/health")
    assert health.json()["remaining_budget_usd"] == "0.000000"
    assert health.json()["status"] == "not_ready"
    assert_error(client.post("/api/ask", json={"question": QUESTION}), 503, "budget_exhausted")
    assert rig.calls == []
    record = read_record(next(rig.settings.runs_dir.glob("*.json")))
    assert record.error.code == "budget_exceeded"


def test_provider_error_has_safe_502_and_persisted_record(client, rig):
    rig.results["generate"] = RuntimeError(f"Provider echoed {SECRET}")
    assert_error(client.post("/api/ask", json={"question": QUESTION}), 502, "provider_error")
    record = read_record(next(rig.settings.runs_dir.glob("*.json")))
    assert record.status == "error"
    assert record.mode == "live"
    assert record.error is not None


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (PipelineError("private-code", SECRET), 502, "provider_error"),
        (IndexMissingError(SECRET), 503, "index_unavailable"),
        (IndexLockedError(), 503, "index_unavailable"),
        (IndexMismatchError(SECRET), 503, "index_unavailable"),
        (MissingAPIKeyError(SECRET), 503, "configuration_error"),
        (RuntimeError(SECRET), 500, "internal_error"),
    ],
)
def test_runtime_errors_are_safe_and_release_lock(client, monkeypatch, error, status, code, caplog):
    original = pipeline.answer

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pipeline, "answer", fail)
    assert_error(client.post("/api/ask", json={"question": SECRET}), status, code)
    monkeypatch.setattr(pipeline, "answer", original)
    assert client.post("/api/ask", json={"question": QUESTION}).status_code == 200
    assert SECRET not in caplog.text


def test_concurrent_ask_rejected_without_queue_and_health_remains_responsive(client, monkeypatch):
    started, release = Event(), Event()
    original = pipeline.answer
    caller_threads, worker_threads = [], []

    def blocking(*args, **kwargs):
        worker_threads.append(get_ident())
        started.set()
        assert release.wait(10), "Test did not release the fake pipeline"
        return original(*args, **kwargs)

    def first_request():
        caller_threads.append(get_ident())
        return client.post("/api/ask", json={"question": QUESTION})

    monkeypatch.setattr(pipeline, "answer", blocking)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(first_request)
        try:
            assert started.wait(10), "Fake pipeline did not start"
            assert_error(client.post("/api/ask", json={"question": QUESTION}), 429, "busy")
            assert client.get("/api/health").status_code == 200
        finally:
            release.set()
        assert first.result(timeout=10).status_code == 200
    assert len(worker_threads) == 1
    assert worker_threads != caller_threads
    assert client.post("/api/ask", json={"question": QUESTION}).status_code == 200


def test_document_contains_only_frontend_source_fields(client):
    response = client.get("/api/documents/returns")
    assert response.status_code == 200
    assert response.json() == {
        "title": "Vrácení zboží",
        "file_name": "returns.md",
        "sections": [
            {
                "heading": "Lhůta",
                "key": "returns.return-window",
                "sentences": [
                    {"id": "returns.return-window.1", "text": "Zboží můžete vrátit do 30 dnů."},
                    {"id": "returns.return-window.2", "text": "Poplatek je 79 Kč."},
                ],
            }
        ],
    }
    assert_headers(response, ask=False)


@pytest.mark.parametrize(
    "doc_id", ["unknown", "Returns", "returns.json", "return_id", "%2Fetc%2Fpasswd"]
)
def test_documents_missing_and_invalid_ids_are_404(client, doc_id):
    assert_error(client.get(f"/api/documents/{doc_id}"), 404, "not_found", ask=False)


def test_unknown_route_and_wrong_method_use_error_envelope(client):
    assert_error(client.get(f"/api/{SECRET}"), 404, "not_found", ask=False)
    assert_error(client.get("/api/ask"), 422, "invalid_request")


def test_cors_is_disabled_and_openapi_is_available(client):
    response = client.options(
        "/api/ask",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert_error(response, 422, "invalid_request")
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_serve_wires_local_settings_without_starting_app(rig, monkeypatch):
    calls = []
    rig.settings.api_port = 5199

    def run(app, **kwargs):
        assert isinstance(app, FastAPI)
        assert not hasattr(app.state, "api")
        calls.append(kwargs)

    monkeypatch.setattr(uvicorn, "run", run)
    main(["serve"], settings=rig.settings)
    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 5199,
            "reload": False,
            "workers": 1,
            "proxy_headers": False,
            "access_log": False,
        }
    ]
    assert rig.calls == []


def test_serves_built_web_app_with_strict_csp_and_keeps_api_errors(rig, tmp_path):
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>Citarilo</title>", encoding="utf-8")
    (web / "assets" / "app.js").write_text("export {};", encoding="utf-8")
    with TestClient(api.create_app(settings=rig.settings, web_dir=web)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "<title>Citarilo</title>" in page.text
        csp = page.headers["Content-Security-Policy"]
        assert "script-src 'self'" in csp and "'unsafe-inline'" not in csp
        assert "frame-ancestors 'none'" in csp
        assert page.headers["Cache-Control"] == "no-cache"
        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert asset.headers["Cache-Control"] == "public, max-age=31536000, immutable"
        assert_error(client.get("/api/unknown"), 404, "not_found", ask=False)
        assert client.get("/api/documents/returns").status_code == 200


def test_without_a_build_only_the_api_is_served(rig, tmp_path):
    with TestClient(api.create_app(settings=rig.settings, web_dir=tmp_path / "missing")) as client:
        response = client.get("/")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


def test_held_local_index_is_unavailable_without_provider_calls(rig):
    from citarilo.index import load_index

    build_index(rig.context.corpus, rig.settings, rig.context.ledger, client=rig.client)
    rig.calls.clear()
    with (
        load_index(rig.context.corpus, rig.settings, rig.context.embeddings),
        TestClient(api.create_app(settings=rig.settings)) as client,
    ):
        health = client.get("/api/health").json()
        assert health["status"] == "not_ready" and not health["index_ready"]
        assert_error(client.post("/api/ask", json={"question": QUESTION}), 503, "index_unavailable")
    assert rig.calls == []


def test_independent_readiness_probe_releases_index_lock(rig):
    from citarilo.index import load_index

    build_index(rig.context.corpus, rig.settings, rig.context.ledger, client=rig.client)
    rig.calls.clear()
    rig.settings.openai_api_key = None
    with TestClient(api.create_app(settings=rig.settings)) as client:
        assert client.get("/api/health").json()["index_ready"]
        with load_index(rig.context.corpus, rig.settings, rig.context.embeddings):
            pass  # A not-ready API must leave the directory available to ingest.
    assert rig.calls == []
