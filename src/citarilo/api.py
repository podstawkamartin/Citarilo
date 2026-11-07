"""Local HTTP boundary for evidence-gated answers and their original sources."""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from threading import Lock
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from citarilo import __version__, pipeline
from citarilo.budget import BudgetExceeded, BudgetLedger
from citarilo.config import Settings
from citarilo.corpus import Corpus, CorpusError, load_corpus
from citarilo.index import (
    IndexMismatchError,
    IndexMissingError,
    MissingAPIKeyError,
    UsageTrackingEmbeddings,
    load_index,
)
from citarilo.records import AskResponse

_MAX_BODY_BYTES = 4096
_DOCUMENT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_ERRORS = {
    "invalid_request": (422, "The request is invalid."),
    "payload_too_large": (413, "The request body exceeds 4 KiB."),
    "busy": (429, "Another question is being processed. Try again later."),
    "not_found": (404, "The requested resource was not found."),
    "budget_exhausted": (503, "The request cannot fit the remaining budget."),
    "index_unavailable": (503, "The index is unavailable. Run citarilo ingest explicitly."),
    "configuration_error": (503, "The service is not configured for live answers."),
    "provider_error": (502, "The answering provider could not complete the request."),
    "internal_error": (500, "An internal error occurred."),
}


def _error(code: str) -> JSONResponse:
    status, message = _ERRORS[code]
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


class _APIGuards:
    """Bound bytes before FastAPI parses JSON and sanitize unexpected failures.

    Catch failures here so Starlette's outer server-error middleware does not
    re-raise them to the server logger with potentially sensitive exception text.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False
        is_ask = scope["path"] == "/api/ask"

        async def safe_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                if scope["path"].startswith("/api/"):
                    headers["Content-Security-Policy"] = (
                        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
                    )
                if is_ask:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        try:
            if is_ask:
                body = bytearray()
                async for chunk in Request(scope, receive).stream():
                    if len(body) + len(chunk) > _MAX_BODY_BYTES:
                        await _error("payload_too_large")(scope, receive, safe_send)
                        return
                    body.extend(chunk)
                replayed = False

                async def bounded_receive() -> Message:
                    nonlocal replayed
                    if not replayed:
                        replayed = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                await self.app(scope, bounded_receive, safe_send)
            else:
                await self.app(scope, receive, safe_send)
        except Exception:
            if not response_started:
                await _error("internal_error")(scope, receive, safe_send)


class AskRequest(BaseModel):
    """Accept only a string question satisfying the pipeline's exact rules."""

    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=3, max_length=500)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        return pipeline.validate_question(value)


class HealthResponse(BaseModel):
    """Readiness without credentials, exception details or local paths."""

    status: Literal["ok", "not_ready"]
    corpus_hash: str | None
    index_ready: bool
    key_configured: bool
    remaining_budget_usd: str
    version: str


class DocumentSentence(BaseModel):
    id: str
    text: str


class DocumentSection(BaseModel):
    heading: str
    key: str
    sentences: list[DocumentSentence]


class DocumentResponse(BaseModel):
    title: str
    sections: list[DocumentSection]


@dataclass
class _AppState:
    corpus: Corpus | None = None
    ledger: BudgetLedger | None = None
    context: pipeline.AnswerContext | None = None
    index_ready: bool = False
    key_configured: bool = False
    error_code: str | None = None


def _load_state(
    settings: Settings | None, context_factory: Callable[[Settings], pipeline.AnswerContext]
) -> _AppState:
    state = _AppState()
    try:
        settings = settings if settings is not None else Settings()
        key = settings.openai_api_key
        state.key_configured = key is not None and bool(key.get_secret_value().strip())
        context = context_factory(settings)
        state.context = replace(context, mode="live")
        state.corpus, state.ledger = context.corpus, context.ledger
        state.index_ready = True
        if not state.key_configured:
            state.error_code = "configuration_error"
        return state
    except (IndexMissingError, IndexMismatchError):
        state.error_code = "index_unavailable"
    except (MissingAPIKeyError, CorpusError, ValidationError, ValueError):
        state.error_code = "configuration_error"
    except Exception:
        state.error_code = "internal_error"

    # Sources and independent readiness checks remain available without a live
    # context. Loading the persisted index never embeds or contacts a provider.
    if settings is not None:
        try:
            state.corpus = load_corpus(settings.data_dir)
            state.ledger = BudgetLedger(
                settings.spend_db_path,
                settings.budget_cap_usd,
                settings.price_table_version,
                price_table=settings.price_table,
            )
            load_index(state.corpus, settings, UsageTrackingEmbeddings(settings, state.ledger))
            state.index_ready = True
        except (IndexMissingError, IndexMismatchError):
            pass
        except (CorpusError, ValueError):
            state.error_code = "configuration_error"
        except Exception:
            state.error_code = "internal_error"
    return state


def create_app(
    *,
    context_factory: Callable[[Settings], pipeline.AnswerContext] = pipeline.create_live_context,
    settings: Settings | None = None,
) -> FastAPI:
    """Build an app; initialize one caller-owned context during lifespan startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.api = await run_in_threadpool(_load_state, settings, context_factory)
        try:
            yield
        finally:
            if app.state.api.context is not None:
                await run_in_threadpool(app.state.api.context.close)

    app = FastAPI(title="Citarilo", version=__version__, lifespan=lifespan)
    app.add_middleware(_APIGuards)
    ask_lock = Lock()

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(pipeline.InvalidQuestionError)
    async def invalid_request(request: Request, exc: Exception) -> JSONResponse:
        return _error("invalid_request")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        # Do not echo router details, request values or custom exception messages.
        return _error("not_found" if exc.status_code == 404 else "invalid_request")

    @app.get("/api/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        state: _AppState = request.app.state.api
        remaining = state.ledger.remaining_usd() if state.ledger is not None else 0
        return HealthResponse(
            status="ok" if state.error_code is None and remaining > 0 else "not_ready",
            corpus_hash=state.corpus.corpus_hash if state.corpus is not None else None,
            index_ready=state.index_ready,
            key_configured=state.key_configured,
            remaining_budget_usd=str(remaining),
            version=__version__,
        )

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse | JSONResponse:
        state: _AppState = request.app.state.api
        if state.error_code is not None:
            return _error(state.error_code)
        if not ask_lock.acquire(blocking=False):
            return _error("busy")
        try:
            record = await run_in_threadpool(
                pipeline.answer, body.question, context=state.context, origin="api"
            )
            return record.to_response()
        except BudgetExceeded:
            return _error("budget_exhausted")
        except (IndexMissingError, IndexMismatchError):
            return _error("index_unavailable")
        except MissingAPIKeyError:
            return _error("configuration_error")
        except pipeline.PipelineError:
            return _error("provider_error")
        finally:
            ask_lock.release()

    @app.get("/api/documents/{doc_id}", response_model=DocumentResponse)
    def document(doc_id: str, request: Request) -> DocumentResponse | JSONResponse:
        state: _AppState = request.app.state.api
        if not _DOCUMENT_ID.fullmatch(doc_id) or state.corpus is None:
            return _error("not_found")
        doc = next((doc for doc in state.corpus.documents if doc.doc_id == doc_id), None)
        if doc is None:
            return _error("not_found")
        return DocumentResponse(
            title=doc.title,
            sections=[
                DocumentSection(
                    heading=section.heading,
                    key=section.key,
                    sentences=[
                        DocumentSentence(id=sentence.id, text=sentence.text)
                        for sentence in section.sentences
                    ],
                )
                for section in doc.sections
            ],
        )

    return app
