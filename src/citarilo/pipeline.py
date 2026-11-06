"""Evidence-gated answering with explicit LangGraph flow and durable usage audits."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field, replace
from decimal import Decimal
from time import perf_counter
from typing import Literal, Self, TypedDict
from uuid import uuid4

import httpx2
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langsmith import tracing_context
from openai import OpenAI
from openai.types.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from citarilo.budget import BudgetExceeded, BudgetLedger, actual_cost, max_cost
from citarilo.checks import check_claims
from citarilo.config import Settings
from citarilo.corpus import Corpus, load_corpus
from citarilo.index import MissingAPIKeyError, UsageTrackingEmbeddings, load_index, search
from citarilo.prompts import PROMPT_VERSION, generation_messages, verification_messages
from citarilo.records import (
    AnswerLanguage,
    CallRecord,
    CandidateClaim,
    CheckResult,
    Claim,
    ClaimVerdict,
    ConfigSnapshot,
    NodeTiming,
    Origin,
    RefusalCode,
    RetrievedSection,
    RunError,
    RunRecord,
    Source,
    get_git_info,
    write_record,
)

# Each UTF-8 byte bounds one token. The additional 1024 tokens cover message
# framing, roles, schema name/wrapper and Responses API protocol overhead for
# our fixed two-message, tool-free requests. Output includes reasoning tokens.
FRAMING_MARGIN_TOKENS = 1024
type StructuredRunnable = Runnable[list[BaseMessage], object]


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GeneratedClaim(OutputModel):
    text: str = Field(min_length=1, max_length=1000, pattern=r"\S")
    citations: list[str] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_citations(self) -> Self:
        if any(not sid.strip() for sid in self.citations) or len(set(self.citations)) != len(
            self.citations
        ):
            raise ValueError("Citations must be nonempty and distinct")
        return self


class GenerationOutput(OutputModel):
    status: Literal["answered", "insufficient_evidence"]
    answer_language: AnswerLanguage
    claims: list[GeneratedClaim] = Field(max_length=4)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if (self.status == "answered") != bool(self.claims):
            raise ValueError("Only answered output may contain claims, and must contain claims")
        return self


class VerificationVerdict(OutputModel):
    claim_index: int = Field(ge=0)
    verdict: Literal["supported", "unsupported", "contradicted"]
    reason: str = Field(min_length=1, max_length=500, pattern=r"\S")


class VerificationOutput(OutputModel):
    verdicts: list[VerificationVerdict] = Field(min_length=1, max_length=4)


class InvalidQuestionError(ValueError):
    """Question validation failure, suitable for an HTTP 422 response."""


class PipelineError(RuntimeError):
    """Safe technical failure, suitable for HTTP 502; record is the local audit."""

    def __init__(self, code: str, message: str, *, record: RunRecord | None = None) -> None:
        self.code = code
        self.message = message
        self.record = record
        super().__init__(message)


def _token_usage(usage: object) -> tuple[int, int]:
    if (
        not isinstance(usage, dict)
        or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
    ):
        raise PipelineError("usage_missing", "The model response has no valid usage.")
    return usage["input_tokens"], usage["output_tokens"]


class _UsageValidatedOpenAI(OpenAI):
    """Validate original usage before SDK coercion and LangChain's zero defaults.

    The installed SDK calls this hook with decoded provider JSON, including for
    with_raw_response.create(). Keep transport-level tests on this boundary.
    """

    def _process_response_data[T](
        self, *, data: object, cast_to: type[T], response: httpx2.Response
    ) -> T:
        if cast_to is Response:
            _token_usage(data.get("usage") if isinstance(data, dict) else None)
        return super()._process_response_data(data=data, cast_to=cast_to, response=response)


class AnswerState(TypedDict, total=False):
    question: str
    answer_language: AnswerLanguage
    retrieved_sections: tuple[RetrievedSection, ...]
    candidate_claims: tuple[CandidateClaim, ...]
    check_results: tuple[CheckResult, ...]
    verdicts: tuple[ClaimVerdict, ...]
    status: Literal["answered", "refused"]
    refusal_code: RefusalCode | None
    claims: tuple[Claim, ...]
    sources: tuple[Source, ...]


@dataclass
class Recorder:
    """Run-local audit, retaining successful state updates even if invoke raises."""

    state: AnswerState = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    timings: list[NodeTiming] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerContext:
    """Caller-owned dependencies. Reuse sequentially; answer creates a fresh recorder."""

    settings: Settings
    corpus: Corpus
    store: InMemoryVectorStore
    embeddings: UsageTrackingEmbeddings
    ledger: BudgetLedger
    generator: StructuredRunnable
    verifier: StructuredRunnable
    mode: Literal["live", "fake"] = "fake"
    group_id: str | None = None
    recorder: Recorder = field(default_factory=Recorder)

    def __post_init__(self) -> None:
        if (
            self.store.embeddings is not self.embeddings
            or self.embeddings.ledger is not self.ledger
            or self.embeddings.group_id != self.group_id
            or self.embeddings.settings is not self.settings
        ):
            raise ValueError("Context must share its store adapter, settings, ledger and group")


def input_token_bound(messages: list[BaseMessage], schema: type[BaseModel]) -> int:
    """Conservative bound, independent of tokenizers and provider model aliases."""
    rendered = json.dumps(
        [{"role": m.type, "content": m.content} for m in messages], ensure_ascii=False
    )
    # ASCII serialization also bounds the SDK's escaped schema serialization.
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True)
    return len(rendered.encode("utf-8")) + len(schema_json) + FRAMING_MARGIN_TOKENS


def _invoke[Output: BaseModel](
    context: AnswerContext,
    node: str,
    runnable: StructuredRunnable,
    messages: list[BaseMessage],
    schema: type[Output],
    model: str,
) -> Output:
    settings = context.settings
    bound = input_token_bound(messages, schema)
    cost = max_cost(model, bound, settings.max_output_tokens, price_table=settings.price_table)
    with context.ledger.reserve(node, model, cost, group_id=context.group_id) as reservation:
        started = perf_counter()
        input_tokens = output_tokens = 0
        charged = cost
        try:
            reservation.mark_sent()
            try:
                response = runnable.invoke(messages)
            except PipelineError:
                raise
            except Exception:
                raise PipelineError("provider_error", "The model request failed.") from None
            raw = response.get("raw") if isinstance(response, dict) else None
            usage = raw.usage_metadata if isinstance(raw, AIMessage) else None
            input_tokens, output_tokens = _token_usage(usage)
            reservation.settle(input_tokens, output_tokens)
            charged = actual_cost(
                model, input_tokens, output_tokens, price_table=settings.price_table
            )
            if response.get("parsing_error") is not None:
                raise PipelineError("model_contract_error", "The model output violated its schema.")
            try:
                # LangChain's JSON parser can repair malformed JSON. Enforce the
                # actual response contract locally, after known usage is settled.
                return schema.model_validate(json.loads(raw.text))
            except (ValidationError, TypeError, ValueError):
                raise PipelineError(
                    "model_contract_error", "The model output violated its schema."
                ) from None
        finally:
            # Unknown usage has zero known tokens and the full conservative charge.
            context.recorder.calls.append(
                CallRecord(
                    node=node,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=charged,
                    reservation_id=reservation.id,
                    duration_ms=(perf_counter() - started) * 1000,
                )
            )


def _embedding_reservations(context: AnswerContext, *, after: int = 0) -> list[sqlite3.Row]:
    # The embedding adapter exposes successful usage, but no failure callback.
    # Read existing durable entries to retain uncertain query calls in the audit.
    with closing(sqlite3.connect(context.ledger.db_path)) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            "SELECT rowid, * FROM reservations WHERE rowid > ? AND operation = ? "
            "AND model = ? AND group_id IS ? ORDER BY rowid",
            (
                after,
                context.embeddings.operation,
                context.settings.embedding_model,
                context.group_id,
            ),
        ).fetchall()


def _retrieve(state: AnswerState, context: AnswerContext) -> AnswerState:
    with closing(sqlite3.connect(context.ledger.db_path)) as connection:
        before = connection.execute("SELECT COALESCE(MAX(rowid), 0) FROM reservations").fetchone()[
            0
        ]
    usage_start = len(context.embeddings.usage)
    started = perf_counter()
    try:
        ranked = search(context.store, state["question"], context.settings.top_k)
        titles = {doc.doc_id: doc.title for doc in context.corpus.documents}
        sections = []
        for key, score, _ in ranked:
            section = context.corpus.section(key)
            sources = tuple(
                Source(
                    sentence_id=s.id,
                    text=s.text,
                    doc_id=section.doc_id,
                    doc_title=titles[section.doc_id],
                    heading=section.heading,
                )
                for s in section.sentences
            )
            sections.append(
                RetrievedSection(
                    section_key=key,
                    doc_id=section.doc_id,
                    doc_title=titles[section.doc_id],
                    heading=section.heading,
                    score=score,
                    sentences=sources,
                )
            )
        return {"retrieved_sections": tuple(sections)}
    finally:
        usages = context.embeddings.usage[usage_start:]
        for usage in usages:
            context.recorder.calls.append(
                CallRecord(
                    node="retrieve",
                    model=context.settings.embedding_model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=0,
                    cost_usd=usage.cost_usd,
                    reservation_id=usage.reservation_id,
                    duration_ms=usage.duration_ms,
                )
            )
        # One validated question is one embedding request. Only failures without
        # usage need a ledger fallback; the normal path uses the adapter directly.
        for row in () if usages else _embedding_reservations(context, after=before):
            if row["status"] in ("abandoned", "in_flight"):
                context.recorder.calls.append(
                    CallRecord(
                        node="retrieve",
                        model=row["model"],
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=Decimal(f"{row['reserved_micro_usd']}e-6"),
                        reservation_id=row["id"],
                        duration_ms=(perf_counter() - started) * 1000,
                    )
                )


def _generate(state: AnswerState, context: AnswerContext) -> AnswerState:
    generated = _invoke(
        context,
        "generate",
        context.generator,
        generation_messages(state["question"], state["retrieved_sections"]),
        GenerationOutput,
        context.settings.generation_model,
    )
    return {
        "answer_language": generated.answer_language,
        "candidate_claims": tuple(
            CandidateClaim(text=c.text, citations=tuple(c.citations)) for c in generated.claims
        ),
        "refusal_code": "insufficient_evidence"
        if generated.status == "insufficient_evidence"
        else None,
    }


def _check(state: AnswerState, context: AnswerContext) -> AnswerState:
    sentences = {
        s.sentence_id: s.text for section in state["retrieved_sections"] for s in section.sentences
    }
    checks = check_claims(
        state["candidate_claims"],
        sentences,
        answer_language=state["answer_language"],
        question=state["question"],
    )
    return {
        "check_results": checks,
        "refusal_code": "check_failed" if any(not check.ok for check in checks) else None,
    }


def _verify(state: AnswerState, context: AnswerContext) -> AnswerState:
    verified = _invoke(
        context,
        "verify",
        context.verifier,
        verification_messages(
            state["question"], state["candidate_claims"], state["retrieved_sections"]
        ),
        VerificationOutput,
        context.settings.verification_model,
    )
    verdicts = tuple(ClaimVerdict(**v.model_dump()) for v in verified.verdicts)
    # Retain malformed index assignments for diagnostics, but never expose claims.
    context.recorder.state["verdicts"] = verdicts
    if sorted(v.claim_index for v in verdicts) != list(range(len(state["candidate_claims"]))):
        raise PipelineError(
            "model_contract_error", "The verifier must judge every claim exactly once."
        )
    refusal = (
        "contradicted"
        if any(v.verdict == "contradicted" for v in verdicts)
        else ("unsupported" if any(v.verdict == "unsupported" for v in verdicts) else None)
    )
    return {"verdicts": verdicts, "refusal_code": refusal}


def _assemble(state: AnswerState, context: AnswerContext) -> AnswerState:
    claims = tuple(Claim(**c.model_dump()) for c in state["candidate_claims"])
    cited = {sid for claim in claims for sid in claim.citations}
    sources = tuple(
        s
        for section in state["retrieved_sections"]
        for s in section.sentences
        if s.sentence_id in cited
    )
    return {"status": "answered", "claims": claims, "sources": sources}


def _refuse(state: AnswerState, context: AnswerContext) -> AnswerState:
    return {"status": "refused", "claims": (), "sources": ()}


def _timed(
    name: str, node: Callable[[AnswerState, AnswerContext], AnswerState]
) -> Callable[[AnswerState, Runtime[AnswerContext]], AnswerState]:
    def wrapped(state: AnswerState, runtime: Runtime[AnswerContext]) -> AnswerState:
        recorder = runtime.context.recorder
        recorder.state.update(state)
        started = perf_counter()
        ok = False
        try:
            update = node(state, runtime.context)
            recorder.state.update(update)
            ok = True
            return update
        finally:
            recorder.timings.append(
                NodeTiming(
                    node=name,
                    duration_ms=(perf_counter() - started) * 1000,
                    ok=ok,
                )
            )

    return wrapped


def _route(state: AnswerState) -> Literal["refuse", "continue"]:
    return "refuse" if state.get("refusal_code") else "continue"


def build_graph() -> CompiledStateGraph[AnswerState, AnswerContext, AnswerState, AnswerState]:
    """Compile the acyclic evidence flow; no models, I/O, retries or global state."""
    graph = StateGraph(AnswerState, context_schema=AnswerContext)
    for name, node in (
        ("retrieve", _retrieve),
        ("generate", _generate),
        ("check", _check),
        ("verify", _verify),
        ("assemble", _assemble),
        ("refuse", _refuse),
    ):
        graph.add_node(name, _timed(name, node))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    for node, next_node in (("generate", "check"), ("check", "verify"), ("verify", "assemble")):
        graph.add_conditional_edges(node, _route, {"refuse": "refuse", "continue": next_node})
    graph.add_edge("assemble", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def validate_question(question: str) -> str:
    if not isinstance(question, str) or not 3 <= len(question) <= 500 or len(question.strip()) < 3:
        raise InvalidQuestionError("Question must contain 3–500 characters.")
    return question.strip()


def answer(
    question: str, *, context: AnswerContext, origin: Origin, eval_case_id: str | None = None
) -> RunRecord:
    """Validate, execute and persist, including errors; never return unverified claims.

    BudgetExceeded is persisted then propagated for HTTP 503. Other execution
    failures raise PipelineError with safe details and the persisted error record.
    If persistence fails, PipelineError carries the constructed, unwritten record.
    Invalid input raises before retrieval or filesystem writes. Contexts are for
    sequential reuse (the API must enforce its one-request concurrency limit).
    """
    question = validate_question(question)
    if (
        origin not in ("api", "eval")
        or (origin == "eval") != (eval_case_id is not None)
        or (eval_case_id is not None and not eval_case_id.strip())
    ):
        raise ValueError("Evaluation origin requires a nonempty case ID, and only eval may use it")
    context = replace(context, recorder=Recorder())
    started = perf_counter()
    git = get_git_info()
    error: PipelineError | BudgetExceeded | None = None
    try:
        # Suppress external LangSmith tracing even if the calling shell enables it.
        with tracing_context(enabled=False):
            build_graph().invoke({"question": question}, context=context)
    except BudgetExceeded as exc:
        error = exc
    except PipelineError as exc:
        error = exc
    except Exception:
        error = PipelineError("pipeline_error", "The answering pipeline failed.")
    state = context.recorder.state
    safe_error = None
    if error is not None:
        safe_error = RunError(
            code="budget_exceeded" if isinstance(error, BudgetExceeded) else error.code,
            message="The request cannot fit the remaining budget."
            if isinstance(error, BudgetExceeded)
            else error.message,
        )
    record = RunRecord(
        mode=context.mode,
        origin=origin,
        eval_case_id=eval_case_id,
        git=git,
        corpus_hash=context.corpus.corpus_hash,
        prompt_version=PROMPT_VERSION,
        config=ConfigSnapshot.from_settings(context.settings, top_k=context.settings.top_k),
        question=question,
        retrieved_sections=state.get("retrieved_sections", ()),
        candidate_claims=state.get("candidate_claims", ()),
        check_results=state.get("check_results", ()),
        verdicts=state.get("verdicts", ()),
        status="error" if error else state["status"],
        answer_language=state["answer_language"]
        if not error and state["status"] == "answered"
        else None,
        refusal_code=None if error else state.get("refusal_code"),
        claims=() if error else state.get("claims", ()),
        sources=() if error else state.get("sources", ()),
        calls=tuple(context.recorder.calls),
        node_timings=tuple(context.recorder.timings),
        error=safe_error,
        latency_ms=(perf_counter() - started) * 1000,
    )
    try:
        write_record(record, context.settings.runs_dir)
    except OSError:
        raise PipelineError(
            "record_write_error", "The run record could not be saved.", record=record
        ) from None
    if isinstance(error, BudgetExceeded):
        raise error from None
    if error is not None:
        raise PipelineError(error.code, error.message, record=record) from None
    return record


def create_live_context(settings: Settings, *, group_id: str | None = None) -> AnswerContext:
    """Load local dependencies once; constructing this context sends no requests."""
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise MissingAPIKeyError("Set OPENAI_API_KEY in the environment or .env file.")
    group_id = group_id if group_id is not None else f"ask-{uuid4().hex}"
    corpus = load_corpus(settings.data_dir)
    ledger = BudgetLedger(
        settings.spend_db_path,
        settings.budget_cap_usd,
        settings.price_table_version,
        price_table=settings.price_table,
    )
    embeddings = UsageTrackingEmbeddings(settings, ledger, group_id=group_id, operation="retrieve")
    store = load_index(corpus, settings, embeddings)
    client = _UsageValidatedOpenAI(
        api_key=key.get_secret_value(),
        max_retries=0,
        timeout=settings.call_timeout_seconds,
    )

    def model(name: str, effort: str, schema: type[BaseModel]) -> StructuredRunnable:
        return ChatOpenAI(
            model=name,
            api_key=key,
            client=client.chat.completions,
            root_client=client,
            use_responses_api=True,
            service_tier="default",
            reasoning_effort=effort,
            store=False,
            max_retries=0,
            timeout=settings.call_timeout_seconds,
            max_tokens=settings.max_output_tokens,
        ).with_structured_output(
            schema.model_json_schema(), method="json_schema", strict=True, include_raw=True
        )

    return AnswerContext(
        settings=settings,
        corpus=corpus,
        store=store,
        embeddings=embeddings,
        ledger=ledger,
        generator=model(
            settings.generation_model, settings.generation_reasoning_effort, GenerationOutput
        ),
        verifier=model(
            settings.verification_model, settings.verification_reasoning_effort, VerificationOutput
        ),
        mode="live",
        group_id=group_id,
    )
