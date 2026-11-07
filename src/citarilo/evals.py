"""Evaluation contracts, JSONL loaders, and budgeted retrieval evaluation.

All required facts and verifier claims are English, regardless of question language.
``expected_refusal`` records the anticipated diagnostic reason, not an exact-match
grading requirement: an unanswerable case passes the refusal-status check with ANY
of the four allowed refusal reasons. It must never pass with an answered status.
Heldout cases are reserved for one final evaluation, not iterative tuning.
"""

import json
import os
import tempfile
from contextlib import closing, nullcontext
from copy import copy
from dataclasses import replace
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from citarilo import pipeline
from citarilo.budget import BudgetExceeded, BudgetLedger, max_cost
from citarilo.config import Settings
from citarilo.corpus import Corpus, load_corpus
from citarilo.index import (
    EmbeddingsClient,
    EmbeddingUsage,
    UsageTrackingEmbeddings,
    load_index,
    search,
)
from citarilo.prompts import PROMPT_VERSION, verification_messages
from citarilo.records import (
    CallRecord,
    CandidateClaim,
    CheckResult,
    Claim,
    ClaimVerdict,
    ConfigSnapshot,
    GitInfo,
    RecordModel,
    RetrievedSection,
    RunError,
    RunRecord,
    Source,
    get_git_info,
)

KebabId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
SectionKey = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*$"),
]
SentenceId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*\.[1-9][0-9]*$"),
]
RefusalReason = Literal["insufficient_evidence", "check_failed", "unsupported", "contradicted"]
Tag = Literal[
    "en",
    "cs",
    "cs-ascii",
    "multi-section",
    "condition",
    "confusable",
    "unanswerable",
    "injection-question",
    "injection-document",
]


class EvalCase(BaseModel):
    """A hand-labelled question with sentence evidence and retrieval expectations.

    JSON arrays load into immutable tuples. Corpus membership is checked by the
    offline validity tests, rather than coupling these contracts to corpus I/O.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: KebabId
    split: Literal["dev", "heldout"]
    question: Text
    language: Literal["en", "cs", "cs-ascii"]
    expected_status: Literal["answered", "refused"]
    expected_refusal: RefusalReason | None
    required_facts: tuple[Text, ...]
    forbidden_claims: tuple[Text, ...]
    supporting_sentence_ids: tuple[SentenceId, ...]
    expected_section_keys: tuple[SectionKey, ...]
    tags: tuple[Tag, ...]

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        if self.expected_status == "refused":
            if self.expected_refusal is None:
                raise ValueError("refused cases require an expected_refusal diagnostic")
            if self.required_facts or self.supporting_sentence_ids or self.expected_section_keys:
                raise ValueError("refused cases must have no required facts or evidence")
        else:
            if self.expected_refusal is not None:
                raise ValueError("answered cases must have expected_refusal=null")
            if not (
                self.required_facts and self.supporting_sentence_ids and self.expected_section_keys
            ):
                raise ValueError("answered cases require facts, supporting sentences and sections")
        for sentence_id in self.supporting_sentence_ids:
            if sentence_id.rsplit(".", 1)[0] not in self.expected_section_keys:
                raise ValueError(
                    f"supporting sentence {sentence_id!r} is outside expected sections"
                )
        return self


class VerifierProbe(BaseModel):
    """One member of a supported/near-miss pair using the same cited sentences."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: KebabId
    claim: Text
    cited_sentence_ids: Annotated[tuple[SentenceId, ...], Field(min_length=1, max_length=3)]
    expected_verdict: Literal["supported", "unsupported", "contradicted"]
    pair_id: KebabId
    note: Text


def _load_jsonl[Record: EvalCase | VerifierProbe](path: Path, model: type[Record]) -> list[Record]:
    records: list[Record] = []
    seen: dict[str, int] = {}
    try:
        with path.open("rb") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    # Validate JSON directly: strict tuple fields accept JSON arrays,
                    # while Python callers still get strict, immutable contracts.
                    record = model.model_validate_json(line)
                except ValidationError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid {model.__name__}: {exc}"
                    ) from exc
                record_id = record.id
                if record_id in seen:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate id {record_id!r} "
                        f"(first seen on line {seen[record_id]})"
                    )
                seen[record_id] = line_number
                records.append(record)
    except OSError as exc:
        raise ValueError(f"{path}:1: cannot read JSONL: {exc.strerror}") from exc
    if not records:
        raise ValueError(f"{path}:1: expected at least one JSON object")
    return records


def load_cases(path: Path) -> list[EvalCase]:
    """Load one case per line; reject invalid or duplicate records with path:line."""
    return _load_jsonl(path, EvalCase)


def load_probes(path: Path) -> list[VerifierProbe]:
    """Load one probe per line; reject invalid or duplicate records with path:line."""
    return _load_jsonl(path, VerifierProbe)


class RetrievalHit(BaseModel):
    """A ranked section and its cosine similarity, without duplicating corpus text."""

    key: str
    score: float


class RetrievalResult(BaseModel):
    """One query; null hit and empty ranks exclude unanswerable cases from recall."""

    id: str
    tags: tuple[str, ...]
    expected_section_keys: tuple[str, ...]
    retrieved: list[RetrievalHit]
    hit: bool | None
    ranks: dict[str, int | None]
    latency_ms: float
    usage: list[EmbeddingUsage]


class RetrievalMetrics(BaseModel):
    """Case recall requires every expected section; section recall is micro-averaged."""

    answerable_cases: int
    unanswerable_cases: int
    hit_cases: int
    expected_sections: int
    retrieved_expected_sections: int
    recall_at_k: float | None
    section_recall_at_k: float | None
    recall_at_5: float | None


class RetrievalReport(BaseModel):
    """Local retrieval audit, with only allowlisted configuration and safe errors."""

    schema_version: Literal[1] = 1
    created_at: datetime
    git: GitInfo
    corpus_hash: str
    embedding_model: str
    embedding_dimensions: int
    price_table_version: str
    top_k: int
    split: Literal["dev", "heldout"]
    cases: list[RetrievalResult]
    metrics: RetrievalMetrics
    usage: dict[str, int | Decimal | str]
    gate: Literal["PASS", "FAIL", "ERROR"]
    gate_threshold: float = 0.9
    error: Literal["budget_exceeded", "retrieval_failed"] | None = None


def estimate_retrieval(cases: list[EvalCase], settings: Settings) -> Decimal:
    """Upper-bound one embedding request per question, including ledger rounding."""
    total = Decimal(0)
    for case in cases:
        bound = len(case.question.encode("utf-8"))
        if not 0 < bound <= 8192:
            raise ValueError("Embedding inputs must contain 1–8192 UTF-8 bytes")
        total += max_cost(settings.embedding_model, bound, 0, price_table=settings.price_table)
    return total


def _write_retrieval_report(report: RetrievalReport, directory: Path, short_id: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = report.created_at.strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / f"retrieval-{timestamp}-{short_id}.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=".retrieval-", suffix=".tmp"
    ) as stream:
        stream.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, destination)
    return destination


def evaluate_retrieval(
    cases: list[EvalCase],
    settings: Settings,
    max_usd: Decimal,
    *,
    split: Literal["dev", "heldout"] = "dev",
    client: EmbeddingsClient | None = None,
) -> tuple[RetrievalReport, Path]:
    """Load once and evaluate; persist partial usage on errors before re-raising.

    The approved gate always measures recall@5. If top_k is overridden, search
    enough sections for both metrics in the same call, reporting only top_k hits.
    """
    if not cases or any(case.split != split for case in cases):
        raise ValueError("Expected nonempty evaluation cases from the selected split")
    estimate_retrieval(cases, settings)  # Validate all inputs before any paid request.
    corpus = load_corpus(settings.data_dir)
    ledger = BudgetLedger(
        settings.spend_db_path,
        settings.budget_cap_usd,
        settings.price_table_version,
        price_table=settings.price_table,
    )
    run_id = uuid4().hex
    group_id = f"retrieval-{run_id}"
    ledger.open_group(group_id, max_usd)
    cap = max_usd.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    embeddings = UsageTrackingEmbeddings(
        settings, ledger, group_id=group_id, client=client, operation="eval-retrieval"
    )
    store = load_index(corpus, settings, embeddings)
    created_at = datetime.now(UTC)
    git = get_git_info()
    results: list[RetrievalResult] = []
    gate_hits = 0
    error = None
    try:
        for case in cases:
            started = perf_counter()
            usage_start = len(embeddings.usage)
            ranked = search(store, case.question, max(5, settings.top_k))
            latency_ms = (perf_counter() - started) * 1000
            hits = [
                RetrievalHit(key=key, score=score) for key, score, _ in ranked[: settings.top_k]
            ]
            positions = {hit.key: rank for rank, hit in enumerate(hits, start=1)}
            expected = case.expected_section_keys
            if expected and set(expected) <= {key for key, _, _ in ranked[:5]}:
                gate_hits += 1
            results.append(
                RetrievalResult(
                    id=case.id,
                    tags=case.tags,
                    expected_section_keys=expected,
                    retrieved=hits,
                    hit=all(key in positions for key in expected) if expected else None,
                    ranks={key: positions.get(key) for key in expected},
                    latency_ms=latency_ms,
                    usage=embeddings.usage[usage_start:],
                )
            )
    except BaseException as exc:
        error = "budget_exceeded" if isinstance(exc, BudgetExceeded) else "retrieval_failed"
        raise
    finally:
        answerable = [result for result in results if result.hit is not None]
        count = len(answerable)
        hit_cases = sum(result.hit is True for result in answerable)
        expected_count = sum(len(result.ranks) for result in answerable)
        found = sum(rank is not None for result in answerable for rank in result.ranks.values())
        recall_at_5 = gate_hits / count if count else None
        spent = ledger.group_spent_usd(group_id)
        report = RetrievalReport(
            created_at=created_at,
            git=git,
            corpus_hash=corpus.corpus_hash,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            price_table_version=settings.price_table_version,
            top_k=settings.top_k,
            split=split,
            cases=results,
            metrics=RetrievalMetrics(
                answerable_cases=count,
                unanswerable_cases=len(results) - count,
                hit_cases=hit_cases,
                expected_sections=expected_count,
                retrieved_expected_sections=found,
                recall_at_k=hit_cases / count if count else None,
                section_recall_at_k=found / expected_count if expected_count else None,
                recall_at_5=recall_at_5,
            ),
            usage={
                "group_id": group_id,
                "prompt_tokens": sum(usage.prompt_tokens for usage in embeddings.usage),
                "total_tokens": sum(usage.total_tokens for usage in embeddings.usage),
                "cost_usd": spent,
                "cap_usd": cap,
                "remaining_usd": max(Decimal(0), cap - spent),
                "global_remaining_usd": ledger.remaining_usd(),
            },
            gate="ERROR"
            if error
            else "PASS"
            if recall_at_5 is not None and recall_at_5 >= 0.9
            else "FAIL",
            error=error,
        )
        path = _write_retrieval_report(report, settings.runtime_dir / "evals", run_id[:8])
    return report, path


# Answer/probe reports deliberately retain the evidence, including rejected candidates,
# so an independent evaluator can grade them offline without rerunning paid calls.


class AnswerResult(EvalCase):
    run_id: str | None = None
    status: Literal["answered", "refused", "error"] | None = None
    refusal_code: RefusalReason | None = None
    answer_language: Literal["en", "cs"] | None = None
    claims: tuple[Claim, ...] = ()
    sources: tuple[Source, ...] = ()
    candidate_claims: tuple[CandidateClaim, ...] = ()
    check_results: tuple[CheckResult, ...] = ()
    verdicts: tuple[ClaimVerdict, ...] = ()
    retrieved_sections: tuple[RetrievedSection, ...] = ()
    calls: tuple[CallRecord, ...] = ()
    latency_ms: float | None = None
    cost_usd: Decimal = Decimal(0)
    error: RunError | None = None


class EvaluationReport(RecordModel):
    schema_version: Literal[1] = 1
    kind: Literal["answers", "probes"]
    created_at: datetime
    git: GitInfo
    corpus_hash: str
    prompt_version: str = PROMPT_VERSION
    config: ConfigSnapshot
    mode: Literal["live", "fake"]
    group_id: str
    max_usd: Decimal
    complete: bool
    error: str | None = None
    metrics: dict[str, object]
    cost_usd: Decimal


class AnswerReport(EvaluationReport):
    kind: Literal["answers"] = "answers"
    split: Literal["dev", "heldout"]
    cases: tuple[AnswerResult, ...]


class ProbeResult(RecordModel):
    probe: VerifierProbe
    run_id: str | None = None
    claim: CandidateClaim
    retrieved_sections: tuple[RetrievedSection, ...]
    verdict: ClaimVerdict | None = None
    calls: tuple[CallRecord, ...] = ()
    latency_ms: float | None = None
    cost_usd: Decimal = Decimal(0)
    error: RunError | None = None


class ProbePair(RecordModel):
    probe_ids: tuple[str, ...]
    complete: bool
    all_verdicts_correct: bool


class ProbeReport(EvaluationReport):
    kind: Literal["probes"] = "probes"
    cases: tuple[ProbeResult, ...]
    pairs: dict[str, ProbePair]
    gate: Literal["PASS", "FAIL"]


def write_json(payload: dict, destination: Path) -> Path:
    """Atomically publish complete JSON; never overwrite an existing artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=".eval-", suffix=".tmp"
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, destination)
    return destination


def _save_evaluation(report: EvaluationReport, settings: Settings) -> Path:
    timestamp = report.created_at.strftime("%Y%m%dT%H%M%S%fZ")
    path = settings.runtime_dir / "evals" / f"{report.kind}-{timestamp}-{uuid4().hex[:8]}.json"
    return write_json(report.model_dump(mode="json"), path)


def _ledger(settings: Settings) -> BudgetLedger:
    return BudgetLedger(
        settings.spend_db_path,
        settings.budget_cap_usd,
        settings.price_table_version,
        price_table=settings.price_table,
    )


def _validate_cap(max_usd: Decimal) -> None:
    if not max_usd.is_finite() or max_usd <= 0:
        raise ValueError("max_usd must be finite and positive")


def estimate_answers(cases: list[EvalCase], settings: Settings) -> Decimal:
    corpus = load_corpus(settings.data_dir)
    return sum(
        (pipeline.estimate_answer(case.question, corpus, settings) for case in cases), Decimal(0)
    )


def answer_metrics(results: list[AnswerResult]) -> dict[str, object]:
    """Micro citation overlap uses distinct IDs per answered case, including unanswerables."""
    answered = [r for r in results if r.status == "answered"]
    cited = [{sid for c in r.claims for sid in c.citations} for r in answered]
    expected = [set(r.supporting_sentence_ids) for r in answered]
    overlap = sum(len(a & b) for a, b in zip(cited, expected, strict=True))
    cited_count, expected_count = sum(map(len, cited)), sum(map(len, expected))
    times = [r.latency_ms for r in results if r.latency_ms is not None]
    refusals = [r.refusal_code for r in results if r.status == "refused"]
    calls = [call for r in results for call in r.calls]
    return {
        "status_accuracy": sum(r.status == r.expected_status for r in results) / len(results),
        "answered_unanswerables": sum(r.expected_status == "refused" for r in answered),
        "refusal_code_distribution": {code: refusals.count(code) for code in sorted(set(refusals))},
        "citation_precision": overlap / cited_count if cited_count else None,
        "citation_recall": overlap / expected_count if expected_count else None,
        "citation_overlap_ids": overlap,
        "cited_ids": cited_count,
        "supporting_ids": expected_count,
        "latency_p50_ms": median(times) if times else None,
        "latency_max_ms": max(times) if times else None,
        "input_tokens": sum(c.input_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
        "total_tokens": sum(c.input_tokens + c.output_tokens for c in calls),
        "cost_usd": str(sum((r.cost_usd for r in results), Decimal(0))),
    }


def _answer_result(
    case: EvalCase, record: RunRecord | None, error: str | None = None
) -> AnswerResult:
    fields = dict(case)
    if record is not None:
        fields.update(
            {
                name: getattr(record, name)
                for name in (
                    "run_id",
                    "status",
                    "refusal_code",
                    "answer_language",
                    "claims",
                    "sources",
                    "candidate_claims",
                    "check_results",
                    "verdicts",
                    "retrieved_sections",
                    "calls",
                    "latency_ms",
                    "error",
                )
            }
        )
        fields["cost_usd"] = record.metrics.cost_usd
    if error:
        fields.update(
            status="error",
            refusal_code=None,
            answer_language=None,
            claims=(),
            sources=(),
            error=record.error
            if record and record.error
            else RunError(code=error, message="Evaluation case failed."),
        )
    return AnswerResult(**fields)


def evaluate_answers(
    cases: list[EvalCase],
    settings: Settings,
    max_usd: Decimal,
    *,
    split: Literal["dev", "heldout"] = "dev",
    context: pipeline.AnswerContext | None = None,
) -> tuple[AnswerReport, Path]:
    """Run sequentially in one capped group; errors continue, budget exhaustion stops."""
    _validate_cap(max_usd)
    if (
        not cases
        or any(c.split != split for c in cases)
        or len({c.id for c in cases}) != len(cases)
    ):
        raise ValueError("Expected unique, nonempty cases from the selected split")
    for case in cases:
        pipeline.validate_question(case.question)
    group_id = f"answers-{uuid4().hex}"
    ledger = context.ledger if context else _ledger(settings)
    ledger.open_group(group_id, max_usd)
    owns_context = context is None
    if context is None:
        context = pipeline.create_live_context(settings, group_id=group_id)
    else:
        if context.settings is not settings:
            raise ValueError("Evaluation settings must match the context")
        embeddings = copy(context.embeddings)
        embeddings.group_id = group_id
        embeddings.usage = []
        store = copy(context.store)
        store.embedding = embeddings
        context = replace(context, embeddings=embeddings, store=store, group_id=group_id)
    with closing(context) if owns_context else nullcontext(context):
        created_at, git = datetime.now(UTC), get_git_info()
        results = [_answer_result(c, None) for c in cases]
        run_error = None
        try:
            for i, case in enumerate(cases):
                try:
                    record = pipeline.answer(
                        case.question, context=context, origin="eval", eval_case_id=case.id
                    )
                    results[i] = _answer_result(case, record)
                except BudgetExceeded as exc:
                    results[i] = _answer_result(
                        case, getattr(exc, "record", None), "budget_exceeded"
                    )
                    run_error = "budget_exceeded"
                    break
                except pipeline.PipelineError as exc:
                    results[i] = _answer_result(case, exc.record, exc.code)
        except BaseException:
            run_error = "interrupted"
            raise
        finally:
            report = AnswerReport(
                created_at=created_at,
                git=git,
                corpus_hash=context.corpus.corpus_hash,
                config=ConfigSnapshot.from_settings(settings, top_k=settings.top_k),
                mode=context.mode,
                group_id=group_id,
                max_usd=max_usd,
                split=split,
                cases=tuple(results),
                complete=run_error is None,
                error=run_error,
                metrics=answer_metrics(results),
                cost_usd=ledger.group_spent_usd(group_id),
            )
            path = _save_evaluation(report, settings)
        return report, path


PROBE_QUESTION = "Verify the candidate claim using only its cited evidence."


def _probe_input(
    probe: VerifierProbe, corpus: Corpus
) -> tuple[CandidateClaim, tuple[RetrievedSection, ...]]:
    for sid in probe.cited_sentence_ids:
        corpus.sentence(sid)  # Fail offline, before any provider call.
    return CandidateClaim(
        text=probe.claim, citations=probe.cited_sentence_ids
    ), pipeline.evidence_sections(
        corpus, [sid.rsplit(".", 1)[0] for sid in probe.cited_sentence_ids]
    )


def estimate_probes(probes: list[VerifierProbe], settings: Settings) -> Decimal:
    corpus = load_corpus(settings.data_dir)
    return sum(
        (
            pipeline.reservation_cost(
                settings,
                settings.verification_model,
                verification_messages(PROBE_QUESTION, (claim,), sections),
                pipeline.VerificationOutput,
            )
            for probe in probes
            for claim, sections in [_probe_input(probe, corpus)]
        ),
        Decimal(0),
    )


def evaluate_probes(
    probes: list[VerifierProbe],
    settings: Settings,
    max_usd: Decimal,
    *,
    context: pipeline.VerificationContext | None = None,
) -> tuple[ProbeReport, Path]:
    _validate_cap(max_usd)
    if not probes or len({p.id for p in probes}) != len(probes):
        raise ValueError("Expected unique, nonempty probes")
    corpus = load_corpus(settings.data_dir)
    inputs = [_probe_input(p, corpus) for p in probes]
    group_id = f"probes-{uuid4().hex}"
    ledger = context.ledger if context else _ledger(settings)
    ledger.open_group(group_id, max_usd)
    if context is not None and context.settings is not settings:
        raise ValueError("Evaluation settings must match the context")
    owns_context = context is None
    context = (
        replace(context, group_id=group_id)
        if context
        else pipeline.create_verification_context(settings, ledger, group_id)
    )
    with closing(context) if owns_context else nullcontext(context):
        created_at, git = datetime.now(UTC), get_git_info()
        results = [
            ProbeResult(probe=p, claim=claim, retrieved_sections=sections)
            for p, (claim, sections) in zip(probes, inputs, strict=True)
        ]
        error = None
        try:
            for i, result in enumerate(results):
                local = replace(context, recorder=pipeline.Recorder())
                started = perf_counter()
                verdict = problem = None
                try:
                    verdict = pipeline.verify_claims(
                        PROBE_QUESTION, (result.claim,), result.retrieved_sections, context=local
                    )[0]
                except BudgetExceeded:
                    error = "budget_exceeded"
                    problem = RunError(
                        code=error, message="The request cannot fit the remaining budget."
                    )
                except pipeline.PipelineError as exc:
                    problem = RunError(code=exc.code, message=exc.message)
                results[i] = result.model_copy(
                    update={
                        "run_id": uuid4().hex,
                        "verdict": verdict,
                        "error": problem,
                        "calls": tuple(local.recorder.calls),
                        "latency_ms": (perf_counter() - started) * 1000,
                        "cost_usd": sum((c.cost_usd for c in local.recorder.calls), Decimal(0)),
                    }
                )
                if error:
                    break
        except BaseException:
            error = "interrupted"
            raise
        finally:
            judged = [r for r in results if r.verdict is not None]
            false_accepts = sum(
                r.probe.expected_verdict != "supported" and r.verdict.verdict == "supported"
                for r in judged
            )
            false_rejects = sum(
                r.probe.expected_verdict == "supported" and r.verdict.verdict != "supported"
                for r in judged
            )
            calls = [c for r in results for c in r.calls]
            times = [r.latency_ms for r in results if r.latency_ms is not None]
            metrics = {
                "false_accepts": false_accepts,
                "false_rejects": false_rejects,
                "exact_verdict_accuracy": sum(
                    r.probe.expected_verdict == r.verdict.verdict for r in judged
                )
                / len(results),
                "latency_p50_ms": median(times) if times else None,
                "latency_max_ms": max(times) if times else None,
                "input_tokens": sum(c.input_tokens for c in calls),
                "output_tokens": sum(c.output_tokens for c in calls),
                "total_tokens": sum(c.input_tokens + c.output_tokens for c in calls),
                "cost_usd": str(sum((r.cost_usd for r in results), Decimal(0))),
            }
            pairs = {}
            for pair_id in sorted({p.pair_id for p in probes}):
                members = [r for r in results if r.probe.pair_id == pair_id]
                pairs[pair_id] = ProbePair(
                    probe_ids=tuple(r.probe.id for r in members),
                    complete=all(r.verdict is not None for r in members),
                    all_verdicts_correct=all(
                        r.verdict is not None and r.verdict.verdict == r.probe.expected_verdict
                        for r in members
                    ),
                )
            report = ProbeReport(
                created_at=created_at,
                git=git,
                corpus_hash=corpus.corpus_hash,
                config=ConfigSnapshot.from_settings(settings, top_k=settings.top_k),
                mode=context.mode,
                group_id=group_id,
                max_usd=max_usd,
                complete=error is None,
                error=error,
                cases=tuple(results),
                pairs=pairs,
                metrics=metrics,
                cost_usd=ledger.group_spent_usd(group_id),
                gate="PASS"
                if not error and len(judged) == len(results) and false_accepts == 0
                else "FAIL",
            )
            path = _save_evaluation(report, settings)
        return report, path
