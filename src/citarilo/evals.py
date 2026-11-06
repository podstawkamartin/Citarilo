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
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
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

from citarilo.budget import BudgetExceeded, BudgetLedger, max_cost
from citarilo.config import Settings
from citarilo.corpus import load_corpus
from citarilo.index import (
    EmbeddingsClient,
    EmbeddingUsage,
    UsageTrackingEmbeddings,
    load_index,
    search,
)
from citarilo.records import GitInfo, get_git_info

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
