"""Immutable run audits and the public API contract."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from citarilo.config import ReasoningEffort, Settings, make_private_dir

SCHEMA_VERSION = 1
MAX_FEEDBACK_NOTE = 1000
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")

type Mode = Literal["live", "fake", "replay"]
type Status = Literal["answered", "refused", "error"]
type RefusalCode = Literal["insufficient_evidence", "check_failed", "unsupported", "contradicted"]
type RefusalStep = Literal[
    "no_evidence", "deterministic_check", "verifier_unsupported", "verifier_contradicted"
]
type CheckName = Literal[
    "invalid_citation_count",
    "unknown_citation",
    "literal_not_in_citations",
    "language_mismatch",
]
type FeedbackLabel = Literal["wrongly_refused", "wrong_or_dangerous", "missing_step", "other"]
type Verdict = Literal["supported", "unsupported", "contradicted"]
type Origin = Literal["api", "eval"]
type AnswerLanguage = Literal["en", "cs"]
type Text = Annotated[str, StringConstraints(min_length=1)]
type TokenCount = Annotated[int, Field(ge=0, strict=True)]
type Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class RecordModel(BaseModel):
    """Reject unknown fields and freeze nested models and tuple collections."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class GitInfo(RecordModel):
    """Unknown Git metadata is represented by null, never a guessed revision."""

    commit: Text | None = None
    dirty: bool | None = None


def get_git_info(repo_dir: str | Path | None = None) -> GitInfo:
    """Read local Git state, tolerating missing Git and non-repository paths."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return GitInfo()
    return GitInfo(commit=commit or None, dirty=bool(status.strip()))


class ConfigSnapshot(RecordModel):
    """Explicit settings allowlist; credentials and filesystem paths are excluded."""

    generation_model: Text
    verification_model: Text
    embedding_model: Text
    generation_reasoning_effort: ReasoningEffort
    verification_reasoning_effort: ReasoningEffort
    top_k: int = Field(gt=0, strict=True)
    call_timeout_seconds: float = Field(gt=0)
    max_output_tokens: int = Field(gt=0, strict=True)
    budget_cap_usd: Money
    price_table_version: Text

    @classmethod
    def from_settings(cls, settings: Settings, *, top_k: int) -> Self:
        """Copy only named safe fields; top_k is not yet available on Settings."""
        return cls(
            generation_model=settings.generation_model,
            verification_model=settings.verification_model,
            embedding_model=settings.embedding_model,
            generation_reasoning_effort=settings.generation_reasoning_effort,
            verification_reasoning_effort=settings.verification_reasoning_effort,
            top_k=top_k,
            call_timeout_seconds=settings.call_timeout_seconds,
            max_output_tokens=settings.max_output_tokens,
            budget_cap_usd=settings.budget_cap_usd,
            price_table_version=settings.price_table_version,
        )


class CandidateClaim(RecordModel):
    """Retain generated citations even when their count fails later checks."""

    text: Text
    citations: tuple[Text, ...]


class Claim(CandidateClaim):
    """A final claim cites one to three distinct sentences."""

    citations: tuple[Text, ...] = Field(min_length=1, max_length=3)

    @field_validator("citations")
    @classmethod
    def unique_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("claim citations must be unique")
        return value


class Source(RecordModel):
    """One public, citable sentence with its original document context."""

    sentence_id: Text
    text: Text
    doc_id: Text
    doc_title: Text
    heading: Text


class RetrievedSection(RecordModel):
    """Retrieved section content and its similarity score at the time of the run."""

    section_key: Text
    doc_id: Text
    doc_title: Text
    heading: Text
    score: float
    sentences: tuple[Source, ...]


class CheckResult(RecordModel):
    """Deterministic check outcome for a zero-based candidate claim index."""

    claim_index: TokenCount
    ok: bool
    reasons: tuple[Text, ...] = ()


class ClaimVerdict(RecordModel):
    """Verifier decision and optional explanation, retained only in the audit."""

    claim_index: TokenCount
    verdict: Verdict
    reason: str = ""


class CallRecord(RecordModel):
    """Usage and ledger reservation for one model or embedding call."""

    node: Text
    model: Text
    input_tokens: TokenCount
    output_tokens: TokenCount
    cost_usd: Money
    reservation_id: Text | None
    duration_ms: Milliseconds


class NodeTiming(RecordModel):
    """Elapsed wall time and completion outcome for a graph node."""

    node: Text
    duration_ms: Milliseconds
    ok: bool


class Metrics(RecordModel):
    """Public usage totals; Decimal costs serialize as JSON strings."""

    latency_ms: Milliseconds
    input_tokens: TokenCount
    output_tokens: TokenCount
    cost_usd: Money
    node_timings: tuple[NodeTiming, ...]


def _validate_final(
    status: Status,
    refusal_code: RefusalCode | None,
    claims: tuple[Claim, ...],
    sources: tuple[Source, ...],
    answer_language: AnswerLanguage | None,
) -> None:
    if (status == "refused") != (refusal_code is not None):
        raise ValueError("refused status requires a refusal_code, and only refused may have one")
    if status == "answered" and not claims:
        raise ValueError("answered status requires at least one claim")
    if status != "answered" and claims:
        raise ValueError("only answered status may expose claims")
    if (status == "answered") != (answer_language is not None):
        raise ValueError("answered status requires answer_language, and only answered may have one")
    cited = {citation for claim in claims for citation in claim.citations}
    source_ids = [source.sentence_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("sources must have unique sentence IDs")
    if set(source_ids) != cited:
        raise ValueError("sources must cover exactly the cited sentences")


_REFUSAL_STEP: dict[RefusalCode, RefusalStep] = {
    "insufficient_evidence": "no_evidence",
    "check_failed": "deterministic_check",
    "unsupported": "verifier_unsupported",
    "contradicted": "verifier_contradicted",
}
_CHECK_NAMES: dict[str, CheckName] = {
    "invalid_citation_count": "invalid_citation_count",
    "unknown_citation": "unknown_citation",
    "literal_not_in_citations": "literal_not_in_citations",
    "language_mismatch": "language_mismatch",
}


class RefusalSection(RecordModel):
    """A retrieved section the reader may open. No sentence or claim text."""

    doc_id: Text
    doc_title: Text
    heading: Text
    section_key: Text


class RefusalExplanation(RecordModel):
    """Why a draft was refused, without unverified claim text or verifier prose."""

    step: RefusalStep
    rejected_claims: int = Field(ge=0, strict=True)
    failed_check: CheckName | None = None
    sections: tuple[RefusalSection, ...] = ()


class AskResponse(RecordModel):
    """HTTP 200 body, with no internal audit fields."""

    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]
    mode: Mode
    status: Literal["answered", "refused"]
    answer_language: AnswerLanguage | None = None
    refusal_code: RefusalCode | None = None
    refusal: RefusalExplanation | None = None
    claims: tuple[Claim, ...] = ()
    sources: tuple[Source, ...] = ()
    metrics: Metrics

    @model_validator(mode="after")
    def validate_final(self) -> Self:
        _validate_final(
            self.status, self.refusal_code, self.claims, self.sources, self.answer_language
        )
        expected = None if self.refusal_code is None else _REFUSAL_STEP[self.refusal_code]
        if (self.refusal is None) != (expected is None):
            raise ValueError(
                "refused status requires a refusal explanation, and only refused may have one"
            )
        if self.refusal is not None and self.refusal.step != expected:
            raise ValueError("refusal explanation does not match the refusal code")
        if (
            self.refusal is not None
            and self.refusal_code != "check_failed"
            and self.refusal.failed_check is not None
        ):
            raise ValueError("only a failed deterministic check names a check")
        return self


class RunError(RecordModel):
    """A safe error code and message; callers must omit provider secrets."""

    code: Text
    message: Text


class RunRecord(RecordModel):
    """Complete local audit; technical failures are persisted before HTTP 502."""

    schema_version: Literal[1] = SCHEMA_VERSION
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")] = Field(
        default_factory=lambda: uuid4().hex
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mode: Mode
    origin: Origin
    eval_case_id: Text | None = None
    git: GitInfo
    corpus_hash: Text
    document_set: Text | None = None
    prompt_version: Text
    config: ConfigSnapshot
    question: Text
    retrieved_sections: tuple[RetrievedSection, ...] = ()
    candidate_claims: tuple[CandidateClaim, ...] = ()
    check_results: tuple[CheckResult, ...] = ()
    verdicts: tuple[ClaimVerdict, ...] = ()
    status: Status
    answer_language: AnswerLanguage | None = None
    refusal_code: RefusalCode | None = None
    claims: tuple[Claim, ...] = ()
    sources: tuple[Source, ...] = ()
    calls: tuple[CallRecord, ...] = ()
    node_timings: tuple[NodeTiming, ...] = ()
    error: RunError | None = None
    latency_ms: Milliseconds

    @field_validator("created_at")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        _validate_final(
            self.status, self.refusal_code, self.claims, self.sources, self.answer_language
        )
        if (self.origin == "eval") != (self.eval_case_id is not None):
            raise ValueError("eval origin requires an eval_case_id, and only eval may have one")
        if (self.status == "error") != (self.error is not None):
            raise ValueError("error status requires error details, and only error may have them")
        if self.status == "answered":
            expected = list(range(len(self.candidate_claims)))
            if not expected or [c.model_dump() for c in self.candidate_claims] != [
                c.model_dump() for c in self.claims
            ]:
                raise ValueError("answered claims must match all candidate claims")
            if sorted(check.claim_index for check in self.check_results) != expected or not all(
                check.ok for check in self.check_results
            ):
                raise ValueError("answered claims require exactly one passing check each")
            if sorted(verdict.claim_index for verdict in self.verdicts) != expected or not all(
                verdict.verdict == "supported" for verdict in self.verdicts
            ):
                raise ValueError("answered claims require exactly one supported verdict each")
        return self

    @property
    def metrics(self) -> Metrics:
        """Derive totals from calls so stored and public usage cannot diverge."""
        # Sum integer coefficients to avoid rounding under a caller's Decimal context.
        exponent = min((call.cost_usd.as_tuple().exponent for call in self.calls), default=0)
        cost_units = 0
        for call in self.calls:
            parts = call.cost_usd.as_tuple()
            coefficient = int("".join(map(str, parts.digits)))
            cost_units += coefficient * 10 ** (parts.exponent - exponent)
        return Metrics(
            latency_ms=self.latency_ms,
            input_tokens=sum(call.input_tokens for call in self.calls),
            output_tokens=sum(call.output_tokens for call in self.calls),
            cost_usd=Decimal(f"{cost_units}e{exponent}"),
            node_timings=self.node_timings,
        )

    def to_response(self) -> AskResponse:
        """Project only reviewed public fields; errors belong to HTTP 502."""
        if self.status == "error":
            raise ValueError("error run records cannot become an HTTP 200 response")
        return AskResponse(
            run_id=self.run_id,
            mode=self.mode,
            status=self.status,
            answer_language=self.answer_language,
            refusal_code=self.refusal_code,
            refusal=refusal_explanation(self),
            claims=self.claims,
            sources=self.sources,
            metrics=self.metrics,
        )


def refusal_explanation(record: RunRecord) -> RefusalExplanation | None:
    """Project a refused audit into the public explanation.

    Check names are a fixed code list. Verdict reasons and claim text stay in the audit.
    """
    if record.status != "refused" or record.refusal_code is None:
        return None
    sections = tuple(
        RefusalSection(
            doc_id=section.doc_id,
            doc_title=section.doc_title,
            heading=section.heading,
            section_key=section.section_key,
        )
        for section in record.retrieved_sections
    )
    failed_check: CheckName | None = None
    if record.refusal_code == "check_failed":
        rejected = sum(1 for check in record.check_results if not check.ok)
        for check in record.check_results:
            if check.ok:
                continue
            for reason in check.reasons:
                failed_check = _CHECK_NAMES.get(reason)
                if failed_check is not None:
                    break
            if failed_check is not None:
                break
    elif record.refusal_code == "insufficient_evidence":
        rejected = 0
    else:
        rejected = sum(1 for verdict in record.verdicts if verdict.verdict != "supported")
    return RefusalExplanation(
        step=_REFUSAL_STEP[record.refusal_code],
        rejected_claims=rejected,
        failed_check=failed_check,
        sections=sections,
    )


class FeedbackEntry(RecordModel):
    """One append-only note. It never changes the run record it refers to."""

    created_at: datetime
    label: FeedbackLabel | None = None
    note: str = ""

    @field_validator("created_at")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("note")
    @classmethod
    def bounded_note(cls, value: str) -> str:
        if len(value) > MAX_FEEDBACK_NOTE:
            raise ValueError("Note is too long")
        return value

    @model_validator(mode="after")
    def not_empty(self) -> Self:
        if self.label is None and not self.note.strip():
            raise ValueError("A note needs text or a label")
        return self


def feedback_path(feedback_dir: str | Path, run_id: str) -> Path:
    """Return the note file for a run id, rejecting anything that is not a bare id."""
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Invalid run id")
    directory = Path(feedback_dir)
    path = directory / f"{run_id}.jsonl"
    if path.parent.resolve() != directory.resolve():
        raise ValueError("Invalid run id")
    return path


def append_feedback(
    feedback_dir: str | Path,
    run_id: str,
    *,
    label: FeedbackLabel | None,
    note: str,
) -> FeedbackEntry:
    """Append one note. The file is private to its owner and is never rewritten."""
    entry = FeedbackEntry(created_at=datetime.now(UTC), label=label, note=note)
    directory = make_private_dir(Path(feedback_dir))
    path = feedback_path(directory, run_id)
    payload = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        os.chmod(path, 0o600)
        os.write(fd, (payload + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return entry


def read_feedback(feedback_dir: str | Path, run_id: str) -> tuple[FeedbackEntry, ...]:
    """Read valid notes for one run. A missing file is an empty tuple."""
    try:
        path = feedback_path(feedback_dir, run_id)
    except ValueError:
        return ()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    entries: list[FeedbackEntry] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            entries.append(FeedbackEntry.model_validate_json(line))
        except (ValidationError, ValueError):
            continue
    return tuple(entries)


def write_record(record: RunRecord, runs_dir: str | Path) -> Path:
    """Publish UTF-8 JSON atomically; an existing run ID is never overwritten."""
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{record.run_id}.json"
    payload = json.dumps(
        record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=f".{record.run_id}.", suffix=".tmp"
    ) as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        # A hard link publishes the complete file atomically and fails if the ID exists.
        os.link(stream.name, destination)
    return destination


def read_record(path: str | Path) -> RunRecord:
    """Read and validate an audit record, including its schema version."""
    return RunRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))


def iter_records(
    runs_dir: str | Path,
    *,
    on_invalid: Callable[[Path], None] | None = None,
) -> Iterator[RunRecord]:
    """Yield valid ``*.json`` audits in ``runs_dir``.

    A missing directory yields nothing. Unreadable or invalid files are skipped,
    and ``on_invalid`` is called with their path when given.
    """
    directory = Path(runs_dir)
    try:
        paths = [path for path in directory.iterdir() if path.is_file() and path.suffix == ".json"]
    except FileNotFoundError:
        return
    except OSError:
        if on_invalid is not None:
            on_invalid(directory)
        return
    for path in sorted(paths):
        try:
            yield read_record(path)
        except (OSError, ValidationError, ValueError):
            if on_invalid is not None:
                on_invalid(path)
