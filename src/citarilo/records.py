"""Immutable run audits and the public API/replay contract."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
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
    field_validator,
    model_validator,
)

from citarilo.config import ReasoningEffort, Settings

SCHEMA_VERSION = 1

type Mode = Literal["live", "fake", "replay"]
type Status = Literal["answered", "refused", "error"]
type RefusalCode = Literal["insufficient_evidence", "check_failed", "unsupported", "contradicted"]
type Verdict = Literal["supported", "unsupported", "contradicted"]
type Origin = Literal["api", "eval"]
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
) -> None:
    if (status == "refused") != (refusal_code is not None):
        raise ValueError("refused status requires a refusal_code, and only refused may have one")
    if status == "answered" and not claims:
        raise ValueError("answered status requires at least one claim")
    if status != "answered" and claims:
        raise ValueError("only answered status may expose claims")
    cited = {citation for claim in claims for citation in claim.citations}
    source_ids = [source.sentence_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("sources must have unique sentence IDs")
    if set(source_ids) != cited:
        raise ValueError("sources must cover exactly the cited sentences")


class AskResponse(RecordModel):
    """HTTP 200 body and phase-2 replay file, with no internal audit fields."""

    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]
    mode: Mode
    status: Literal["answered", "refused"]
    refusal_code: RefusalCode | None = None
    claims: tuple[Claim, ...] = ()
    sources: tuple[Source, ...] = ()
    metrics: Metrics

    @model_validator(mode="after")
    def validate_final(self) -> Self:
        _validate_final(self.status, self.refusal_code, self.claims, self.sources)
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
    prompt_version: Text
    config: ConfigSnapshot
    question: Text
    retrieved_sections: tuple[RetrievedSection, ...] = ()
    candidate_claims: tuple[CandidateClaim, ...] = ()
    check_results: tuple[CheckResult, ...] = ()
    verdicts: tuple[ClaimVerdict, ...] = ()
    status: Status
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
        _validate_final(self.status, self.refusal_code, self.claims, self.sources)
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
            refusal_code=self.refusal_code,
            claims=self.claims,
            sources=self.sources,
            metrics=self.metrics,
        )


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
