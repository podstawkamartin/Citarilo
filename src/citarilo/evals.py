"""Offline evaluation contracts and JSONL loaders; no live runner.

All required facts and verifier claims are English, regardless of question language.
``expected_refusal`` records the anticipated diagnostic reason, not an exact-match
grading requirement: an unanswerable case passes the refusal-status check with ANY
of the four allowed refusal reasons. It must never pass with an answered status.
Heldout cases are reserved for one final evaluation, not iterative tuning.
"""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
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
