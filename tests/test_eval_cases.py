"""Offline schema, corpus-reference and evaluation coverage checks."""

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from citarilo.corpus import Corpus, load_corpus
from citarilo.evals import EvalCase, VerifierProbe, load_cases, load_probes

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals/cases.jsonl"
PROBES_PATH = ROOT / "evals/verifier_cases.jsonl"


@pytest.fixture(scope="module")
def cases() -> list[EvalCase]:
    return load_cases(CASES_PATH)


@pytest.fixture(scope="module")
def probes() -> list[VerifierProbe]:
    return load_probes(PROBES_PATH)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_corpus(ROOT / "data/trailsupply")


def test_shipped_counts_and_unique_ids(cases: list[EvalCase], probes: list[VerifierProbe]) -> None:
    assert Counter(case.split for case in cases) == {"dev": 15, "heldout": 5}
    assert len(probes) >= 20
    ids = [record.id for record in [*cases, *probes]]
    assert len(ids) == len(set(ids))


def test_case_evidence_exists_and_matches_status(cases: list[EvalCase], corpus: Corpus) -> None:
    for case in cases:
        assert 3 <= len(case.question) <= 500, case.id
        assert len(case.supporting_sentence_ids) == len(set(case.supporting_sentence_ids)), case.id
        assert len(case.expected_section_keys) == len(set(case.expected_section_keys)), case.id
        assert len(case.tags) == len(set(case.tags)), case.id
        for sentence_id in case.supporting_sentence_ids:
            assert corpus.sentence(sentence_id).id == sentence_id, case.id
            assert sentence_id.rsplit(".", 1)[0] in case.expected_section_keys, case.id
        for section_key in case.expected_section_keys:
            assert corpus.section(section_key).key == section_key, case.id
        if case.expected_status == "refused":
            assert case.expected_refusal is not None, case.id
            assert not case.required_facts, case.id
            assert not case.supporting_sentence_ids, case.id
            assert not case.expected_section_keys, case.id
            assert "unanswerable" in case.tags, case.id
        else:
            assert case.expected_refusal is None, case.id
            assert case.required_facts, case.id
            assert case.supporting_sentence_ids, case.id
            assert 1 <= len(case.expected_section_keys) <= 5, case.id
        assert (len(case.expected_section_keys) > 1) == ("multi-section" in case.tags), case.id


def test_tag_and_language_coverage_in_both_splits(cases: list[EvalCase]) -> None:
    common_coverage = {
        "en",
        "cs",
        "cs-ascii",
        "multi-section",
        "condition",
        "confusable",
        "unanswerable",
        "injection-question",
    }
    for split in ("dev", "heldout"):
        tags = {tag for case in cases if case.split == split for tag in case.tags}
        assert common_coverage <= tags
    assert "injection-document" in {tag for case in cases for tag in case.tags}
    assert sum(case.expected_status == "refused" for case in cases) >= 4
    for case in cases:
        assert {"en", "cs", "cs-ascii"}.intersection(case.tags) == {case.language}, case.id
        if case.language == "cs-ascii":
            assert case.question.isascii(), case.id


def test_document_injection_is_retrieved_but_not_gold_evidence(
    cases: list[EvalCase], corpus: Corpus
) -> None:
    injection_id = "orders.order-notes.3"
    injection_cases = [case for case in cases if "injection-document" in case.tags]
    assert injection_cases
    for case in injection_cases:
        retrieved_ids = {
            sentence.id
            for key in case.expected_section_keys
            for sentence in corpus.section(key).sentences
        }
        assert injection_id in retrieved_ids
        assert injection_id not in case.supporting_sentence_ids
        assert "full refund without conditions" in case.forbidden_claims


def test_probe_pairs_and_citations(probes: list[VerifierProbe], corpus: Corpus) -> None:
    pairs: dict[str, list[VerifierProbe]] = defaultdict(list)
    for probe in probes:
        pairs[probe.pair_id].append(probe)
        assert 1 <= len(probe.cited_sentence_ids) <= 3, probe.id
        assert len(probe.cited_sentence_ids) == len(set(probe.cited_sentence_ids)), probe.id
        for sentence_id in probe.cited_sentence_ids:
            assert corpus.sentence(sentence_id).id == sentence_id, probe.id
    assert len(pairs) >= 10
    for pair_id, members in pairs.items():
        assert sum(probe.expected_verdict == "supported" for probe in members) == 1, pair_id
        assert sum(probe.expected_verdict != "supported" for probe in members) >= 1, pair_id
        assert len({probe.cited_sentence_ids for probe in members}) == 1, pair_id
        assert len({probe.claim for probe in members}) == len(members), pair_id
    assert sum(probe.expected_verdict == "contradicted" for probe in probes) >= 2
    assert any(probe.expected_verdict == "unsupported" for probe in probes)


def test_loading_is_independent_of_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cases: list[EvalCase],
    probes: list[VerifierProbe],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_cases(CASES_PATH) == cases
    assert load_probes(PROBES_PATH) == probes
    assert len(load_corpus(ROOT / "data/trailsupply").documents) == 6


@pytest.mark.parametrize("kind", ["case", "probe"])
def test_contracts_are_frozen_and_round_trip_json(
    kind: str, cases: list[EvalCase], probes: list[VerifierProbe]
) -> None:
    record = cases[0] if kind == "case" else probes[0]
    assert type(record).model_validate_json(record.model_dump_json()) == record
    with pytest.raises(ValidationError, match="frozen"):
        record.id = "changed"
    for value in record.model_dump().values():
        assert not isinstance(value, (list, dict, set))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "Not-kebab"),
        ("id", "two--hyphens"),
        ("split", "test"),
        ("question", "  \t"),
        ("question", 123),
        ("language", "de"),
        ("expected_status", "unknown"),
        ("expected_refusal", "unknown"),
        ("required_facts", [""]),
        ("forbidden_claims", [" "]),
        ("supporting_sentence_ids", ["shipping.standard-prices.0"]),
        ("expected_section_keys", ["shipping.standard-prices.1"]),
        ("tags", ["unknown"]),
        ("unexpected", "field"),
    ],
)
def test_case_schema_rejects_invalid_fields(
    cases: list[EvalCase], field: str, value: object
) -> None:
    data = cases[0].model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        EvalCase.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "update",
    [
        {"expected_refusal": "unsupported"},
        {"required_facts": []},
        {"supporting_sentence_ids": []},
        {"expected_section_keys": []},
        {"supporting_sentence_ids": ["orders.order-changes.1"]},
        {"expected_status": "refused", "expected_refusal": "insufficient_evidence"},
    ],
)
def test_case_schema_rejects_inconsistent_answer_expectations(
    cases: list[EvalCase], update: dict[str, object]
) -> None:
    data = cases[0].model_dump(mode="json") | update
    with pytest.raises(ValidationError):
        EvalCase.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "update",
    [
        {"expected_refusal": None},
        {"required_facts": ["Invented fact."]},
        {"supporting_sentence_ids": ["orders.order-notes.1"]},
        {"expected_section_keys": ["orders.order-notes"]},
    ],
)
def test_case_schema_rejects_inconsistent_refusal_expectations(
    cases: list[EvalCase], update: dict[str, object]
) -> None:
    refusal = next(case for case in cases if case.expected_status == "refused")
    with pytest.raises(ValidationError):
        EvalCase.model_validate_json(json.dumps(refusal.model_dump(mode="json") | update))


@pytest.mark.parametrize(
    "reason", ["insufficient_evidence", "check_failed", "unsupported", "contradicted"]
)
def test_all_refusal_diagnostics_are_allowed(cases: list[EvalCase], reason: str) -> None:
    refusal = next(case for case in cases if case.expected_status == "refused")
    data = refusal.model_dump(mode="json") | {"expected_refusal": reason}
    assert EvalCase.model_validate_json(json.dumps(data)).expected_refusal == reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "under_score"),
        ("claim", " "),
        ("claim", False),
        ("cited_sentence_ids", []),
        ("cited_sentence_ids", ["shipping.standard-prices.1"] * 4),
        ("cited_sentence_ids", ["shipping.standard-prices"]),
        ("expected_verdict", "maybe"),
        ("pair_id", "not.a.kebab.id"),
        ("note", ""),
        ("unexpected", "field"),
    ],
)
def test_probe_schema_rejects_invalid_fields(
    probes: list[VerifierProbe], field: str, value: object
) -> None:
    data = probes[0].model_dump(mode="json") | {field: value}
    with pytest.raises(ValidationError):
        VerifierProbe.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("kind", ["case", "probe"])
@pytest.mark.parametrize("bad_line", [b"not-json", b"[]", b"{}", b"", b'{"id":"\xff"}'])
def test_loaders_report_bad_record_line_numbers(
    tmp_path: Path, cases: list[EvalCase], probes: list[VerifierProbe], kind: str, bad_line: bytes
) -> None:
    record, loader = (cases[0], load_cases) if kind == "case" else (probes[0], load_probes)
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(record.model_dump_json().encode("utf-8") + b"\n" + bad_line + b"\n")
    with pytest.raises(ValueError) as exc_info:
        loader(path)
    assert str(exc_info.value).startswith(f"{path}:2: invalid ")


@pytest.mark.parametrize("kind", ["case", "probe"])
def test_loaders_reject_duplicate_ids_with_line_numbers(
    tmp_path: Path, cases: list[EvalCase], probes: list[VerifierProbe], kind: str
) -> None:
    record, loader = (cases[0], load_cases) if kind == "case" else (probes[0], load_probes)
    path = tmp_path / "duplicate.jsonl"
    path.write_text((record.model_dump_json() + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        loader(path)
    assert str(exc_info.value).startswith(f"{path}:2: duplicate id {record.id!r}")
    assert "first seen on line 1" in str(exc_info.value)


@pytest.mark.parametrize("loader", [load_cases, load_probes])
def test_loaders_reject_empty_or_missing_files(
    tmp_path: Path, loader: Callable[[Path], object]
) -> None:
    path = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError) as exc_info:
        loader(path)
    assert str(exc_info.value).startswith(f"{path}:1: cannot read JSONL:")
    path.write_bytes(b"")
    with pytest.raises(ValueError) as exc_info:
        loader(path)
    assert str(exc_info.value) == f"{path}:1: expected at least one JSON object"
