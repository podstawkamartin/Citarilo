"""Run audit validation, public projections, and atomic persistence without network."""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from pydantic import ValidationError

from citarilo import records
from citarilo.config import Settings
from citarilo.records import (
    SCHEMA_VERSION,
    AskResponse,
    CallRecord,
    CandidateClaim,
    CheckResult,
    Claim,
    ClaimVerdict,
    ConfigSnapshot,
    GitInfo,
    Metrics,
    NodeTiming,
    RetrievedSection,
    RunError,
    RunRecord,
    Source,
    get_git_info,
    read_record,
    write_record,
)


@pytest.fixture
def record_data() -> dict:
    source = Source(
        sentence_id="returns.window.1",
        text="Zboží lze vrátit do 30 dnů.",
        doc_id="returns",
        doc_title="Vrácení zboží",
        heading="Lhůta pro vrácení",
    )
    claim = Claim(text="You can return goods within 30 days.", citations=(source.sentence_id,))
    return {
        "run_id": "0123456789abcdef0123456789abcdef",
        "created_at": datetime(2026, 11, 6, 12, tzinfo=UTC),
        "mode": "fake",
        "origin": "api",
        "git": GitInfo(commit="abc123", dirty=True),
        "corpus_hash": "corpus-sha256",
        "prompt_version": "1",
        "config": ConfigSnapshot.from_settings(
            Settings(_env_file=None, openai_api_key=None), top_k=5
        ),
        "question": "When can I return goods?",
        "retrieved_sections": (
            RetrievedSection(
                section_key="returns.window",
                doc_id=source.doc_id,
                doc_title=source.doc_title,
                heading=source.heading,
                score=0.9,
                sentences=(source,),
            ),
        ),
        "candidate_claims": (CandidateClaim(**claim.model_dump()),),
        "check_results": (CheckResult(claim_index=0, ok=True),),
        "verdicts": (ClaimVerdict(claim_index=0, verdict="supported", reason="Explicit deadline"),),
        "status": "answered",
        "answer_language": "en",
        "claims": (claim,),
        "sources": (source,),
        "calls": (
            CallRecord(
                node="generate",
                model="test-model",
                input_tokens=10,
                output_tokens=4,
                cost_usd=Decimal("0.000003"),
                reservation_id="reservation-1",
                duration_ms=2,
            ),
            CallRecord(
                node="verify",
                model="test-model",
                input_tokens=20,
                output_tokens=6,
                cost_usd=Decimal("0.000005"),
                reservation_id="reservation-2",
                duration_ms=3,
            ),
        ),
        "node_timings": (NodeTiming(node="generate", duration_ms=2.5, ok=True),),
        "latency_ms": 10,
    }


def test_answered_record_and_public_projection(record_data: dict) -> None:
    record = RunRecord(**record_data)
    response = record.to_response()
    assert isinstance(response, AskResponse)
    assert response.metrics == Metrics(
        latency_ms=10,
        input_tokens=30,
        output_tokens=10,
        cost_usd=Decimal("0.000008"),
        node_timings=record.node_timings,
    )
    assert response.claims == record.claims
    assert response.sources == record.sources
    assert response.refusal_code is None
    assert response.answer_language == record.answer_language == "en"
    assert record.schema_version == SCHEMA_VERSION == 1
    public = json.loads(response.model_dump_json())
    assert set(public) == {
        "run_id",
        "mode",
        "status",
        "answer_language",
        "refusal_code",
        "claims",
        "sources",
        "metrics",
    }
    assert set(public["metrics"]) == {
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "node_timings",
    }
    assert public["metrics"]["cost_usd"] == "0.000008"
    assert set(public["claims"][0]) == {"text", "citations"}
    assert set(public["sources"][0]) == {"sentence_id", "text", "doc_id", "doc_title", "heading"}
    assert "Explicit deadline" not in response.model_dump_json()
    assert "reservation-1" not in response.model_dump_json()
    assert "test-model" not in response.model_dump_json()
    assert record.config.generation_model not in response.model_dump_json()
    assert AskResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(
    "code", ["insufficient_evidence", "check_failed", "unsupported", "contradicted"]
)
def test_refused_records_keep_audit_but_expose_no_claims(record_data: dict, code: str) -> None:
    record_data.update(
        status="refused", refusal_code=code, claims=(), sources=(), answer_language=None
    )
    record = RunRecord(**record_data)
    assert record.candidate_claims
    assert record.to_response().status == "refused"
    assert record.to_response().refusal_code == code
    assert record.to_response().claims == record.to_response().sources == ()
    assert record.answer_language is record.to_response().answer_language is None


def test_error_record_preserves_partial_usage_and_failed_timing(record_data: dict) -> None:
    record_data.update(
        status="error",
        answer_language=None,
        claims=(),
        sources=(),
        error=RunError(code="provider_failure", message="Provider request failed"),
        verdicts=(),
        node_timings=(NodeTiming(node="verify", duration_ms=3, ok=False),),
    )
    record = RunRecord(**record_data)
    assert record.error.code == "provider_failure"
    assert record.metrics.input_tokens == 30
    assert not record.node_timings[0].ok
    with pytest.raises(ValueError, match="HTTP 200"):
        record.to_response()
    assert RunRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("mode", ["live", "fake", "replay"])
def test_modes_and_replay_contract(record_data: dict, mode: str) -> None:
    response = RunRecord(**(record_data | {"mode": mode})).to_response()
    assert json.loads(response.model_dump_json())["mode"] == mode
    assert AskResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"status": "refused", "claims": (), "sources": ()}, "refusal_code"),
        ({"refusal_code": "unsupported"}, "refusal_code"),
        ({"status": "error", "refusal_code": "unsupported"}, "refusal_code"),
        ({"claims": (), "sources": ()}, "at least one claim"),
        ({"status": "refused", "refusal_code": "unsupported"}, "only answered"),
        ({"sources": ()}, "exactly the cited"),
        ({"error": {"code": "oops", "message": "Failure"}}, "only error"),
        (
            {"status": "error", "claims": (), "sources": (), "answer_language": None},
            "error details",
        ),
        ({"answer_language": None}, "requires answer_language"),
        ({"answer_language": "de"}, "answer_language"),
        ({"origin": "eval"}, "eval_case_id"),
        ({"eval_case_id": "case-1"}, "only eval"),
        ({"candidate_claims": ()}, "match all candidate"),
        ({"check_results": ()}, "passing check"),
        ({"check_results": [{"claim_index": 0, "ok": False}]}, "passing check"),
        ({"check_results": [{"claim_index": 1, "ok": True}]}, "passing check"),
        ({"verdicts": ()}, "supported verdict"),
        ({"verdicts": [{"claim_index": 0, "verdict": "unsupported"}]}, "supported verdict"),
        ({"verdicts": [{"claim_index": 0, "verdict": "contradicted"}]}, "supported verdict"),
        ({"verdicts": [{"claim_index": 1, "verdict": "supported"}]}, "supported verdict"),
        ({"created_at": datetime(2026, 11, 6)}, "timezone-aware"),
        ({"schema_version": 2}, "schema_version"),
        ({"mode": "mock"}, "mode"),
        ({"status": "unknown"}, "status"),
        ({"origin": "batch"}, "origin"),
        ({"run_id": "../escape"}, "run_id"),
        ({"latency_ms": -1}, "latency_ms"),
        ({"latency_ms": float("nan")}, "latency_ms"),
        ({"prompt": "internal prompt"}, "Extra inputs"),
        ({"metrics": {"input_tokens": 99}}, "Extra inputs"),
    ],
)
def test_invalid_run_fields(record_data: dict, changes: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        RunRecord(**(record_data | changes))


@pytest.mark.parametrize("language", ["en", "cs"])
def test_answer_language_round_trip(record_data: dict, tmp_path: Path, language: str) -> None:
    record = RunRecord(**(record_data | {"answer_language": language}))
    assert read_record(write_record(record, tmp_path)).answer_language == language
    response = AskResponse.model_validate_json(record.to_response().model_dump_json())
    assert response.answer_language == language


@pytest.mark.parametrize("status", ["refused", "error"])
def test_non_answered_records_reject_answer_language(record_data: dict, status: str) -> None:
    record_data.update(status=status, claims=(), sources=())
    if status == "refused":
        record_data["refusal_code"] = "unsupported"
    else:
        record_data["error"] = RunError(code="provider_error", message="Request failed.")
    with pytest.raises(ValidationError, match="only answered may have one"):
        RunRecord(**record_data)
    if status == "refused":
        public = RunRecord(**(record_data | {"answer_language": None})).to_response().model_dump()
        with pytest.raises(ValidationError, match="only answered may have one"):
            AskResponse(**(public | {"answer_language": "cs"}))


@pytest.mark.parametrize("field", ["sources", "check_results", "verdicts"])
def test_duplicate_final_evidence_rejected(record_data: dict, field: str) -> None:
    record_data[field] = record_data[field] * 2
    with pytest.raises(ValidationError):
        RunRecord(**record_data)


def test_unused_source_rejected(record_data: dict) -> None:
    extra = record_data["sources"][0].model_dump() | {"sentence_id": "returns.window.2"}
    record_data["sources"] += (Source(**extra),)
    with pytest.raises(ValidationError, match="exactly the cited"):
        RunRecord(**record_data)


@pytest.mark.parametrize("citations", [(), ("1", "2", "3", "4"), ("1", "1")])
def test_final_citation_rules_and_candidate_audit(citations: tuple[str, ...]) -> None:
    assert CandidateClaim(text="Generated text", citations=citations).citations == citations
    with pytest.raises(ValidationError):
        Claim(text="Generated text", citations=citations)


def test_two_claims_can_share_one_source(record_data: dict) -> None:
    claim = Claim(text="The deadline is 30 days.", citations=("returns.window.1",))
    record_data["claims"] += (claim,)
    record_data["candidate_claims"] += (CandidateClaim(**claim.model_dump()),)
    record_data["check_results"] += (CheckResult(claim_index=1, ok=True),)
    record_data["verdicts"] += (ClaimVerdict(claim_index=1, verdict="supported"),)
    assert len(RunRecord(**record_data).to_response().sources) == 1


def test_eval_origin_timestamp_normalization_and_default_id(record_data: dict) -> None:
    record_data.pop("run_id")
    record_data.update(
        origin="eval",
        eval_case_id="dev-1",
        created_at=datetime(2026, 11, 6, 14, tzinfo=timezone(timedelta(hours=2))),
    )
    record = RunRecord(**record_data)
    assert record.created_at == datetime(2026, 11, 6, 12, tzinfo=UTC)
    assert record.created_at.tzinfo is UTC
    assert len(record.run_id) == 32
    assert record.eval_case_id == "dev-1"


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "error"},
        {"status": "refused"},
        {"refusal_code": "unsupported"},
        {"claims": [], "sources": []},
        {"sources": []},
        {"prompts": []},
        {"config": {}},
        {"verdicts": []},
        {"answer_language": None},
        {"answer_language": "de"},
    ],
)
def test_response_rejects_invalid_results_and_internal_fields(record_data: dict, changes: dict):
    public = RunRecord(**record_data).to_response().model_dump()
    with pytest.raises(ValidationError):
        AskResponse(**(public | changes))


def test_empty_calls_derive_zero_totals(record_data: dict) -> None:
    record = RunRecord(**(record_data | {"calls": ()}))
    assert record.metrics.input_tokens == record.metrics.output_tokens == 0
    assert record.metrics.cost_usd == Decimal("0")
    assert json.loads(record.to_response().model_dump_json())["metrics"]["cost_usd"] == "0"


def test_cost_totals_remain_exact_under_low_decimal_precision(record_data: dict) -> None:
    call_data = record_data["calls"][0].model_dump() | {"cost_usd": Decimal("0.123456789")}
    record_data["calls"] = (CallRecord(**call_data), record_data["calls"][1])
    record = RunRecord(**record_data)
    with localcontext() as context:
        context.prec = 2
        assert record.to_response().metrics.cost_usd == Decimal("0.123461789")


def test_models_are_frozen_and_forbid_unknown_fields(record_data: dict) -> None:
    record = RunRecord(**record_data)
    models = (
        record,
        record.git,
        record.config,
        record.claims[0],
        record.candidate_claims[0],
        record.sources[0],
        record.retrieved_sections[0],
        record.check_results[0],
        record.verdicts[0],
        record.calls[0],
        record.node_timings[0],
        record.metrics,
        record.to_response(),
        RunError(code="test", message="Test failure"),
    )
    for model in models:
        field = next(iter(type(model).model_fields))
        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, field, getattr(model, field))
        with pytest.raises(ValidationError, match="Extra inputs"):
            type(model).model_validate(model.model_dump() | {"unexpected": "value"})
    assert isinstance(record.claims, tuple)
    assert isinstance(record.claims[0].citations, tuple)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_tokens", -1),
        ("output_tokens", True),
        ("output_tokens", 1.5),
        ("cost_usd", "-1"),
        ("cost_usd", "NaN"),
        ("cost_usd", "Infinity"),
        ("duration_ms", float("inf")),
    ],
)
def test_invalid_call_measurements(record_data: dict, field: str, value: object) -> None:
    data = record_data["calls"][0].model_dump() | {field: value}
    with pytest.raises(ValidationError):
        CallRecord(**data)


def test_write_round_trip_stable_utf8_and_no_overwrite(record_data: dict, tmp_path: Path) -> None:
    record = RunRecord(**record_data)
    runs_dir = tmp_path / "nested" / "runs"
    path = write_record(record, runs_dir)
    assert path == runs_dir / f"{record.run_id}.json"
    assert read_record(path) == record
    original = path.read_bytes()
    assert "Zboží" in original.decode("utf-8")
    payload = json.loads(original)
    assert list(payload) == sorted(payload)
    assert payload["calls"][0]["cost_usd"] == "0.000003"
    assert (
        original.decode("utf-8")
        == json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    with pytest.raises(FileExistsError):
        write_record(record, runs_dir)
    assert path.read_bytes() == original
    assert list(runs_dir.iterdir()) == [path]


def test_publish_link_receives_complete_file_in_same_directory(
    record_data: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = RunRecord(**record_data)
    link = os.link
    published = []

    def inspect_link(source, destination):
        source, destination = Path(source), Path(destination)
        assert source.parent == destination.parent == tmp_path
        assert not destination.exists()
        assert read_record(source) == record
        published.append(destination)
        link(source, destination)

    monkeypatch.setattr(records.os, "link", inspect_link)
    assert write_record(record, tmp_path) == published[0]
    assert [path.name for path in tmp_path.iterdir()] == [published[0].name]


def test_failed_publish_cleans_temp_and_allows_retry(
    record_data: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = RunRecord(**record_data)

    def fail_link(*args):
        raise OSError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(records.os, "link", fail_link)
        with pytest.raises(OSError, match="simulated"):
            write_record(record, tmp_path)
    assert not list(tmp_path.iterdir())
    assert read_record(write_record(record, tmp_path)) == record


def test_concurrent_writers_never_overwrite(record_data: dict, tmp_path: Path) -> None:
    record = RunRecord(**record_data)

    def attempt(_: int) -> bool:
        try:
            write_record(record, tmp_path)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(attempt, range(8))) == 1
    assert read_record(tmp_path / f"{record.run_id}.json") == record
    assert len(list(tmp_path.iterdir())) == 1


def test_reader_validates_schema(record_data: dict, tmp_path: Path) -> None:
    path = write_record(RunRecord(**record_data), tmp_path)
    payload = json.loads(path.read_text()) | {"schema_version": 999}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="schema_version"):
        read_record(path)


def test_settings_secret_never_enters_record_or_response(record_data: dict, tmp_path: Path) -> None:
    secret = "sk-test-DO-NOT-SERIALIZE"
    settings = Settings(_env_file=None, openai_api_key=secret, runtime_dir=tmp_path)
    record_data["config"] = ConfigSnapshot.from_settings(settings, top_k=5)
    record = RunRecord(**record_data)
    serializations = (
        record.model_dump_json(),
        record.to_response().model_dump_json(),
        write_record(record, settings.runs_dir).read_text(encoding="utf-8"),
    )
    for serialized in serializations:
        assert secret not in serialized
        assert "openai_api_key" not in serialized
        assert "SecretStr" not in serialized
        assert str(tmp_path) not in serialized
    assert record.config.max_output_tokens == settings.max_output_tokens
    assert record.config.price_table_version == settings.price_table_version


@pytest.mark.parametrize("dirty", [False, True])
def test_git_helper_reads_commit_and_dirty_flag(
    monkeypatch: pytest.MonkeyPatch, dirty: bool
) -> None:
    calls = []

    def git_run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == Path("repo")
        assert kwargs["check"] and kwargs["capture_output"]
        assert kwargs["timeout"] == 5
        output = "abc123\n" if command[1] == "rev-parse" else ("?? new.py\n" if dirty else "")
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(records.subprocess, "run", git_run)
    assert get_git_info(Path("repo")) == GitInfo(commit="abc123", dirty=dirty)
    assert [command[1] for command in calls] == ["rev-parse", "status"]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, "git"),
        subprocess.TimeoutExpired("git", 5),
    ],
)
def test_git_helper_tolerates_unavailable_git(monkeypatch: pytest.MonkeyPatch, error: Exception):
    def fail_git(*args, **kwargs):
        raise error

    monkeypatch.setattr(records.subprocess, "run", fail_git)
    assert get_git_info() == GitInfo(commit=None, dirty=None)
