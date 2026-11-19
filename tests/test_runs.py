"""Offline listing and stub export for local run audits."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import citarilo.cli as cli_module
import citarilo.index as index_module
from citarilo.cli import main
from citarilo.config import Settings
from citarilo.evals import EvalCase
from citarilo.records import (
    CandidateClaim,
    CheckResult,
    Claim,
    ClaimVerdict,
    ConfigSnapshot,
    GitInfo,
    RetrievedSection,
    RunRecord,
    Source,
    write_record,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=None,
        data_dir=tmp_path / "corpus",
        runtime_dir=tmp_path / "runtime",
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("runs tests must not load settings or open a network client")

    monkeypatch.setattr(cli_module, "Settings", forbidden)
    monkeypatch.setattr(index_module, "OpenAI", forbidden)


def _record(run_id: str, created_at: datetime, **overrides: object) -> RunRecord:
    source = Source(
        sentence_id="returns.window.1",
        text="Zboží lze vrátit do 30 dnů.",
        doc_id="returns",
        doc_title="Vrácení zboží",
        heading="Lhůta",
    )
    claim = Claim(text="Goods can be returned within 30 days.", citations=("returns.window.1",))
    config = ConfigSnapshot.from_settings(Settings(_env_file=None, openai_api_key=None), top_k=5)
    data: dict[str, object] = {
        "run_id": run_id,
        "created_at": created_at,
        "mode": "fake",
        "origin": "api",
        "git": GitInfo(commit="abc123", dirty=False),
        "corpus_hash": "corpus",
        "prompt_version": "7",
        "config": config,
        "question": f"Question {run_id[:8]}",
        "retrieved_sections": (
            RetrievedSection(
                section_key="returns.window",
                doc_id=source.doc_id,
                doc_title=source.doc_title,
                heading=source.heading,
                score=0.5,
                sentences=(source,),
            ),
        ),
        "candidate_claims": (CandidateClaim(**claim.model_dump()),),
        "check_results": (CheckResult(claim_index=0, ok=True),),
        "verdicts": (ClaimVerdict(claim_index=0, verdict="supported", reason="Explicit."),),
        "status": "answered",
        "answer_language": "en",
        "claims": (claim,),
        "sources": (source,),
        "calls": (),
        "latency_ms": 1,
    }
    data.update(overrides)
    return RunRecord(**data)


def _ids(output: str) -> list[str]:
    return [line.split()[1] for line in output.splitlines() if line.strip()]


def test_runs_lists_newest_first_and_skips_eval_runs(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    older = _record(
        "aaaa1111bbbb2222cccc3333dddd4444",
        datetime(2026, 11, 1, tzinfo=UTC),
        question="Older question",
    )
    middle = _record(
        "bbbb1111bbbb2222cccc3333dddd4444",
        datetime(2026, 11, 17, tzinfo=UTC),
        question="Middle question",
    )
    newest = _record(
        "cccc1111bbbb2222cccc3333dddd4444",
        datetime(2026, 11, 18, tzinfo=UTC),
        question="Newest question",
    )
    evaluation = _record(
        "dddd1111bbbb2222cccc3333dddd4444",
        datetime(2026, 11, 19, tzinfo=UTC),
        question="Eval question",
        origin="eval",
        eval_case_id="returns-window",
    )
    for record in (older, middle, newest, evaluation):
        write_record(record, settings.runs_dir)

    main(["runs"], settings=settings)
    listed = capsys.readouterr().out
    assert _ids(listed) == [newest.run_id, middle.run_id, older.run_id]
    assert evaluation.run_id not in listed
    assert listed.splitlines()[0].startswith("2026-11-18T00:00:00+00:00")
    assert "  answered  7  Newest question" in listed.splitlines()[0]

    main(["runs", "--include-evals"], settings=settings)
    assert _ids(capsys.readouterr().out)[0] == evaluation.run_id

    main(["runs", "--since", "2026-11-17", "--limit", "1"], settings=settings)
    limited = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert limited == [line for line in listed.splitlines() if newest.run_id in line]


def test_failed_runs_print_check_failures_and_rejecting_verdicts(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _record(
        "eeee1111bbbb2222cccc3333dddd4444",
        datetime(2026, 11, 18, 1, tzinfo=UTC),
        question="What is the return window?",
        status="refused",
        refusal_code="unsupported",
        answer_language=None,
        claims=(),
        sources=(),
        candidate_claims=(
            CandidateClaim(text="Return within forty days.", citations=("returns.window.1",)),
            CandidateClaim(
                text="Shipping is refunded.",
                citations=("returns.window.1", "returns.shipping.2"),
            ),
            CandidateClaim(text="The window is 30 days.", citations=("returns.window.1",)),
        ),
        check_results=(
            CheckResult(
                claim_index=0,
                ok=False,
                reasons=("literal_not_in_citations", "language_mismatch"),
            ),
            CheckResult(claim_index=1, ok=True),
            CheckResult(claim_index=2, ok=True),
        ),
        verdicts=(
            ClaimVerdict(
                claim_index=0,
                verdict="unsupported",
                reason="Forty is not in the citation.",
            ),
            ClaimVerdict(claim_index=1, verdict="contradicted", reason="Shipping is excluded."),
            ClaimVerdict(claim_index=2, verdict="supported", reason="This one is fine."),
        ),
    )
    answered = _record("ffff1111bbbb2222cccc3333dddd4444", datetime(2026, 11, 18, 2, tzinfo=UTC))
    write_record(record, settings.runs_dir)
    write_record(answered, settings.runs_dir)

    main(["runs", "--failed"], settings=settings)
    output = capsys.readouterr().out
    assert answered.run_id not in output
    assert "refused/unsupported" in output
    assert "What is the return window?" in output
    assert "check 0: literal_not_in_citations, language_mismatch" in output
    assert "verdict 0 unsupported: Return within forty days. [returns.window.1]" in output
    assert "reason: Forty is not in the citation." in output
    assert (
        "verdict 1 contradicted: Shipping is refunded. [returns.window.1, returns.shipping.2]"
    ) in output
    assert "reason: Shipping is excluded." in output
    assert "This one is fine." not in output


def test_invalid_run_file_is_skipped(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _record("abab1111bbbb2222cccc3333dddd4444", datetime(2026, 11, 18, tzinfo=UTC))
    write_record(record, settings.runs_dir)
    (settings.runs_dir / "broken.json").write_text("{", encoding="utf-8")
    (settings.runs_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    main(["runs"], settings=settings)
    captured = capsys.readouterr()
    assert record.run_id in captured.out
    assert "Skipping unreadable or invalid run file broken.json" in captured.err
    assert "notes.txt" not in captured.err


def test_to_case_writes_a_stub_outside_the_eval_loader_and_refuses_duplicates(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    monkeypatch.setattr(cli_module, "_REPO_ROOT", tmp_path)
    set_dir = tmp_path / "evals" / "sample"
    set_dir.mkdir(parents=True)
    cases = set_dir / "cases.jsonl"
    cases.write_text('{"id": "already-there"}\n', encoding="utf-8")
    before = cases.read_bytes()
    answered = _record(
        "1234abcd5678ef901234abcd5678ef90",
        datetime(2026, 11, 18, tzinfo=UTC),
        question="When can goods be returned?",
        answer_language="en",
    )
    refused = _record(
        "99abcdef0123456789abcdef01234567",
        datetime(2026, 11, 18, tzinfo=UTC),
        question="Kdy se to kontroluje?",
        status="refused",
        refusal_code="check_failed",
        answer_language=None,
        claims=(),
        sources=(),
        candidate_claims=(
            CandidateClaim(
                text="Kontrola je roční.",
                citations=("conveyor.check.1", "conveyor.check.2"),
            ),
        ),
        check_results=(CheckResult(claim_index=0, ok=False, reasons=("unknown_citation",)),),
        verdicts=(),
    )
    write_record(answered, settings.runs_dir)
    write_record(refused, settings.runs_dir)

    main(["runs", "--to-case", answered.run_id, "--eval-set", "sample"], settings=settings)
    manual = set_dir / "manual_cases.jsonl"
    first = capsys.readouterr().out
    assert str(manual) in first
    assert "Eval does not load this file." in first
    assert "A human must fill required_facts" in first
    assert cases.read_bytes() == before
    stub = json.loads(manual.read_text(encoding="utf-8"))
    assert stub == {
        "id": "manual-1234abcd",
        "split": "dev",
        "question": "When can goods be returned?",
        "language": "en",
        "expected_status": "answered",
        "expected_refusal": None,
        "required_facts": [],
        "forbidden_claims": [],
        "supporting_sentence_ids": [],
        "expected_section_keys": [],
        "tags": ["manual"],
        "hint_cited_ids": ["returns.window.1"],
    }
    with pytest.raises(ValidationError):
        EvalCase.model_validate_json(json.dumps(stub))

    main(["runs", "--to-case", refused.run_id, "--eval-set", "sample"], settings=settings)
    lines = manual.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1])["language"] == "cs"
    assert json.loads(lines[1])["hint_cited_ids"] == ["conveyor.check.1", "conveyor.check.2"]
    assert "A human must fill required_facts" in capsys.readouterr().out

    with pytest.raises(SystemExit) as duplicate:
        main(["runs", "--to-case", answered.run_id, "--eval-set", "sample"], settings=settings)
    assert duplicate.value.code == 1
    assert "manual-1234abcd already exists" in capsys.readouterr().err
    assert len(manual.read_text(encoding="utf-8").splitlines()) == 2

    seeded = "abcdef991234567890abcdef99123456"
    cases.write_text(cases.read_text(encoding="utf-8") + '{"id": "manual-abcdef99"}\n')
    write_record(
        _record(seeded, datetime(2026, 11, 18, tzinfo=UTC)),
        settings.runs_dir,
    )
    with pytest.raises(SystemExit) as collision:
        main(["runs", "--to-case", seeded, "--eval-set", "sample"], settings=settings)
    assert collision.value.code == 1
    assert "manual-abcdef99 already exists" in capsys.readouterr().err
    assert "manual-abcdef99" not in manual.read_text(encoding="utf-8")
