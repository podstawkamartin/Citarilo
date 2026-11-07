"""Budgeted answer/probe runners and offline grading with fake providers only."""

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from test_pipeline import rig as pipeline_rig  # noqa: F401
from test_pipeline import structured

import citarilo.cli as cli
import citarilo.evals as ev
from citarilo.cli import main
from citarilo.evals import EvalCase, VerifierProbe, evaluate_answers, evaluate_probes
from citarilo.grading import grading_template, summarize, write_grading_template
from citarilo.pipeline import VerificationContext, estimate_answer
from citarilo.records import read_record


@pytest.fixture
def rig(pipeline_rig, monkeypatch):  # noqa: F811
    # These contracts use a separate, deliberately tiny canonical test corpus.
    monkeypatch.setattr(ev, "_CANONICAL_DATA_DIR", pipeline_rig.context.settings.data_dir)
    return pipeline_rig


def case(case_id="returns", *, refused=False, injection=False):
    return EvalCase(
        id=case_id,
        split="dev",
        question="What is the return period?",
        language="en",
        expected_status="refused" if refused else "answered",
        expected_refusal="insufficient_evidence" if refused else None,
        required_facts=() if refused else ("Returns allowed within thirty days.",),
        forbidden_claims=("Returns are always free.",),
        supporting_sentence_ids=() if refused else ("returns.window.1", "returns.window.2"),
        expected_section_keys=() if refused else ("returns.window",),
        tags=("injection-question",) if injection else ("en",),
    )


def run(rig, cases=None, cap="0.05"):
    return evaluate_answers(
        cases or [case()], rig.context.settings, Decimal(cap), context=rig.context
    )


def graded(report, path, *, good=True):
    grades = grading_template(report, path).model_dump(mode="json")
    grades.update(grader="Independent test evaluator", graded_at="2026-11-06T12:00:00Z")
    for result, grade in zip(report.cases, grades["cases"], strict=True):
        for claim in grade["claims"]:
            claim["supported"] = good
        for claim in grade["candidate_claims"]:
            claim["candidate_supported"] = good
        for fact in grade["required_facts"]:
            fact["present"] = good and result.status == "answered"
        for claim in grade["forbidden_claims"]:
            claim["present"] = False
        grade["follows_injection"] = False
    dest = Path(str(path) + ".test-grades.json")
    dest.write_text(json.dumps(grades))
    return grades, dest


def test_answer_report_reuses_context_preserves_evidence_and_metrics(rig):
    report, path = run(rig, [case("one"), case("two")])
    assert report.complete and report.error is None and report.mode == "fake"
    assert report.group_id.startswith("answers-") and report.max_usd == Decimal("0.05")
    assert report.corpus_hash == rig.context.corpus.corpus_hash
    assert report.config.generation_model == rig.context.settings.generation_model
    assert report.git.commit and report.prompt_version
    assert len(rig.calls["embedding"]) == len(rig.calls["verify"]) == 2
    assert rig.context.group_id == "test-ask"  # Injected context remains reusable.
    assert len(list(path.parent.glob("answers-*.json"))) == 1
    assert not list(path.parent.glob("*.tmp"))
    data = json.loads(path.read_text())
    assert data["kind"] == "answers" and "openai_api_key" not in path.read_text()
    for result in report.cases:
        assert result.question and result.required_facts and result.forbidden_claims
        assert result.run_id and result.status == "answered" and result.answer_language == "en"
        assert result.sources[0].text == "Zboží můžete vrátit do 30 dnů."
        assert result.candidate_claims and result.check_results and result.verdicts
        record = read_record(rig.context.settings.runs_dir / f"{result.run_id}.json")
        assert record.origin == "eval" and record.eval_case_id == result.id
        assert result.cost_usd == record.metrics.cost_usd
    metrics = report.metrics
    times = sorted(r.latency_ms for r in report.cases)
    assert metrics["latency_p50_ms"] == sum(times) / 2
    assert metrics["latency_max_ms"] == times[-1]
    assert "latency_p95_ms" not in metrics
    assert metrics["status_accuracy"] == 1
    assert metrics["citation_precision"] == 1
    assert metrics["citation_recall"] == 0.5
    assert metrics["total_tokens"] == 610
    assert report.cost_usd == rig.context.ledger.group_spent_usd(report.group_id)
    assert ev.AnswerReport.model_validate_json(path.read_bytes()) == report


def test_answer_errors_continue_then_budget_stops(rig):
    generator = rig.context.generator
    count = 0

    def generate(messages):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("fake-secret")
        if count == 2:
            return structured(
                {"status": "insufficient_evidence", "answer_language": "en", "claims": []}
            )
        return generator.invoke(messages)

    rig.context = replace(rig.context, generator=RunnableLambda(generate))
    report, path = run(rig, [case("error"), case("refusal", refused=True), case("answer")])
    assert [r.status for r in report.cases] == ["error", "refused", "answered"]
    assert report.cases[0].error.code == "provider_error"
    assert report.cases[0].cost_usd > 0 and "fake-secret" not in path.read_text()
    assert report.metrics["refusal_code_distribution"] == {"insufficient_evidence": 1}
    assert report.metrics["status_accuracy"] == pytest.approx(2 / 3)
    report, _ = run(rig, [case("stopped"), case("unrun")], cap="0.00001")
    assert not report.complete and report.error == "budget_exceeded"
    assert report.cases[0].error.code == "budget_exceeded"
    assert report.cases[0].run_id and report.cases[0].calls
    assert report.cases[1].status is None
    assert report.cost_usd <= Decimal("0.00001")


def test_contract_error_continues_and_zero_budget_stops_before_dispatch(rig):
    rig.results["generate"] = structured({"invalid": True})
    report, _ = run(rig, [case("bad-one"), case("bad-two")])
    assert all(r.error.code == "model_contract_error" for r in report.cases)
    assert len(rig.calls["generate"]) == 2
    rig.calls["embedding"].clear()
    report, _ = run(rig, cap="0.0000001")
    assert not report.complete and not rig.calls["embedding"]
    assert report.cases[0].run_id and report.cases[0].cost_usd == 0


def test_template_all_candidates_and_no_overwrite(rig):
    rig.results["generate"]["parsed"]["claims"][0]["citations"] = ["unknown.id.1"]
    report, path = run(rig)
    assert report.cases[0].refusal_code == "check_failed"
    destination = write_grading_template(path)
    payload = json.loads(destination.read_text())
    grade = payload["cases"][0]
    assert grade["claims"] == []
    assert grade["candidate_claims"][0]["candidate_supported"] is None
    assert grade["candidate_claims"][0]["cited_sentences"] == {"unknown.id.1": None}
    assert grade["required_facts"][0]["present"] is None
    assert payload["grader"] == "" and payload["graded_at"] is None
    with pytest.raises(FileExistsError):
        write_grading_template(path)
    assert json.loads(destination.read_text()) == payload


@pytest.mark.parametrize("good,code", [(True, 0), (False, 2)])
def test_summarize_cli_pass_fail(rig, capsys, good, code):
    report, path = run(rig, [case(injection=True)])
    _, grades_path = graded(report, path, good=good)
    if code:
        with pytest.raises(SystemExit) as error:
            main(["eval", "--summarize", str(path), str(grades_path)])
        assert error.value.code == code
    else:
        main(["eval", "--summarize", str(path), str(grades_path)])
    summary = json.loads(Path(str(path) + ".summary.json").read_text())
    assert summary["gate"] == ("PASS" if good else "FAIL")
    assert summary["answered_correct_rate"] == int(good)
    assert summary["case_correctness"]["returns"] is good
    assert summary["successful_injections"] == 0  # A merely wrong answer is not injection success.
    assert summary["verifier_benefit"]["misses"] == int(not good)
    assert "Overall:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "problem",
    [
        "null",
        "ids",
        "duplicate",
        "extra",
        "coerce",
        "text",
        "fact",
        "hash",
        "metadata",
        "candidate",
        "injection",
    ],
)
def test_invalid_grades_exit_one(rig, capsys, problem):
    report, path = run(rig, [case(injection=True)])
    grades, dest = graded(report, path)
    row = grades["cases"][0]
    if problem == "null":
        row["claims"][0]["supported"] = None
    elif problem == "ids":
        row["case_id"] = "other"
    elif problem == "duplicate":
        grades["cases"].append(row)
    elif problem == "extra":
        row["unknown"] = True
    elif problem == "coerce":
        row["claims"][0]["supported"] = "true"
    elif problem == "text":
        row["claims"][0]["text"] = "changed"
    elif problem == "fact":
        row["required_facts"][0]["fact"] = "changed"
    elif problem == "hash":
        grades["report_sha256"] = "other"
    elif problem == "metadata":
        grades["graded_at"] = "2026-11-06T12:00:00"
    elif problem == "candidate":
        row["candidate_claims"] = []
    elif problem == "injection":
        row["follows_injection"] = None
    dest.write_text(json.dumps(grades))
    with pytest.raises(SystemExit) as error:
        main(["eval", "--summarize", str(path), str(dest)])
    assert error.value.code == 1
    assert "Invalid/incomplete" in capsys.readouterr().err
    assert not Path(str(path) + ".summary.json").exists()


@pytest.mark.parametrize("forbidden", [True, False])
def test_injection_success_requires_final_output_and_explicit_compliance(rig, forbidden):
    report, path = run(rig, [case(injection=True), case("unanswerable", refused=True)])
    grades, dest = graded(report, path, good=False)
    row = grades["cases"][0]
    if forbidden:
        row["forbidden_claims"][0]["present"] = True
    else:
        row["follows_injection"] = True
        row["injection_note"] = "Final output followed the requested rule override."
    dest.write_text(json.dumps(grades))
    summary, _ = summarize(path, dest)
    assert summary["successful_injections"] == 1
    assert summary["answered_unanswerables"] == 1
    assert summary["gates"]["zero_successful_injections"] == "FAIL"
    assert summary["gates"]["zero_answered_unanswerables"] == "FAIL"


def test_verifier_benefit_and_refused_cases(rig):
    rig.results["verify"]["parsed"]["verdicts"][0]["verdict"] = "unsupported"
    report, path = run(
        rig, [case("caught"), case("false-reject"), case("unanswerable", refused=True)]
    )
    grades, dest = graded(report, path)
    grades["cases"][0]["candidate_claims"][0]["candidate_supported"] = False
    dest.write_text(json.dumps(grades))
    summary, _ = summarize(path, dest)
    assert summary["verifier_benefit"] == {
        "caught": 1,
        "false_rejections": 2,
        "misses": 0,
        "not_verified": 0,
    }
    assert summary["case_correctness"]["unanswerable"]
    assert summary["answered_correct_rate"] == 0


def test_incomplete_run_never_passes_and_check_rejected_is_not_verifier_caught(rig):
    report, path = run(rig, [case("unrun")], cap="0.0000001")
    _, dest = graded(report, path)
    summary, _ = summarize(path, dest)
    assert summary["gate"] == "FAIL" and summary["gates"]["complete_without_errors"] == "FAIL"
    rig.results["generate"]["parsed"]["claims"][0]["text"] = "Within 999 days."
    report, path = run(rig)
    _, dest = graded(report, path, good=False)
    summary, _ = summarize(path, dest)
    assert summary["verifier_benefit"] == {
        "caught": 0,
        "false_rejections": 0,
        "misses": 0,
        "not_verified": 1,
    }


def probes():
    return [
        VerifierProbe(
            id=name,
            claim="Returns within thirty days.",
            cited_sentence_ids=("returns.window.1",),
            expected_verdict=verdict,
            pair_id="returns-pair",
            note="Test pair.",
        )
        for name, verdict in (("supported", "supported"), ("near-miss", "unsupported"))
    ]


def probe_context(rig, verifier=None):
    return VerificationContext(
        settings=rig.context.settings,
        ledger=rig.context.ledger,
        verifier=verifier or rig.context.verifier,
    )


def test_probes_only_verify_false_accept_gate_and_full_context(rig):
    report, path = evaluate_probes(
        probes(), rig.context.settings, Decimal("0.05"), context=probe_context(rig)
    )
    assert rig.calls["generate"] == rig.calls["embedding"] == []
    assert len(rig.calls["verify"]) == 2
    assert report.mode == "fake" and report.gate == "FAIL"
    assert report.metrics["false_accepts"] == 1 and report.metrics["false_rejects"] == 0
    assert report.metrics["exact_verdict_accuracy"] == 0.5
    assert report.pairs["returns-pair"].probe_ids == ("supported", "near-miss")
    assert report.pairs["returns-pair"].complete
    assert not report.pairs["returns-pair"].all_verdicts_correct
    assert "Poplatek je 79 Kč." in str(rig.calls["verify"][0])
    assert len(report.cases[0].retrieved_sections[0].sentences) == 2
    assert report.cost_usd == rig.context.ledger.group_spent_usd(report.group_id)
    assert ev.ProbeReport.model_validate_json(path.read_bytes()) == report


def test_probes_false_rejects_errors_and_budget_stop(rig):
    rig.results["verify"]["parsed"]["verdicts"][0]["verdict"] = "contradicted"
    report, _ = evaluate_probes(
        probes(), rig.context.settings, Decimal("0.05"), context=probe_context(rig)
    )
    assert report.metrics["false_rejects"] == 1 and report.gate == "PASS"
    rig.results["verify"] = RuntimeError("fake-sensitive")
    report, path = evaluate_probes(
        probes(), rig.context.settings, Decimal("0.05"), context=probe_context(rig)
    )
    assert all(c.error.code == "provider_error" for c in report.cases)
    assert report.gate == "FAIL" and report.cost_usd > 0
    assert "fake-sensitive" not in path.read_text()
    rig.calls["verify"].clear()
    report, _ = evaluate_probes(
        probes(), rig.context.settings, Decimal("0.0000001"), context=probe_context(rig)
    )
    assert not report.complete and report.error == "budget_exceeded"
    assert report.cases[0].error and report.cases[1].latency_ms is None
    assert not rig.calls["verify"]


@pytest.mark.parametrize(
    "mode,expected", [("--live", "21 answer cases; split dev"), ("--probes", "24 verifier probes")]
)
def test_dry_run_no_ledger_index_context_or_secrets(mode, expected, tmp_path, monkeypatch, capsys):
    from pydantic import SecretStr

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must be offline and independent of credentials/index/ledger")

    monkeypatch.setattr(ev, "BudgetLedger", forbidden)
    monkeypatch.setattr(ev.pipeline, "create_live_context", forbidden)
    monkeypatch.setattr(ev.pipeline, "create_verification_context", forbidden)
    monkeypatch.setattr(SecretStr, "get_secret_value", forbidden)
    settings = cli.Settings(_env_file=None, runtime_dir=tmp_path / "missing")
    main(["eval", mode, "--max-usd", "0.05", "--dry-run"], settings=settings)
    output = capsys.readouterr().out
    assert expected in output and "Worst-case cost: $" in output
    assert not settings.runtime_dir.exists()


def test_estimate_bounds_actual_reservations(rig):
    report, _ = run(rig)
    result = report.cases[0]
    estimate = estimate_answer(result.question, rig.context.corpus, rig.context.settings)
    assert estimate > sum((c.cost_usd for c in result.calls), Decimal(0))
    assert estimate > sum(
        (
            ev.max_cost(
                c.model,
                c.input_tokens,
                c.output_tokens,
                price_table=rig.context.settings.price_table,
            )
            for c in result.calls
        ),
        Decimal(0),
    )


def test_live_cli_context_once_heldout_warning_and_error_exit(rig, monkeypatch, capsys):
    heldout = case().model_copy(update={"split": "heldout"})
    monkeypatch.setattr(cli, "load_cases", lambda _: [heldout])
    main(["eval", "--live", "--max-usd", "0.05", "--split", "heldout"], context=rig.context)
    assert "heldout is reserved" in capsys.readouterr().out
    assert len(rig.calls["generate"]) == 1
    rig.results["generate"] = RuntimeError("fake-secret")
    with pytest.raises(SystemExit) as error:
        main(["eval", "--live", "--max-usd", "0.05", "--split", "heldout"], context=rig.context)
    assert error.value.code == 1
    assert "fake-secret" not in capsys.readouterr().out


def test_live_factory_built_once_after_group_open(rig, monkeypatch, tracked_http_client):
    from copy import copy

    builds = []

    def create(settings, *, group_id):
        builds.append(group_id)
        assert rig.context.ledger.group_spent_usd(group_id) == 0
        embeddings = copy(rig.context.embeddings)
        embeddings.group_id = group_id
        embeddings.usage = []
        store = copy(rig.context.store)
        store.embedding = embeddings
        return replace(
            rig.context,
            group_id=group_id,
            embeddings=embeddings,
            store=store,
            http_client=tracked_http_client,
        )

    monkeypatch.setattr(ev, "_ledger", lambda _: rig.context.ledger)
    monkeypatch.setattr(ev.pipeline, "create_live_context", create)
    report, _ = evaluate_answers([case("one"), case("two")], rig.context.settings, Decimal("0.05"))
    assert builds == [report.group_id]
    assert len(rig.calls["generate"]) == 2
    assert tracked_http_client.is_closed
    tracked_http_client.close.assert_called_once_with()


@pytest.mark.parametrize("bad_count,expected", [(1, "PASS"), (2, "FAIL")])
def test_answerable_gate_boundary(rig, bad_count, expected):
    report, path = run(rig, [case(f"case-{i}") for i in range(5)])
    grades, dest = graded(report, path)
    for grade in grades["cases"][:bad_count]:
        grade["required_facts"][0]["present"] = False
    dest.write_text(json.dumps(grades))
    summary, _ = summarize(path, dest)
    assert summary["answered_correct_rate"] == (5 - bad_count) / 5
    assert summary["gate"] == expected


@pytest.mark.parametrize(
    "verdict,exit_code", [("supported", 2), ("contradicted", 0), ("invalid", 1)]
)
def test_probe_cli_exit_codes_with_fake_factory(
    rig, monkeypatch, capsys, tracked_http_client, verdict, exit_code
):
    rig.results["verify"]["parsed"]["verdicts"][0]["verdict"] = verdict
    monkeypatch.setattr(cli, "load_probes", lambda _: probes())

    def create(settings, ledger, group_id):
        return VerificationContext(
            settings=settings,
            ledger=ledger,
            group_id=group_id,
            verifier=rig.context.verifier,
            http_client=tracked_http_client,
        )

    monkeypatch.setattr(ev.pipeline, "create_verification_context", create)
    args = ["eval", "--probes", "--max-usd", "0.05"]
    if exit_code:
        with pytest.raises(SystemExit) as error:
            main(args, settings=rig.context.settings)
        assert error.value.code == exit_code
    else:
        main(args, settings=rig.context.settings)
    assert "Report:" in capsys.readouterr().out
    assert rig.calls["generate"] == rig.calls["embedding"] == []
    assert tracked_http_client.is_closed
    tracked_http_client.close.assert_called_once_with()


@pytest.mark.parametrize("kind", ["answers", "probes"])
@pytest.mark.parametrize("injected", [False, True])
def test_evaluation_report_failure_closes_only_created_contexts(
    rig, monkeypatch, tracked_http_client, kind, injected
):
    from copy import copy

    def save_failure(*args):
        raise OSError("Report write failed")

    monkeypatch.setattr(ev, "_save_evaluation", save_failure)
    monkeypatch.setattr(ev, "_ledger", lambda _: rig.context.ledger)
    if kind == "answers":
        context = replace(rig.context, http_client=tracked_http_client)

        def create(settings, *, group_id):
            embeddings = copy(context.embeddings)
            embeddings.group_id = group_id
            store = copy(context.store)
            store.embedding = embeddings
            return replace(context, embeddings=embeddings, store=store, group_id=group_id)

        monkeypatch.setattr(ev.pipeline, "create_live_context", create)
        evaluate, inputs = evaluate_answers, [case()]
    else:
        context = VerificationContext(
            settings=rig.context.settings,
            ledger=rig.context.ledger,
            verifier=rig.context.verifier,
            http_client=tracked_http_client,
        )
        monkeypatch.setattr(
            ev.pipeline,
            "create_verification_context",
            lambda settings, ledger, group_id: replace(context, group_id=group_id),
        )
        evaluate, inputs = evaluate_probes, probes()
    with pytest.raises(OSError, match="Report write failed"):
        evaluate(
            inputs,
            context.settings,
            Decimal("0.05"),
            context=context if injected else None,
        )
    assert tracked_http_client.is_closed is not injected
    assert tracked_http_client.close.call_count == (0 if injected else 1)


def test_template_cli_is_offline_and_never_overwrites(rig, monkeypatch, capsys):
    _, path = run(rig)

    def forbidden(*args, **kwargs):
        pytest.fail("Offline grading must not load settings")

    monkeypatch.setattr(cli, "Settings", forbidden)
    main(["eval", "--grading-template", str(path)])
    assert "Grading template:" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["eval", "--grading-template", str(path)])
    assert error.value.code == 1
    assert "not overwritten" in capsys.readouterr().err


def test_record_write_error_cannot_be_reported_as_a_success(rig, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("fake-sensitive-write-error")

    monkeypatch.setattr(ev.pipeline, "write_record", fail)
    report, path = run(rig)
    result = report.cases[0]
    assert result.status == "error" and result.error.code == "record_write_error"
    assert result.claims == result.sources == () and result.answer_language is None
    assert result.run_id and result.candidate_claims and result.calls
    assert "fake-sensitive" not in path.read_text()


def test_estimate_uses_real_reservation_bounds_and_maximal_claims(rig, monkeypatch):
    from citarilo import pipeline
    from citarilo.prompts import generation_messages, verification_messages
    from citarilo.records import CandidateClaim

    settings = rig.context.settings
    sections = pipeline.evidence_sections(rig.context.corpus)
    claims = tuple(
        CandidateClaim(text="ř" * 1000, citations=("returns.window.1", "returns.window.2"))
        for _ in range(4)
    )
    question = "What is the return period?"
    actual_reservations = (
        ev.max_cost(
            settings.embedding_model,
            len(question.encode("utf-8")),
            0,
            price_table=settings.price_table,
        )
        + pipeline.reservation_cost(
            settings,
            settings.generation_model,
            generation_messages(question, sections),
            pipeline.GenerationOutput,
        )
        + pipeline.reservation_cost(
            settings,
            settings.verification_model,
            verification_messages(question, claims, sections),
            pipeline.VerificationOutput,
        )
    )
    assert estimate_answer(question, rig.context.corpus, settings) >= actual_reservations
