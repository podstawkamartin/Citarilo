"""Retrieval evaluation over the real corpus, entirely with fake SDK responses."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import citarilo.evals as evals_module
from citarilo.budget import BudgetLedger, max_cost
from citarilo.cli import main
from citarilo.config import Settings
from citarilo.corpus import load_corpus, section_documents
from citarilo.evals import estimate_retrieval, load_cases
from citarilo.index import build_index

ROOT = Path(__file__).resolve().parents[1]


class RankedClient:
    """Orthogonal section vectors and labelled questions give exact known ranks."""

    def __init__(self, settings: Settings) -> None:
        self.embeddings = self
        self.calls: list[list[str]] = []
        self.misses: set[str] = set()
        self.error: BaseException | None = None
        documents = section_documents(load_corpus(settings.data_dir))
        self.keys = [document.metadata["section_key"] for document in documents]
        self.documents = {
            document.page_content: self.vector([key])
            for key, document in zip(self.keys, documents, strict=True)
        }
        self.cases = {case.question: case for case in load_cases(ROOT / "evals/cases.jsonl")}

    def vector(self, keys: list[str] | tuple[str, ...]) -> list[float]:
        return [float(len(keys) - keys.index(key)) if key in keys else 0.0 for key in self.keys]

    def create(self, **kwargs):
        self.calls.append(kwargs["input"])
        if self.error:
            raise self.error
        vectors = []
        for text in kwargs["input"]:
            if text in self.documents:
                vectors.append(self.documents[text])
                continue
            case = self.cases[text]
            expected = case.expected_section_keys
            if case.id in self.misses:
                # Keep the first expected section of a multi-section case, losing
                # the second: section recall and all-sections case recall differ.
                keys = list(expected[:1]) if len(expected) > 1 else []
                keys += [key for key in self.keys if key not in expected][: 5 - len(keys)]
            else:
                keys = list(expected) or self.keys[:5]
            vectors.append(self.vector(keys))
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=vector) for i, vector in enumerate(vectors)],
            usage=SimpleNamespace(prompt_tokens=8, total_tokens=8),
        )


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Settings, RankedClient]:
    settings = Settings(data_dir=ROOT / "data/trailsupply", runtime_dir=tmp_path / "runtime")
    settings.embedding_dimensions = len(section_documents(load_corpus(settings.data_dir)))
    client = RankedClient(settings)
    ledger = BudgetLedger(
        settings.spend_db_path,
        settings.budget_cap_usd,
        settings.price_table_version,
        price_table=settings.price_table,
    )
    build_index(load_corpus(settings.data_dir), settings, ledger, client=client)
    client.calls.clear()
    return settings, client


def read_report(settings: Settings) -> dict:
    paths = list((settings.runtime_dir / "evals").glob("retrieval-*.json"))
    assert len(paths) == 1
    assert list((settings.runtime_dir / "evals").glob("*.tmp")) == []
    return json.loads(paths[0].read_text())


@pytest.mark.parametrize(
    "misses,exit_code",
    [
        (set(), 0),
        ({"standard-shipping-threshold"}, 0),
        ({"standard-shipping-threshold", "cancel-packed-order"}, 2),
    ],
)
def test_eval_pass_fail_reports_and_metrics(
    setup: tuple[Settings, RankedClient],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    misses: set[str],
    exit_code: int,
) -> None:
    settings, client = setup
    client.misses = misses
    loads = []
    load = evals_module.load_index

    def counted_load(*args):
        loads.append(1)
        return load(*args)

    monkeypatch.setattr(evals_module, "load_index", counted_load)
    if exit_code:
        with pytest.raises(SystemExit) as error:
            main(["eval", "--retrieval", "--max-usd", "0.01"], settings=settings, client=client)
        assert error.value.code == exit_code
    else:
        main(["eval", "--retrieval", "--max-usd", "0.01"], settings=settings, client=client)
    assert loads == [1]
    assert len(client.calls) == 21
    report = read_report(settings)
    assert report["schema_version"] == 1
    assert report["created_at"].endswith("Z")
    assert set(report["git"]) == {"commit", "dirty"}
    assert report["corpus_hash"] == load_corpus(settings.data_dir).corpus_hash
    assert report["embedding_model"] == settings.embedding_model
    assert report["embedding_dimensions"] == settings.embedding_dimensions
    assert report["top_k"] == 5
    assert report["split"] == "dev"
    assert report["metrics"]["answerable_cases"] == 17
    assert report["metrics"]["unanswerable_cases"] == 4
    assert report["metrics"]["recall_at_k"] == pytest.approx((17 - len(misses)) / 17)
    assert report["metrics"]["recall_at_5"] == report["metrics"]["recall_at_k"]
    assert report["metrics"]["section_recall_at_k"] == pytest.approx((21 - len(misses)) / 21)
    assert report["gate"] == ("FAIL" if exit_code else "PASS")
    assert report["error"] is None
    usage = report["usage"]
    ledger = BudgetLedger(
        settings.spend_db_path, settings.budget_cap_usd, settings.price_table_version
    )
    assert (
        Decimal(usage["cost_usd"])
        == ledger.group_spent_usd(usage["group_id"])
        == Decimal("0.000042")
    )
    assert usage["prompt_tokens"] == usage["total_tokens"] == 168
    assert Decimal(usage["remaining_usd"]) == Decimal("0.009958")
    for case in report["cases"]:
        assert len(case["retrieved"]) == 5
        assert case["latency_ms"] >= case["usage"][0]["duration_ms"] >= 0
        assert case["usage"][0]["reservation_id"]
        assert Decimal(case["usage"][0]["cost_usd"]) == Decimal("0.00000104")
        assert all(set(hit) == {"key", "score"} for hit in case["retrieved"])
        if case["expected_section_keys"]:
            assert case["hit"] == (case["id"] not in misses)
            assert set(case["ranks"]) == set(case["expected_section_keys"])
            if not case["hit"]:
                assert None in case["ranks"].values()
        else:
            assert case["hit"] is None
            assert case["ranks"] == {}
    output = capsys.readouterr()
    assert f"Gate recall@5 >= 0.900: {report['gate']}" in output.out
    assert "Cost $0.000042" in output.out
    assert "remaining run budget $0.009958" in output.out
    assert "n/a" in output.out
    assert "top hits:" in output.out
    assert output.err == ""


def test_eval_requires_max_usd_before_settings_or_io(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["eval", "--retrieval"])
    assert error.value.code == 1
    assert "--max-usd is required" in capsys.readouterr().err


@pytest.mark.parametrize("cap", ["0", "-1", "NaN", "Infinity", "nonsense"])
def test_eval_invalid_cap(cap: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["eval", "--retrieval", "--max-usd", cap])
    assert error.value.code == 1
    assert "finite positive" in capsys.readouterr().err


def test_eval_dry_run_no_index_ledger_or_secret(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import SecretStr

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not access credentials, index, or budget ledger")

    monkeypatch.setattr(evals_module, "BudgetLedger", forbidden)
    monkeypatch.setattr(evals_module, "load_index", forbidden)
    monkeypatch.setattr(SecretStr, "get_secret_value", forbidden)
    settings = Settings(runtime_dir=tmp_path / "runtime")
    main(["eval", "--retrieval", "--dry-run"], settings=settings)
    output = capsys.readouterr()
    cases = [case for case in load_cases(ROOT / "evals/cases.jsonl") if case.split == "dev"]
    expected = sum(
        max_cost(
            settings.embedding_model,
            len(case.question.encode("utf-8")),
            0,
            price_table=settings.price_table,
        )
        for case in cases
    )
    assert estimate_retrieval(cases, settings) == expected
    assert (
        output.out
        == f"Dry run: 21 queries; split dev; top_k 5.\nWorst-case cost: ${expected:.6f}.\n"
    )
    assert not settings.runtime_dir.exists()


def test_group_cap_stops_run_and_saves_partial_report(
    setup: tuple[Settings, RankedClient], capsys: pytest.CaptureFixture[str]
) -> None:
    settings, client = setup
    with pytest.raises(SystemExit) as error:
        main(["eval", "--retrieval", "--max-usd", "0.000025"], settings=settings, client=client)
    assert error.value.code == 1
    assert 0 < len(client.calls) < 21
    report = read_report(settings)
    assert report["gate"] == "ERROR"
    assert report["error"] == "budget_exceeded"
    assert Decimal(report["usage"]["cost_usd"]) <= Decimal("0.000025")
    assert len(report["cases"]) == len(client.calls)
    assert "Budget exceeded" in capsys.readouterr().err


def test_zero_available_group_budget_prevents_first_request(
    setup: tuple[Settings, RankedClient], capsys: pytest.CaptureFixture[str]
) -> None:
    settings, client = setup
    with pytest.raises(SystemExit) as error:
        main(["eval", "--retrieval", "--max-usd", "0.0000001"], settings=settings, client=client)
    assert error.value.code == 1
    assert client.calls == []
    report = read_report(settings)
    assert report["cases"] == []
    assert report["metrics"]["recall_at_k"] is None
    assert report["gate"] == "ERROR"
    assert "Budget exceeded" in capsys.readouterr().err


def test_heldout_only_when_requested(
    setup: tuple[Settings, RankedClient], capsys: pytest.CaptureFixture[str]
) -> None:
    settings, client = setup
    main(
        ["eval", "--retrieval", "--max-usd", "0.01", "--split", "heldout"],
        settings=settings,
        client=client,
    )
    report = read_report(settings)
    assert report["split"] == "heldout"
    assert len(report["cases"]) == len(client.calls) == 5
    assert "heldout is reserved for the final one-off run" in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing", "mismatch"])
def test_invalid_index_fails_before_requests(
    setup: tuple[Settings, RankedClient], capsys: pytest.CaptureFixture[str], problem: str
) -> None:
    settings, client = setup
    if problem == "missing":
        settings.index_manifest_path.unlink()
    else:
        settings.embedding_dimensions += 1
    with pytest.raises(SystemExit) as error:
        main(["eval", "--retrieval", "--max-usd", "0.01"], settings=settings, client=client)
    assert error.value.code == 1
    assert client.calls == []
    output = capsys.readouterr()
    assert f"Index {problem}" in output.err
    assert "run citarilo ingest explicitly" in output.err


def test_provider_error_is_sanitized_and_spend_preserved(
    setup: tuple[Settings, RankedClient], capsys: pytest.CaptureFixture[str]
) -> None:
    settings, client = setup
    client.error = RuntimeError("provider echoed fake-sensitive-key")
    with pytest.raises(SystemExit) as error:
        main(["eval", "--retrieval", "--max-usd", "0.01"], settings=settings, client=client)
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "Retrieval evaluation failed" in output.err
    assert "fake-sensitive-key" not in output.out + output.err
    report = read_report(settings)
    assert report["error"] == "retrieval_failed"
    assert Decimal(report["usage"]["cost_usd"]) > 0
    assert "fake-sensitive-key" not in json.dumps(report)


def test_overridden_top_k_keeps_approved_recall_at_5_gate(
    setup: tuple[Settings, RankedClient],
) -> None:
    settings, client = setup
    settings.top_k = 1
    main(["eval", "--retrieval", "--max-usd", "0.01"], settings=settings, client=client)
    report = read_report(settings)
    assert report["metrics"]["recall_at_k"] < 0.9
    assert report["metrics"]["recall_at_5"] == 1
    assert report["gate"] == "PASS"
    assert all(len(case["retrieved"]) == 1 for case in report["cases"])


def test_interrupted_run_never_records_pass(setup: tuple[Settings, RankedClient]) -> None:
    settings, client = setup
    create = client.create

    def interrupted(**kwargs):
        if len(client.calls) == 1:
            raise KeyboardInterrupt
        return create(**kwargs)

    client.create = interrupted
    with pytest.raises(KeyboardInterrupt):
        main(["eval", "--retrieval", "--max-usd", "0.01"], settings=settings, client=client)
    report = read_report(settings)
    assert len(report["cases"]) == 1
    assert report["metrics"]["recall_at_5"] == 1
    assert report["gate"] == "ERROR"
    assert report["error"] == "retrieval_failed"
