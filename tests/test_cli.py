"""CLI behavior with isolated settings and SDK fakes, never live credentials."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import SecretStr

import citarilo.cli as cli_module
import citarilo.index as index_module
from citarilo.budget import BudgetLedger
from citarilo.cli import main
from citarilo.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "returns.md").write_text(
        "# Vrácení zboží\n\n## Lhůta {#window}\nZboží můžete vrátit do 30 dnů.\n",
        encoding="utf-8",
    )
    return Settings(
        _env_file=None,
        openai_api_key=None,
        data_dir=data_dir,
        runtime_dir=tmp_path / "runtime",
        embedding_dimensions=8,
    )


@pytest.fixture(autouse=True)
def forbid_sdk_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("CLI tests must never construct a real OpenAI client")

    monkeypatch.setattr(index_module, "OpenAI", forbidden)


def test_dry_run_has_no_network_key_access_or_runtime_writes(
    settings: Settings, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not read credentials or open a budget ledger")

    monkeypatch.setattr(cli_module, "BudgetLedger", forbidden)
    monkeypatch.setattr(SecretStr, "get_secret_value", forbidden)
    main(["ingest", "--dry-run"], settings=settings)
    output = capsys.readouterr()
    assert "Dry run: 1 sections" in output.out
    assert "upper-bound tokens; 1 request(s)" in output.out
    assert "Worst-case cost: $" in output.out
    assert "Estimated cost" in output.out
    assert "not measured" in output.out
    assert output.err == ""
    assert not settings.runtime_dir.exists()


@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_missing_key_is_clear_nonzero_and_has_no_traceback(
    settings: Settings, capsys: pytest.CaptureFixture[str], key: SecretStr | None
) -> None:
    settings.openai_api_key = key
    with pytest.raises(SystemExit) as error:
        main(["ingest"], settings=settings)
    assert error.value.code != 0
    output = capsys.readouterr()
    assert output.out == ""
    assert "Set OPENAI_API_KEY" in output.err
    assert "Traceback" not in output.err
    assert not settings.runtime_dir.exists()


def test_ingest_with_injected_client(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    # This string is a dummy, and the injected client never reads it.
    settings.openai_api_key = SecretStr("fake-cli-test-key")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        fake = DeterministicFakeEmbedding(size=kwargs["dimensions"])
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=vector)
                for i, vector in enumerate(fake.embed_documents(kwargs["input"]))
            ],
            usage=SimpleNamespace(prompt_tokens=12, total_tokens=12),
        )

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    main(["ingest"], settings=settings, client=client)
    assert len(calls) == 1
    assert settings.qdrant_path.is_dir()
    manifest = json.loads(settings.index_manifest_path.read_text())
    assert manifest["usage"][0]["prompt_tokens"] == 12
    ledger = BudgetLedger(
        settings.spend_db_path, settings.budget_cap_usd, settings.price_table_version
    )
    assert ledger.summary()["settled"] == 1
    output = capsys.readouterr()
    assert "Indexed 1 sections; 12 prompt tokens, 12 total tokens" in output.out
    assert "cost $0.000002" in output.out
    assert "remaining budget $2.999998" in output.out
    assert "fake-cli-test-key" not in output.out + output.err
    assert output.err == ""


def test_provider_error_does_not_echo_credentials(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    settings.openai_api_key = SecretStr("fake-sensitive-key")

    def create(**kwargs):
        raise RuntimeError("Provider accidentally echoed fake-sensitive-key")

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with pytest.raises(SystemExit) as error:
        main(["ingest"], settings=settings, client=client)
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "Ingest failed" in output.err
    assert "fake-sensitive-key" not in output.err + output.out
    assert "Traceback" not in output.err
    ledger = BudgetLedger(
        settings.spend_db_path, settings.budget_cap_usd, settings.price_table_version
    )
    assert ledger.summary()["abandoned"] == 1
    assert ledger.spent_usd() > 0
    assert not settings.index_manifest_path.exists()


def test_invalid_settings_are_reported_without_echoing_values(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "fake-sensitive-invalid-value")
    with pytest.raises(SystemExit) as error:
        main(["ingest", "--dry-run"])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "Invalid settings" in output.err
    assert "fake-sensitive-invalid-value" not in output.err


def test_budget_exceeded_is_reported_without_dispatch(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    settings.openai_api_key = SecretStr("fake-key")
    settings.budget_cap_usd = Decimal("0.000001")
    with pytest.raises(SystemExit) as error:
        main(["ingest"], settings=settings)
    assert error.value.code == 1
    assert "Budget exceeded" in capsys.readouterr().err
    assert not settings.index_manifest_path.exists()


def test_no_command_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    main([])
    assert "ingest" in capsys.readouterr().out


def test_version_does_not_load_settings(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Version should not load settings")

    monkeypatch.setattr(cli_module, "Settings", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out == "citarilo 0.1.0\n"


def test_ingest_locked_index_has_clear_error_and_no_reservation(settings, capsys):
    settings.openai_api_key = SecretStr("fake-key")
    with (
        index_module._acquire_index_lock(settings),
        pytest.raises(SystemExit) as exc,
    ):
        main(["ingest"], settings=settings)
    assert exc.value.code == 1
    assert "The local index is in use by another Citarilo process" in capsys.readouterr().err
    ledger = BudgetLedger(
        settings.spend_db_path, settings.budget_cap_usd, settings.price_table_version
    )
    assert ledger.spent_usd() == 0 and ledger.summary()["reserved"] == 0


def test_eval_requires_bundled_set_for_custom_data(settings, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--retrieval", "--dry-run"], settings=settings)
    assert exc.value.code == 1
    message = capsys.readouterr().err
    assert "--eval-set" in message and "trailsupply" in message and "regulations" in message
    assert not settings.runtime_dir.exists()


@pytest.mark.parametrize("selected", ["unknown", "../escape", "Bad_Name"])
def test_invalid_eval_set_lists_available_sets(settings, capsys, selected):
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--eval-set", selected, "--probes", "--dry-run"], settings=settings)
    assert exc.value.code == 1
    assert "available sets" in capsys.readouterr().err
    assert not settings.runtime_dir.exists()


def test_missing_probe_file_lists_sets_without_provider(settings, capsys, monkeypatch, tmp_path):
    root = tmp_path / "repository"
    evaluation = root / "evals" / "trailsupply"
    evaluation.mkdir(parents=True)
    (evaluation / "cases.jsonl").write_text("{}\n")
    (root / "data" / "trailsupply").mkdir(parents=True)
    monkeypatch.setattr(cli_module, "_REPO_ROOT", root)
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--eval-set", "trailsupply", "--probes", "--dry-run"], settings=settings)
    assert exc.value.code == 1
    assert "available sets: trailsupply" in capsys.readouterr().err
    assert not settings.runtime_dir.exists()


def test_case_filter_dry_run_reduces_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, runtime_dir=tmp_path / "runtime")
    main(
        [
            "eval",
            "--eval-set",
            "trailsupply",
            "--retrieval",
            "--dry-run",
            "--case",
            "clearance-return-window",
            "--case",
            "standard-shipping-threshold",
            "--case",
            "standard-shipping-threshold",
        ],
        settings=settings,
    )
    assert "Dry run: 2 queries; split dev; top_k 8." in capsys.readouterr().out
    main(
        [
            "eval",
            "--eval-set",
            "trailsupply",
            "--live",
            "--dry-run",
            "--case",
            "standard-shipping-threshold",
        ],
        settings=settings,
    )
    assert "Dry run: 1 answer cases; split dev; top_k 8." in capsys.readouterr().out
    main(
        [
            "eval",
            "--eval-set",
            "trailsupply",
            "--retrieval",
            "--dry-run",
            "--split",
            "heldout",
            "--case",
            "rental-fault-reporting",
        ],
        settings=settings,
    )
    output = capsys.readouterr().out
    assert "Dry run: 1 queries; split heldout; top_k 8." in output
    assert not settings.runtime_dir.exists()


def test_probe_pair_id_selects_both_probes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, runtime_dir=tmp_path / "runtime")
    main(
        [
            "eval",
            "--eval-set",
            "trailsupply",
            "--probes",
            "--dry-run",
            "--case",
            "shipping-price-destination",
            "--case",
            "shipping-price-destination",
        ],
        settings=settings,
    )
    assert "Dry run: 2 verifier probes" in capsys.readouterr().out
    main(
        [
            "eval",
            "--eval-set",
            "trailsupply",
            "--probes",
            "--dry-run",
            "--case",
            "shipping-price-destination-supported",
        ],
        settings=settings,
    )
    assert "Dry run: 1 verifier probes" in capsys.readouterr().out
    assert not settings.runtime_dir.exists()


def test_unknown_probe_exits_before_estimate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Unknown probe must fail before estimation or a run")

    monkeypatch.setattr(cli_module, "estimate_probes", forbidden)
    monkeypatch.setattr(cli_module, "evaluate_probes", forbidden)
    settings = Settings(_env_file=None, runtime_dir=tmp_path / "runtime")
    with pytest.raises(SystemExit) as error:
        main(
            ["eval", "--eval-set", "trailsupply", "--probes", "--dry-run", "--case", "not-a-probe"],
            settings=settings,
        )
    assert error.value.code == 2
    assert "not-a-probe" in capsys.readouterr().err
    assert not settings.runtime_dir.exists()


@pytest.mark.parametrize("case_id", ["missing-case", "rental-fault-reporting"])
def test_unknown_case_exits_before_estimate(
    case_id: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Unknown id must fail before estimation or a run")

    monkeypatch.setattr(cli_module, "estimate_retrieval", forbidden)
    monkeypatch.setattr(cli_module, "evaluate_retrieval", forbidden)
    settings = Settings(_env_file=None, runtime_dir=tmp_path / "runtime")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "eval",
                "--eval-set",
                "trailsupply",
                "--retrieval",
                "--dry-run",
                "--case",
                "standard-shipping-threshold",
                "--case",
                case_id,
            ],
            settings=settings,
        )
    assert error.value.code == 2
    output = capsys.readouterr()
    assert case_id in output.err
    assert "Worst-case" not in output.out
    assert not settings.runtime_dir.exists()


@pytest.mark.parametrize(
    "argv",
    [
        [
            "eval",
            "--summarize",
            "report.json",
            "grades.json",
            "--case",
            "standard-shipping-threshold",
        ],
        ["eval", "--grading-template", "report.json", "--case", "standard-shipping-threshold"],
    ],
)
def test_case_filter_rejected_with_grading_commands(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 1
    assert "do not accept run options" in capsys.readouterr().err
