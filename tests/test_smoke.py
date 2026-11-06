"""Scaffold checks that require neither credentials nor network access."""

import sys
from pathlib import Path

import pytest

import citarilo
from citarilo.cli import main
from citarilo.config import Settings


def test_package_imports() -> None:
    assert citarilo.__version__ == "0.1.0"


def test_cli_version(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["citarilo", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"citarilo {citarilo.__version__}\n"


def test_settings_without_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert Settings().openai_api_key is None


def test_settings_load_env_file_and_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=scaffold-test-key\nAPI_PORT=5175\n", encoding="utf-8"
    )
    monkeypatch.setenv("API_PORT", "5176")

    settings = Settings()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "scaffold-test-key"
    assert "scaffold-test-key" not in repr(settings)
    assert "scaffold-test-key" not in settings.model_dump_json()
    assert settings.api_port == 5176
