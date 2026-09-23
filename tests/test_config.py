"""Literal optional-AI configuration; never make network calls or expose secrets."""
import os

import pytest

from src.config import AISettings, DEFAULT_MODEL, get_ai_settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr("src.config.DEFAULT_ENV_PATH", tmp_path / ".env")


def test_missing_file_defaults_keep_ai_optional():
    settings = get_ai_settings()
    assert settings.api_key == ""
    assert settings.model == DEFAULT_MODEL


def test_nonempty_environment_takes_precedence_per_setting(tmp_path, monkeypatch):
    path = tmp_path / "settings.env"
    path.write_text("OPENAI_API_KEY=file-only-test-key\nOPENAI_MODEL=file-model\n")
    monkeypatch.setenv("OPENAI_API_KEY", " process-only-test-key ")
    monkeypatch.setenv("OPENAI_MODEL", " process-model ")
    settings = get_ai_settings(path)
    assert settings.api_key == "process-only-test-key"
    assert settings.model == "process-model"
    monkeypatch.setenv("OPENAI_MODEL", " \t ")
    assert get_ai_settings(path).model == "file-model"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert get_ai_settings(path).api_key == "file-only-test-key"


@pytest.mark.parametrize("assignment,expected", [
    ("OPENAI_API_KEY=plain-test-key # local comment", "plain-test-key"),
    ("OPENAI_API_KEY=plain#test-key", "plain#test-key"),
    ('OPENAI_API_KEY="quoted#test-key" # outside comment', "quoted#test-key"),
    ("OPENAI_API_KEY='single#test-key' # outside comment", "single#test-key"),
    ('OPENAI_API_KEY="escaped\\\"quote"', 'escaped"quote'),
    ("OPENAI_API_KEY='escaped\\'quote'", "escaped'quote"),
    ('OPENAI_API_KEY="literal\\nvalue"', "literal\\nvalue"),
    (" export OPENAI_API_KEY = ' padded-test-key ' ", "padded-test-key"),
    ("export\tOPENAI_API_KEY=tab-test-key", "tab-test-key"),
    ("OPENAI_API_KEY= # empty", ""),
    ('OPENAI_API_KEY=""', ""),
])
def test_literal_quotes_and_comments(tmp_path, assignment, expected):
    path = tmp_path / ".env"
    path.write_text(assignment + "\n", encoding="utf-8")
    assert get_ai_settings(path).api_key == expected


def test_file_reloads_and_does_not_mutate_environment(tmp_path):
    path = tmp_path / ".env"
    before = dict(os.environ)
    path.write_text("OPENAI_API_KEY=first-fake-key\nOPENAI_MODEL=first-model\n")
    assert get_ai_settings().api_key == "first-fake-key"
    path.write_text("OPENAI_API_KEY=second-fake-key\nOPENAI_MODEL=second-model\n")
    settings = get_ai_settings()
    assert settings.api_key == "second-fake-key" and settings.model == "second-model"
    assert dict(os.environ) == before


def test_only_two_keys_are_loaded_without_interpolation_or_execution(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-exist"
    monkeypatch.setenv("OTHER_SECRET", "must-not-expand")
    before = dict(os.environ)
    path = tmp_path / ".env"
    path.write_text(f"UNRELATED_VAR=changed\nOPENAI_API_KEY='$(touch {marker})'\n"
                    "OPENAI_MODEL=${OTHER_SECRET}\n", encoding="utf-8")
    settings = get_ai_settings(path)
    assert settings.api_key == f"$(touch {marker})"
    assert settings.model == "${OTHER_SECRET}"
    assert not marker.exists() and dict(os.environ) == before


def test_bom_duplicate_assignments_and_invalid_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# local settings\nOPENAI_API_KEY=first-fake\nOPENAI_API_KEY=last-fake\n"
                    "OPENAI_API_KEY='unclosed-private-text\nOPENAI_API_KEY='private' trailing\n"
                    "OPENAI_MODEL=\nOPENAI_MODEL_NOPE=unknown\nnot-an-assignment\n", encoding="utf-8-sig")
    settings = get_ai_settings(path)
    assert settings.api_key == "last-fake"
    assert settings.model == DEFAULT_MODEL


def test_bad_file_encoding_or_directory_keeps_defaults(tmp_path):
    path = tmp_path / "invalid.env"
    path.write_bytes(b"\xff\xfe\xff")
    assert get_ai_settings(path) == AISettings()
    assert get_ai_settings(tmp_path) == AISettings()


def test_settings_representation_and_file_errors_do_not_expose_key(tmp_path, monkeypatch, capsys):
    secret = "never-print-this-fake-key"
    settings = AISettings(api_key=secret, model="safe-model")
    assert secret not in repr(settings) and secret not in str(settings)

    def cannot_read(*args, **kwargs):
        raise PermissionError(secret)

    monkeypatch.setattr("pathlib.Path.read_text", cannot_read)
    assert get_ai_settings(tmp_path / ".env") == AISettings()
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
