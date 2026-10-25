"""Tests for config (credential resolution) and utils helpers."""

from __future__ import annotations

import pytest

from onet_data_collector.config import load_dotenv, resolve_credentials
from onet_data_collector.exceptions import OnetConfigError, OnetValidationError
from onet_data_collector.utils import as_list, check_for_error, dig


def test_load_dotenv_parses_and_sets_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('export ONET_USERNAME="alice"\nONET_PASSWORD=secret # inline\n', encoding="utf-8")
    monkeypatch.delenv("ONET_USERNAME", raising=False)
    monkeypatch.delenv("ONET_PASSWORD", raising=False)
    parsed = load_dotenv(env)
    assert parsed["ONET_USERNAME"] == "alice"
    # inline comment is kept as part of value only if not stripped; we strip quotes not comments
    assert parsed["ONET_PASSWORD"].startswith("secret")


def test_resolve_precedence_explicit_wins(monkeypatch):
    monkeypatch.setenv("ONET_USERNAME", "envuser")
    monkeypatch.setenv("ONET_PASSWORD", "envpass")
    user, pw = resolve_credentials("explicit", "explicitpw", dotenv_path=None)
    assert (user, pw) == ("explicit", "explicitpw")


def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv("ONET_USERNAME", "envuser")
    monkeypatch.setenv("ONET_PASSWORD", "envpass")
    assert resolve_credentials(dotenv_path=None) == ("envuser", "envpass")


def test_resolve_missing_raises(monkeypatch):
    monkeypatch.delenv("ONET_USERNAME", raising=False)
    monkeypatch.delenv("ONET_PASSWORD", raising=False)
    with pytest.raises(OnetConfigError):
        resolve_credentials(dotenv_path=None, allow_prompt=False)


def test_as_list():
    assert as_list(None) == []
    assert as_list("x") == ["x"]
    assert as_list([1, 2]) == [1, 2]
    assert as_list({"a": 1}) == [{"a": 1}]


def test_dig():
    d = {"a": {"b": {"c": 1}}}
    assert dig(d, "a", "b", "c") == 1
    assert dig(d, "a", "x", default="fallback") == "fallback"
    assert dig(None, "a") is None


def test_check_for_error_raises_on_error():
    with pytest.raises(OnetValidationError):
        check_for_error({"error": "boom"})


def test_check_for_error_passthrough():
    ok = {"occupation": []}
    assert check_for_error(ok) is ok
