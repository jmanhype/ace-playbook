"""Tests for JSON mode runtime helpers."""

from __future__ import annotations

import pytest

from ace.utils import json_mode


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Ensure environment variables and caches are reset between tests."""

    if hasattr(json_mode._load_dspy, "cache_clear"):
        json_mode._load_dspy.cache_clear()
    monkeypatch.delenv("ACE_JSON_MODE", raising=False)
    monkeypatch.delenv("ACE_JSON_MODEL", raising=False)
    yield
    if hasattr(json_mode._load_dspy, "cache_clear"):
        json_mode._load_dspy.cache_clear()
    monkeypatch.delenv("ACE_JSON_MODE", raising=False)
    monkeypatch.delenv("ACE_JSON_MODEL", raising=False)


def test_json_mode_context_disabled(monkeypatch):
    """When disabled explicitly the context should not activate JSON mode."""

    with json_mode.json_mode_context(enabled=False) as active:
        assert active is False


def test_json_mode_requires_dspy(monkeypatch):
    """JSON mode must raise if DSPy cannot be imported."""

    monkeypatch.setenv("ACE_JSON_MODE", "on")
    monkeypatch.setattr(json_mode, "_load_dspy", lambda: None)

    with pytest.raises(json_mode.JSONModeUnavailableError):
        with json_mode.json_mode_context():
            pass


def test_json_mode_requires_json_adapter(monkeypatch):
    """JSON mode must raise if DSPy lacks the JSONAdapter helper."""

    class NoAdapterModule:
        pass

    monkeypatch.setenv("ACE_JSON_MODE", "1")
    monkeypatch.setattr(json_mode, "_load_dspy", lambda: NoAdapterModule())

    with pytest.raises(json_mode.JSONModeUnavailableError):
        with json_mode.json_mode_context():
            pass
