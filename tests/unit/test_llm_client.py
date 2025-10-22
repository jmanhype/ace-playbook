"""Tests for lightweight LLM client helpers."""

from __future__ import annotations

import pytest
from typing import Any

from pydantic import BaseModel

from ace.llm_client import DSPyLLMClient, DummyLLMClient, LLMError


class ExampleResponse(BaseModel):
    value: int


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure ACE_ENV does not leak between tests."""

    yield
    monkeypatch.delenv("ACE_ENV", raising=False)


@pytest.mark.parametrize("env_value", ["prod", "production", "staging"])
def test_dummy_llm_client_blocks_restricted_environments(monkeypatch, env_value):
    """Instantiation should fail when ACE_ENV indicates a runtime surface."""

    monkeypatch.setenv("ACE_ENV", env_value)
    with pytest.raises(RuntimeError):
        DummyLLMClient()


def test_dummy_llm_client_allows_explicit_override(monkeypatch):
    """The guardrail can be bypassed deliberately for controlled tests."""

    monkeypatch.setenv("ACE_ENV", "production")
    client = DummyLLMClient(allow_insecure=True)
    client.register("ExampleResponse", {"value": 1})
    response = client.structured_completion(
        prompt="ignored",
        response_model=ExampleResponse,
        metadata={"response_key": "ExampleResponse"},
    )
    assert response.value == 1


def test_dummy_llm_client_rejects_non_mapping_payloads():
    """Canned responses must remain JSON-like."""

    client = DummyLLMClient()
    with pytest.raises(TypeError):
        client.register("ExampleResponse", "invalid")


def test_dummy_llm_client_requires_registered_payload():
    """Requests without registered payloads raise LLMError."""

    client = DummyLLMClient()
    with pytest.raises(LLMError):
        client.structured_completion(
            prompt="ignored",
            response_model=ExampleResponse,
        )


class _StubLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def __call__(self, prompt: str, **_: Any) -> str:
        return self.payload


def test_dspy_llm_client_with_explicit_lm():
    """DSPyLLMClient should parse responses from an explicit LM callable."""

    payload = ExampleResponse(value=7).model_dump_json()
    client = DSPyLLMClient(lm=_StubLM(payload))
    response = client.structured_completion(
        prompt="ignored",
        response_model=ExampleResponse,
    )
    assert response.value == 7


def test_dspy_llm_client_requires_configured_lm(monkeypatch):
    """An informative error is raised when no DSPy LM is configured."""

    import types
    import sys

    dummy_module = types.SimpleNamespace(settings=types.SimpleNamespace(lm=None))
    monkeypatch.setitem(sys.modules, "dspy", dummy_module)
    client = DSPyLLMClient()
    with pytest.raises(RuntimeError):
        client.structured_completion(prompt="{}", response_model=ExampleResponse)


def test_dspy_llm_client_normalizes_code_fences():
    """The DSPy client should extract JSON from fenced responses."""

    payload = [
        "```json\n{\n  \"value\": 3\n}\n```",
    ]
    assert DSPyLLMClient._normalize_output(payload) == '{\n  "value": 3\n}'
