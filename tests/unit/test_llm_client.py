"""Tests for lightweight LLM client helpers."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ace.llm_client import DummyLLMClient, LLMError


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
