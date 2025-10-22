"""Lightweight LLM client abstractions with JSON-safe helpers.

The Agent Learning (EE) stack needs deterministic, JSON-safe interactions with
language models so we can parse structured outputs without heuristics.  The ACE
playbook already ships utilities for toggling DSPy JSON mode; this module wraps
those helpers into reusable clients that other components can depend on without
knowing the exact vendor SDK (OpenAI, Anthropic, etc.).

The design intentionally mirrors the minimal interface from the AgentLearningEE
project while remaining backend-agnostic:

* :class:`BaseLLMClient` exposes ``structured_completion`` returning validated
  ``pydantic`` models.
* :class:`JSONSafeLLMClient`` adds parsing/validation around a concrete
  ``_complete`` implementation.
* :class:`DummyLLMClient`` is a testing/backfill utility used by unit tests and
  the quick-start example; it returns canned structured payloads without making
  network calls.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ace.utils.json_mode import json_mode_context
from ace.utils.logging_config import get_logger

__all__ = [
    "BaseLLMClient",
    "JSONSafeLLMClient",
    "DummyLLMClient",
]

logger = get_logger(__name__, component="llm_client")

TModel = TypeVar("TModel", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the underlying model cannot satisfy the structured contract."""


class BaseLLMClient(ABC):
    """Abstract base class for structured completions."""

    @abstractmethod
    def structured_completion(
        self,
        *,
        prompt: str,
        response_model: Type[TModel],
        model: Optional[str] = None,
        temperature: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TModel:
        """Return a validated response of ``response_model`` type."""


class JSONSafeLLMClient(BaseLLMClient):
    """Helper that toggles DSPy JSON mode around a raw completion call."""

    def structured_completion(
        self,
        *,
        prompt: str,
        response_model: Type[TModel],
        model: Optional[str] = None,
        temperature: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TModel:
        metadata = dict(metadata or {})
        logger.debug(
            "structured_completion", prompt_preview=prompt[:120], model=model, temperature=temperature
        )
        with json_mode_context(model=model):
            raw = self._complete(
                prompt=prompt,
                model=model,
                temperature=temperature,
                metadata=metadata,
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise LLMError("Model response was not valid JSON") from exc

        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:  # pragma: no cover - defensive
            raise LLMError("Model response failed validation") from exc

    @abstractmethod
    def _complete(
        self,
        *,
        prompt: str,
        model: Optional[str],
        temperature: float,
        metadata: MutableMapping[str, Any],
    ) -> str:
        """Perform the actual completion and return a JSON string."""


class DummyLLMClient(BaseLLMClient):
    """Deterministic mock used for tests, demos, and local development."""

    def __init__(
        self,
        responses: Optional[Mapping[str, Any]] = None,
        *,
        default_factory: Optional[Callable[[Type[TModel], Mapping[str, Any]], Mapping[str, Any]]] = None,
    ) -> None:
        self._responses: Dict[str, Any] = dict(responses or {})
        self._default_factory = default_factory

    def register(self, key: str, value: Any) -> None:
        """Register or update a canned response used by the dummy client."""

        self._responses[key] = value

    def structured_completion(
        self,
        *,
        prompt: str,
        response_model: Type[TModel],
        model: Optional[str] = None,
        temperature: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TModel:
        key = (metadata or {}).get("response_key") or response_model.__name__
        logger.debug("dummy_completion", key=key, prompt_preview=prompt[:120])
        data: Any
        if key in self._responses:
            value = self._responses[key]
            data = value() if callable(value) else value
        elif self._default_factory is not None:
            data = self._default_factory(response_model, metadata or {})
        else:  # pragma: no cover - defensive guard
            raise LLMError(f"No dummy response registered for key '{key}'")
        try:
            return response_model.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - defensive
            raise LLMError(f"Dummy response for key '{key}' failed validation") from exc
