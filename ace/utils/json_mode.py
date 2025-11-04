"""Utilities for toggling DSPy JSON adapter mode at runtime."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Optional

from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="json_mode")


class JSONModeUnavailableError(RuntimeError):
    """Raised when JSON mode is requested but the DSPy adapter is unavailable."""

_JSON_MODE_ENV = "ACE_JSON_MODE"
_JSON_MODEL_ENV = "ACE_JSON_MODEL"


def is_json_mode_enabled(explicit: Optional[bool] = None) -> bool:
    """Return ``True`` when JSON-safe generation should be used."""

    if explicit is not None:
        return bool(explicit)

    value = os.getenv(_JSON_MODE_ENV, "").strip().lower()
    if not value:
        return False
    return value in {"1", "true", "yes", "on"}


def _resolve_model(preferred: Optional[str] = None) -> Optional[str]:
    """Resolve the model identifier to use for JSON mode."""

    if preferred:
        return preferred
    value = os.getenv(_JSON_MODEL_ENV, "").strip()
    return value or None


@lru_cache(maxsize=1)
def _load_dspy():
    """Load DSPy module with error handling.

    Returns:
        dspy module if available, None otherwise
    """
    try:
        import dspy  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - defensive import guard
        logger.error("json_mode_adapter_unavailable", error=str(exc))
        return None
    return dspy


@contextmanager
def json_mode_context(
    *,
    enabled: Optional[bool] = None,
    model: Optional[str] = None,
) -> Iterator[bool]:
    """Temporarily configure DSPy to use the JSON adapter.

    Yields ``True`` when JSON mode was activated, otherwise ``False``.
    ``enabled`` can force a specific behaviour; when ``None`` the helper falls
    back to the ``ACE_JSON_MODE`` environment variable.
    """

    active = is_json_mode_enabled(enabled)
    if not active:
        yield False
        return

    dspy = _load_dspy()
    if dspy is None or not hasattr(dspy, "JSONAdapter"):
        logger.error("json_mode_required_but_unavailable")
        raise JSONModeUnavailableError(
            "DSPy JSON adapter is unavailable while JSON mode was requested."
        )

    adapter = dspy.JSONAdapter()
    lm_model = _resolve_model(model)
    context_kwargs = {"adapter": adapter}
    if lm_model:
        context_kwargs["lm"] = dspy.LM(lm_model)

    logger.debug("json_mode_enabled", model=lm_model)
    with dspy.context(**context_kwargs):
        yield True
    logger.debug("json_mode_disabled")


__all__ = ["json_mode_context", "is_json_mode_enabled", "JSONModeUnavailableError"]
