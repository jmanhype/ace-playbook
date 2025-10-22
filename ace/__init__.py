"""ACE (Adaptive Code Evolution) framework public interface."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.14.0"

_MODEL_EXPORTS = {
    "Base",
    "Task",
    "TaskOutput",
    "Reflection",
    "InsightCandidate",
    "PlaybookBullet",
    "PlaybookStage",
    "DiffJournalEntry",
    "MergeOperation",
}

__all__ = ["__version__", *_MODEL_EXPORTS]


def __getattr__(name: str) -> Any:
    """Lazily import heavy ORM models on demand.

    Importing :mod:`ace` should stay lightweight so modules such as
    :mod:`ace.agent_learning` can be used without eagerly pulling in SQLAlchemy
    and other optional dependencies.  When callers access one of the exported
    ORM symbols we import :mod:`ace.models` just-in-time and cache the result in
    ``globals()`` for subsequent lookups.
    """

    if name in _MODEL_EXPORTS:
        models = import_module("ace.models")
        value = getattr(models, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'ace' has no attribute '{name}'")
