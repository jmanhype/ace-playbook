"""Runtime utilities for online learning and metric evaluation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "RuntimeAdapter",
    "RuntimeClient",
    "OnlineUpdatePayload",
    "OnlineUpdateResponse",
    "MetricRegistry",
    "MetricResult",
    "DEFAULT_REGISTRY",
]

_LAZY_EXPORTS = {
    "RuntimeAdapter": ("ace.runtime.adaptation", "RuntimeAdapter"),
    "RuntimeClient": ("ace.runtime.client", "RuntimeClient"),
    "OnlineUpdatePayload": ("ace.runtime.client", "OnlineUpdatePayload"),
    "OnlineUpdateResponse": ("ace.runtime.client", "OnlineUpdateResponse"),
    "MetricRegistry": ("ace.runtime.metrics", "MetricRegistry"),
    "MetricResult": ("ace.runtime.metrics", "MetricResult"),
    "DEFAULT_REGISTRY": ("ace.runtime.metrics", "DEFAULT_REGISTRY"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'ace.runtime' has no attribute '{name}'")
    module_name, attr = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value

