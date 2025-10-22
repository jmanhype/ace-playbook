"""Lazy utility exports to keep optional dependencies optional."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "FAISSIndexManager",
    "get_faiss_manager",
    "get_session",
    "get_engine",
    "init_database",
    "configure_logging",
    "get_logger",
    "json_mode_context",
    "is_json_mode_enabled",
]

_LAZY_EXPORTS = {
    "EmbeddingService": ("ace.utils.embeddings", "EmbeddingService"),
    "get_embedding_service": ("ace.utils.embeddings", "get_embedding_service"),
    "FAISSIndexManager": ("ace.utils.faiss_index", "FAISSIndexManager"),
    "get_faiss_manager": ("ace.utils.faiss_index", "get_faiss_manager"),
    "get_session": ("ace.utils.database", "get_session"),
    "get_engine": ("ace.utils.database", "get_engine"),
    "init_database": ("ace.utils.database", "init_database"),
    "configure_logging": ("ace.utils.logging_config", "configure_logging"),
    "get_logger": ("ace.utils.logging_config", "get_logger"),
    "json_mode_context": ("ace.utils.json_mode", "json_mode_context"),
    "is_json_mode_enabled": ("ace.utils.json_mode", "is_json_mode_enabled"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'ace.utils' has no attribute '{name}'")
    module_name, attr = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
