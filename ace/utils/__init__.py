"""
ACE Utilities Module

Shared utilities for embeddings, FAISS indexing, database, and logging.
"""

from ace.utils.embeddings import EmbeddingService, get_embedding_service
from ace.utils.faiss_index import FAISSIndexManager, get_faiss_manager
from ace.utils.database import get_session, get_engine, init_database
from ace.utils.logging_config import configure_logging, get_logger
from ace.utils.json_mode import json_mode_context, is_json_mode_enabled

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
