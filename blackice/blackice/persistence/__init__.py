"""Persistence layer for BLACKICE 3.0.

Provides durable storage for:
- Events (append-only log with hash chain)
- Artifacts (workspace files and outputs)
"""

from blackice.persistence.event_store import EventStore, EventStoreConfig

__all__ = [
    "EventStore",
    "EventStoreConfig",
]
