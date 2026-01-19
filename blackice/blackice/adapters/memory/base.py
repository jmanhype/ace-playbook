"""Base interface for Memory Providers in BLACKICE 3.0.

Memory providers abstract persistent memory storage, enabling
the system to learn from past runs and maintain context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from blackice.primitives.types import PIIPolicy, RunId


class MemoryType(str, Enum):
    """Types of memories that can be stored."""

    PATTERN = "pattern"  # Learned coding patterns
    ERROR = "error"  # Error patterns and fixes
    DECISION = "decision"  # Architectural decisions
    CONTEXT = "context"  # Project context
    PREFERENCE = "preference"  # User preferences
    FEEDBACK = "feedback"  # User feedback


@dataclass
class RetentionPolicy:
    """Policy for memory retention."""

    max_age: timedelta | None = None  # None = forever
    max_entries: int | None = None  # None = unlimited
    pii_policy: PIIPolicy = PIIPolicy.REDACT
    archive_after: timedelta | None = None  # Move to archive after this time


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: str
    memory_type: MemoryType
    content: str
    embedding: list[float] | None = None

    # Metadata
    run_id: RunId | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime | None = None
    access_count: int = 0

    # Relevance
    importance: float = 0.5  # 0-1 scale
    confidence: float = 1.0  # 0-1 scale


@dataclass
class SearchResult:
    """Result from a memory search."""

    entry: MemoryEntry
    score: float  # Similarity/relevance score
    highlights: list[str] = field(default_factory=list)


@dataclass
class ContextWindow:
    """A window of relevant context for a task."""

    entries: list[MemoryEntry]
    total_tokens: int
    truncated: bool = False


@dataclass
class HealthStatus:
    """Health status of a memory provider."""

    healthy: bool
    entry_count: int = 0
    storage_used_bytes: int = 0
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory providers.

    Implementations must provide methods for:
    - Storage (put, delete)
    - Retrieval (search, get)
    - Context building (load_context)
    - Lifecycle (apply_retention, health)
    """

    @property
    def name(self) -> str:
        """Provider name for identification."""
        ...

    async def put(
        self,
        entry: MemoryEntry,
        *,
        retention_policy: RetentionPolicy | None = None,
    ) -> str:
        """Store a memory entry.

        Args:
            entry: The memory entry to store
            retention_policy: Optional retention policy

        Returns:
            The ID of the stored entry
        """
        ...

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID.

        Args:
            entry_id: ID of the entry to retrieve

        Returns:
            The entry if found, None otherwise
        """
        ...

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry.

        Args:
            entry_id: ID of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    async def search(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for relevant memories.

        Args:
            query: Search query (semantic search)
            memory_types: Filter by memory types
            tags: Filter by tags
            limit: Maximum results to return
            min_score: Minimum relevance score

        Returns:
            List of matching entries with scores
        """
        ...

    async def load_context(
        self,
        query: str,
        *,
        max_tokens: int = 4000,
        memory_types: list[MemoryType] | None = None,
    ) -> ContextWindow:
        """Load relevant context for a task.

        Args:
            query: Context query (what the task needs)
            max_tokens: Maximum tokens in context window
            memory_types: Filter by memory types

        Returns:
            ContextWindow with relevant entries
        """
        ...

    async def apply_retention(
        self,
        policy: RetentionPolicy,
    ) -> int:
        """Apply retention policy to stored memories.

        Args:
            policy: Retention policy to apply

        Returns:
            Number of entries affected (deleted/archived)
        """
        ...

    async def health(self) -> HealthStatus:
        """Check provider health.

        Returns:
            HealthStatus indicating if provider is operational
        """
        ...


class BaseMemoryProvider(ABC):
    """Abstract base class for memory providers.

    Provides common functionality and default implementations
    for the MemoryProvider protocol.
    """

    def __init__(
        self,
        default_retention: RetentionPolicy | None = None,
    ) -> None:
        self.default_retention = default_retention or RetentionPolicy()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @abstractmethod
    async def put(
        self,
        entry: MemoryEntry,
        *,
        retention_policy: RetentionPolicy | None = None,
    ) -> str:
        """Store a memory entry."""
        ...

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID."""
        ...

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for relevant memories."""
        ...

    async def load_context(
        self,
        query: str,
        *,
        max_tokens: int = 4000,
        memory_types: list[MemoryType] | None = None,
    ) -> ContextWindow:
        """Default context loading via search."""
        results = await self.search(
            query, memory_types=memory_types, limit=50, min_score=0.3
        )

        entries: list[MemoryEntry] = []
        total_tokens = 0
        truncated = False

        for result in results:
            # Rough token estimate: 1 token ≈ 4 chars
            entry_tokens = len(result.entry.content) // 4

            if total_tokens + entry_tokens > max_tokens:
                truncated = True
                break

            entries.append(result.entry)
            total_tokens += entry_tokens

        return ContextWindow(
            entries=entries,
            total_tokens=total_tokens,
            truncated=truncated,
        )

    async def apply_retention(
        self,
        policy: RetentionPolicy,
    ) -> int:
        """Default retention: no-op. Override for actual implementation."""
        return 0

    async def health(self) -> HealthStatus:
        """Default health check."""
        try:
            # Try a simple operation
            await self.search("health_check", limit=1)
            return HealthStatus(healthy=True)
        except Exception as e:
            return HealthStatus(healthy=False, error=str(e))
