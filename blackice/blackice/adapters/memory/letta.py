"""Letta Memory Provider for BLACKICE 3.0.

Connects to Letta MAS at the AI Factory (192.168.1.143:8283) for
persistent memory storage with semantic search capabilities.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

import httpx

from blackice.adapters.memory.base import (
    BaseMemoryProvider,
    ContextWindow,
    HealthStatus,
    MemoryEntry,
    MemoryType,
    RetentionPolicy,
    SearchResult,
)
from blackice.infrastructure import LettaConfig, get_ai_factory_config
from blackice.primitives.errors import ProviderError


class LettaMemoryProvider(BaseMemoryProvider):
    """Letta-based memory provider using the AI Factory MAS.

    Connects to Letta at 192.168.1.143:8283 for:
    - Semantic memory storage and retrieval
    - Agent state persistence
    - Context window management
    - Multi-session memory continuity

    Example:
        ```python
        config = get_ai_factory_config()
        provider = LettaMemoryProvider(
            base_url=config.letta.base_url,
            api_token=config.letta.api_token,
        )

        # Store a memory
        entry = MemoryEntry(
            id="mem-001",
            memory_type=MemoryType.PATTERN,
            content="Prefer pytest over unittest",
        )
        await provider.put(entry)

        # Search memories
        results = await provider.search("testing best practices")
        ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_token: str | None = None,
        timeout: float = 120.0,
        agent_id: str | None = None,
        default_retention: RetentionPolicy | None = None,
    ) -> None:
        """Initialize Letta memory provider.

        Args:
            base_url: Letta API base URL (default: from config)
            api_token: Letta API token (default: from config)
            timeout: Request timeout in seconds
            agent_id: Letta agent ID for memory operations
            default_retention: Default retention policy
        """
        super().__init__(default_retention)

        # Load config if not provided
        if base_url is None or api_token is None:
            config = get_ai_factory_config()
            base_url = base_url or config.letta.base_url
            api_token = api_token or config.letta.api_token

        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.agent_id = agent_id
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "letta"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_token}",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _entry_to_letta_message(self, entry: MemoryEntry) -> dict[str, Any]:
        """Convert MemoryEntry to Letta message format."""
        return {
            "role": "system",
            "content": f"[{entry.memory_type.value.upper()}] {entry.content}",
            "metadata": {
                "blackice_id": entry.id,
                "memory_type": entry.memory_type.value,
                "tags": entry.tags,
                "importance": entry.importance,
                "confidence": entry.confidence,
                "run_id": entry.run_id,
                "source": entry.source,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
            },
        }

    def _letta_message_to_entry(self, message: dict[str, Any]) -> MemoryEntry | None:
        """Convert Letta message to MemoryEntry."""
        metadata = message.get("metadata", {})
        if not metadata.get("blackice_id"):
            return None

        content = message.get("content", "")
        # Strip the type prefix if present
        if content.startswith("[") and "]" in content:
            content = content.split("]", 1)[1].strip()

        memory_type_str = metadata.get("memory_type", "context")
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            memory_type = MemoryType.CONTEXT

        created_at = metadata.get("created_at")
        if created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.utcnow()

        return MemoryEntry(
            id=metadata.get("blackice_id", str(uuid.uuid4())),
            memory_type=memory_type,
            content=content,
            tags=metadata.get("tags", []),
            importance=metadata.get("importance", 0.5),
            confidence=metadata.get("confidence", 1.0),
            run_id=metadata.get("run_id"),
            source=metadata.get("source"),
            created_at=created_at,
        )

    async def _ensure_agent(self) -> str:
        """Ensure we have an agent ID, creating one if needed."""
        if self.agent_id:
            return self.agent_id

        client = await self._get_client()

        # List agents to find or create BLACKICE agent
        try:
            response = await client.get("/agents")
            if response.status_code == 200:
                agents = response.json()
                # Look for existing BLACKICE agent
                for agent in agents:
                    if agent.get("name", "").startswith("blackice-"):
                        self.agent_id = agent["id"]
                        return self.agent_id

            # Create new agent
            response = await client.post(
                "/agents",
                json={
                    "name": f"blackice-{uuid.uuid4().hex[:8]}",
                    "description": "BLACKICE memory agent for persistent context",
                    "system": "You are a memory agent for the BLACKICE software factory.",
                },
            )

            if response.status_code in (200, 201):
                data = response.json()
                self.agent_id = data.get("id")
                return self.agent_id

            raise ProviderError(
                f"Failed to create Letta agent: {response.text}",
                context={"status_code": response.status_code},
            )

        except httpx.RequestError as e:
            raise ProviderError(
                f"Letta API request failed: {e}",
                context={"provider": self.name},
            ) from e

    async def put(
        self,
        entry: MemoryEntry,
        *,
        retention_policy: RetentionPolicy | None = None,
    ) -> str:
        """Store a memory entry in Letta.

        Args:
            entry: The memory entry to store
            retention_policy: Optional retention policy

        Returns:
            The ID of the stored entry
        """
        client = await self._get_client()
        agent_id = await self._ensure_agent()

        message = self._entry_to_letta_message(entry)

        try:
            # Use Letta's archival memory for persistent storage
            response = await client.post(
                f"/agents/{agent_id}/archival",
                json={"text": message["content"], "metadata": message["metadata"]},
            )

            if response.status_code not in (200, 201):
                raise ProviderError(
                    f"Failed to store memory: {response.text}",
                    context={"status_code": response.status_code},
                )

            return entry.id

        except httpx.RequestError as e:
            raise ProviderError(
                f"Letta API request failed: {e}",
                context={"provider": self.name},
            ) from e

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID.

        Args:
            entry_id: ID of the entry to retrieve

        Returns:
            The entry if found, None otherwise
        """
        client = await self._get_client()
        agent_id = await self._ensure_agent()

        try:
            # Search archival memory for the entry
            response = await client.get(
                f"/agents/{agent_id}/archival",
                params={"query": entry_id, "limit": 1},
            )

            if response.status_code != 200:
                return None

            passages = response.json()
            for passage in passages:
                metadata = passage.get("metadata", {})
                if metadata.get("blackice_id") == entry_id:
                    return MemoryEntry(
                        id=entry_id,
                        memory_type=MemoryType(metadata.get("memory_type", "context")),
                        content=passage.get("text", ""),
                        tags=metadata.get("tags", []),
                        importance=metadata.get("importance", 0.5),
                        confidence=metadata.get("confidence", 1.0),
                        run_id=metadata.get("run_id"),
                        source=metadata.get("source"),
                    )

            return None

        except httpx.RequestError:
            return None

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry.

        Note: Letta's archival memory doesn't support direct deletion.
        This marks the entry as deleted via metadata.

        Args:
            entry_id: ID of the entry to delete

        Returns:
            True if marked as deleted, False if not found
        """
        # Letta archival memory is append-only
        # We can't directly delete, but we can search and mark
        entry = await self.get(entry_id)
        if not entry:
            return False

        # Store a deletion marker
        deletion_entry = MemoryEntry(
            id=f"deleted-{entry_id}",
            memory_type=MemoryType.CONTEXT,
            content=f"DELETED: {entry_id}",
            tags=["deleted"],
        )
        await self.put(deletion_entry)
        return True

    async def search(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for relevant memories using Letta's semantic search.

        Args:
            query: Search query (semantic search)
            memory_types: Filter by memory types
            tags: Filter by tags
            limit: Maximum results to return
            min_score: Minimum relevance score

        Returns:
            List of matching entries with scores
        """
        client = await self._get_client()
        agent_id = await self._ensure_agent()

        try:
            response = await client.get(
                f"/agents/{agent_id}/archival",
                params={"query": query, "limit": limit * 2},  # Get extra to filter
            )

            if response.status_code != 200:
                return []

            passages = response.json()
            results: list[SearchResult] = []

            for i, passage in enumerate(passages):
                metadata = passage.get("metadata", {})

                # Skip if no BLACKICE ID (not our memory)
                if not metadata.get("blackice_id"):
                    continue

                # Skip deleted entries
                if "deleted" in metadata.get("tags", []):
                    continue

                # Filter by memory type if specified
                if memory_types:
                    entry_type = metadata.get("memory_type", "context")
                    if entry_type not in [t.value for t in memory_types]:
                        continue

                # Filter by tags if specified
                if tags:
                    entry_tags = set(metadata.get("tags", []))
                    if not entry_tags.intersection(set(tags)):
                        continue

                # Calculate score (Letta returns in order of relevance)
                score = 1.0 - (i * 0.05)  # Decay by position
                if score < min_score:
                    continue

                entry = MemoryEntry(
                    id=metadata.get("blackice_id"),
                    memory_type=MemoryType(metadata.get("memory_type", "context")),
                    content=passage.get("text", ""),
                    tags=metadata.get("tags", []),
                    importance=metadata.get("importance", 0.5),
                    confidence=metadata.get("confidence", 1.0),
                    run_id=metadata.get("run_id"),
                    source=metadata.get("source"),
                )

                results.append(SearchResult(entry=entry, score=score))

                if len(results) >= limit:
                    break

            return results

        except httpx.RequestError as e:
            raise ProviderError(
                f"Letta search failed: {e}",
                context={"provider": self.name},
            ) from e

    async def load_context(
        self,
        query: str,
        *,
        max_tokens: int = 4000,
        memory_types: list[MemoryType] | None = None,
    ) -> ContextWindow:
        """Load relevant context for a task.

        Uses Letta's semantic search to find the most relevant memories
        that fit within the token budget.

        Args:
            query: Context query (what the task needs)
            max_tokens: Maximum tokens in context window
            memory_types: Filter by memory types

        Returns:
            ContextWindow with relevant entries
        """
        # Use the parent class implementation which calls search()
        return await super().load_context(
            query,
            max_tokens=max_tokens,
            memory_types=memory_types,
        )

    async def health(self) -> HealthStatus:
        """Check Letta API health.

        Returns:
            HealthStatus indicating if Letta is operational
        """
        start = time.monotonic()
        try:
            client = await self._get_client()
            response = await client.get("/health")
            latency = (time.monotonic() - start) * 1000

            if response.status_code == 200:
                return HealthStatus(
                    healthy=True,
                    latency_ms=latency,
                    details={
                        "provider": self.name,
                        "base_url": self.base_url,
                        "agent_id": self.agent_id,
                    },
                )

            return HealthStatus(
                healthy=False,
                error=f"Letta returned status {response.status_code}",
                latency_ms=latency,
                details={"provider": self.name},
            )

        except Exception as e:
            return HealthStatus(
                healthy=False,
                error=str(e),
                details={"provider": self.name, "base_url": self.base_url},
            )

    async def apply_retention(self, policy: RetentionPolicy) -> int:
        """Apply retention policy.

        Note: Letta's archival memory is append-only, so retention
        is handled by marking entries as archived/deleted.

        Args:
            policy: Retention policy to apply

        Returns:
            Number of entries affected
        """
        # Letta handles retention internally
        # This is a no-op for now
        return 0
