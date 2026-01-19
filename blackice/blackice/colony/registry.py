"""Registry module for BLACKICE 3.0 colony.

Manages agent registration and discovery:
- Agent registration and deregistration
- Capability-based discovery
- Service location
- Health status tracking
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from blackice.primitives.types import AgentId, AgentRole
from blackice.schemas.agent import AgentCapabilities

if TYPE_CHECKING:
    from blackice.colony.agents.base import BaseColonyAgent


logger = structlog.get_logger(__name__)


@dataclass
class AgentRegistration:
    """Registration entry for an agent."""

    agent_id: AgentId
    role: AgentRole
    capabilities: AgentCapabilities
    endpoint: str | None = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if agent is considered healthy based on heartbeat."""
        age = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
        return age < 60.0  # Healthy if heartbeat within 60 seconds


@dataclass
class RegistryConfig:
    """Configuration for the registry."""

    max_registrations: int = 100
    heartbeat_timeout: float = 60.0
    cleanup_interval: float = 30.0


class Registry:
    """Agent registry for discovery and management.

    Provides:
    - Agent registration with capabilities
    - Role-based discovery
    - Capability-based discovery
    - Health tracking via heartbeats
    """

    def __init__(self, config: RegistryConfig | None = None) -> None:
        """Initialize the registry.

        Args:
            config: Registry configuration
        """
        self.config = config or RegistryConfig()
        self._registrations: dict[AgentId, AgentRegistration] = {}
        self._by_role: dict[AgentRole, set[AgentId]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

    @property
    def count(self) -> int:
        """Get number of registered agents."""
        return len(self._registrations)

    async def start(self) -> None:
        """Start the registry and cleanup task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("registry_started")

    async def stop(self) -> None:
        """Stop the registry."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        logger.info("registry_stopped", registrations=self.count)

    async def register(
        self,
        agent: BaseColonyAgent,
        endpoint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRegistration:
        """Register an agent.

        Args:
            agent: The agent to register
            endpoint: Optional endpoint URL
            metadata: Optional metadata

        Returns:
            The registration entry

        Raises:
            RuntimeError: If max registrations reached
        """
        async with self._lock:
            if len(self._registrations) >= self.config.max_registrations:
                raise RuntimeError(
                    f"Maximum registrations ({self.config.max_registrations}) reached"
                )

            registration = AgentRegistration(
                agent_id=agent.id,
                role=agent.role,
                capabilities=agent.capabilities,
                endpoint=endpoint,
                metadata=metadata or {},
            )

            self._registrations[agent.id] = registration

            # Add to role index
            if agent.role not in self._by_role:
                self._by_role[agent.role] = set()
            self._by_role[agent.role].add(agent.id)

            logger.info(
                "agent_registered",
                agent_id=str(agent.id),
                role=agent.role.value,
                total=len(self._registrations),
            )

            return registration

    async def deregister(self, agent_id: AgentId) -> bool:
        """Deregister an agent.

        Args:
            agent_id: ID of agent to deregister

        Returns:
            True if deregistered, False if not found
        """
        async with self._lock:
            registration = self._registrations.pop(agent_id, None)
            if registration is None:
                return False

            # Remove from role index
            if registration.role in self._by_role:
                self._by_role[registration.role].discard(agent_id)

            logger.info(
                "agent_deregistered",
                agent_id=str(agent_id),
                role=registration.role.value,
            )

            return True

    async def heartbeat(self, agent_id: AgentId) -> bool:
        """Update agent heartbeat.

        Args:
            agent_id: ID of agent

        Returns:
            True if updated, False if not found
        """
        registration = self._registrations.get(agent_id)
        if registration is None:
            return False

        registration.last_heartbeat = datetime.now(timezone.utc)
        return True

    async def get(self, agent_id: AgentId) -> AgentRegistration | None:
        """Get a registration by agent ID.

        Args:
            agent_id: The agent ID

        Returns:
            The registration if found
        """
        return self._registrations.get(agent_id)

    async def get_by_role(
        self,
        role: AgentRole,
        healthy_only: bool = True,
    ) -> list[AgentRegistration]:
        """Get all agents with a specific role.

        Args:
            role: The role to filter by
            healthy_only: Only return healthy agents

        Returns:
            List of registrations
        """
        agent_ids = self._by_role.get(role, set())
        registrations = [
            self._registrations[aid]
            for aid in agent_ids
            if aid in self._registrations
        ]

        if healthy_only:
            registrations = [r for r in registrations if r.is_healthy]

        return registrations

    async def find_by_capability(
        self,
        capability: str,
        value: Any = True,
    ) -> list[AgentRegistration]:
        """Find agents with a specific capability.

        Args:
            capability: Capability name (e.g., 'can_write_code')
            value: Expected value

        Returns:
            List of matching registrations
        """
        results = []

        for registration in self._registrations.values():
            if not registration.is_healthy:
                continue

            cap_value = getattr(registration.capabilities, capability, None)
            if cap_value == value:
                results.append(registration)

        return results

    async def find_by_language(self, language: str) -> list[AgentRegistration]:
        """Find agents that support a programming language.

        Args:
            language: The language to search for

        Returns:
            List of matching registrations
        """
        results = []

        for registration in self._registrations.values():
            if not registration.is_healthy:
                continue

            if language.lower() in [l.lower() for l in registration.capabilities.languages]:
                results.append(registration)

        return results

    async def find_by_domain(self, domain: str) -> list[AgentRegistration]:
        """Find agents that work in a domain.

        Args:
            domain: The domain to search for

        Returns:
            List of matching registrations
        """
        results = []

        for registration in self._registrations.values():
            if not registration.is_healthy:
                continue

            if domain.lower() in [d.lower() for d in registration.capabilities.domains]:
                results.append(registration)

        return results

    async def get_all(
        self,
        healthy_only: bool = True,
    ) -> list[AgentRegistration]:
        """Get all registrations.

        Args:
            healthy_only: Only return healthy agents

        Returns:
            List of all registrations
        """
        registrations = list(self._registrations.values())

        if healthy_only:
            registrations = [r for r in registrations if r.is_healthy]

        return registrations

    async def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics about registered agents
        """
        total = len(self._registrations)
        healthy = sum(1 for r in self._registrations.values() if r.is_healthy)
        by_role = {
            role.value: len(ids)
            for role, ids in self._by_role.items()
        }

        return {
            "total_registrations": total,
            "healthy_agents": healthy,
            "unhealthy_agents": total - healthy,
            "by_role": by_role,
        }

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale registrations."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)

                # Find stale registrations
                stale = [
                    agent_id
                    for agent_id, reg in self._registrations.items()
                    if not reg.is_healthy
                ]

                # Remove stale registrations
                for agent_id in stale:
                    await self.deregister(agent_id)

                if stale:
                    logger.info(
                        "registry_cleanup",
                        removed=len(stale),
                        remaining=len(self._registrations),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("registry_cleanup_error", error=str(e))
