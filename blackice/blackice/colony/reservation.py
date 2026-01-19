"""Reservation module for BLACKICE 3.0 colony.

Manages resource reservations to prevent conflicts:
- File-level reservations
- Directory-level reservations
- TTL-based expiration
- Conflict detection
- Reservation transfer
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from blackice.primitives.types import AgentId


logger = structlog.get_logger(__name__)


class ReservationType(str, Enum):
    """Type of resource reservation."""

    FILE = "file"
    DIRECTORY = "directory"


class ReservationStatus(str, Enum):
    """Status of a reservation."""

    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"
    TRANSFERRED = "transferred"


@dataclass
class Reservation:
    """A resource reservation by an agent."""

    id: str
    agent_id: AgentId
    resource: str
    resource_type: ReservationType
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    status: ReservationStatus = ReservationStatus.ACTIVE
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if reservation has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_active(self) -> bool:
        """Check if reservation is still active."""
        return self.status == ReservationStatus.ACTIVE and not self.is_expired


@dataclass
class ReservationConfig:
    """Configuration for the reservation system."""

    default_ttl: float = 300.0  # 5 minutes
    max_reservations_per_agent: int = 50
    cleanup_interval: float = 30.0


class ReservationConflictError(Exception):
    """Raised when a reservation conflicts with an existing one."""

    def __init__(
        self,
        resource: str,
        existing: Reservation,
        message: str | None = None,
    ) -> None:
        self.resource = resource
        self.existing = existing
        super().__init__(
            message or f"Resource '{resource}' is already reserved by {existing.agent_id}"
        )


class ReservationSystem:
    """Manages resource reservations for agents.

    Provides:
    - File and directory reservations
    - Conflict detection
    - TTL-based expiration
    - Reservation transfer between agents
    """

    def __init__(self, config: ReservationConfig | None = None) -> None:
        """Initialize the reservation system.

        Args:
            config: Reservation configuration
        """
        self.config = config or ReservationConfig()
        self._reservations: dict[str, Reservation] = {}  # id -> Reservation
        self._by_resource: dict[str, str] = {}  # resource -> reservation_id
        self._by_agent: dict[AgentId, set[str]] = {}  # agent -> reservation_ids
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the reservation system."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("reservation_system_started")

    async def stop(self) -> None:
        """Stop the reservation system."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        logger.info("reservation_system_stopped", active=len(self._reservations))

    async def reserve(
        self,
        agent_id: AgentId,
        resource: str,
        resource_type: ReservationType = ReservationType.FILE,
        ttl_seconds: float | None = None,
        reason: str = "",
    ) -> Reservation:
        """Reserve a resource for an agent.

        Args:
            agent_id: The reserving agent
            resource: The resource path to reserve
            resource_type: Type of resource
            ttl_seconds: Time-to-live in seconds (None = use default)
            reason: Optional reason for reservation

        Returns:
            The created reservation

        Raises:
            ReservationConflictError: If resource is already reserved
        """
        async with self._lock:
            # Check for conflicts
            await self._check_conflicts(resource, resource_type)

            # Check agent limits
            agent_reservations = self._by_agent.get(agent_id, set())
            if len(agent_reservations) >= self.config.max_reservations_per_agent:
                raise RuntimeError(
                    f"Agent {agent_id} has reached maximum reservations"
                )

            # Create reservation
            ttl = ttl_seconds if ttl_seconds is not None else self.config.default_ttl
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl) if ttl > 0 else None

            reservation = Reservation(
                id=str(uuid4()),
                agent_id=agent_id,
                resource=resource,
                resource_type=resource_type,
                expires_at=expires_at,
                reason=reason,
            )

            # Store reservation
            self._reservations[reservation.id] = reservation
            self._by_resource[resource] = reservation.id

            if agent_id not in self._by_agent:
                self._by_agent[agent_id] = set()
            self._by_agent[agent_id].add(reservation.id)

            logger.info(
                "resource_reserved",
                reservation_id=reservation.id,
                agent_id=str(agent_id),
                resource=resource,
                resource_type=resource_type.value,
                ttl_seconds=ttl,
            )

            return reservation

    async def release(
        self,
        reservation_id: str,
        agent_id: AgentId | None = None,
    ) -> bool:
        """Release a reservation.

        Args:
            reservation_id: ID of reservation to release
            agent_id: Optional agent ID for verification

        Returns:
            True if released, False if not found
        """
        async with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                return False

            # Verify ownership if agent_id provided
            if agent_id is not None and reservation.agent_id != agent_id:
                raise ValueError(
                    f"Reservation {reservation_id} belongs to {reservation.agent_id}, not {agent_id}"
                )

            return await self._release_internal(reservation)

    async def _release_internal(self, reservation: Reservation) -> bool:
        """Internal release without lock."""
        reservation.status = ReservationStatus.RELEASED

        # Remove from indices
        self._by_resource.pop(reservation.resource, None)
        if reservation.agent_id in self._by_agent:
            self._by_agent[reservation.agent_id].discard(reservation.id)

        self._reservations.pop(reservation.id, None)

        logger.info(
            "reservation_released",
            reservation_id=reservation.id,
            resource=reservation.resource,
        )

        return True

    async def transfer(
        self,
        reservation_id: str,
        from_agent: AgentId,
        to_agent: AgentId,
    ) -> Reservation:
        """Transfer a reservation to another agent.

        Args:
            reservation_id: ID of reservation to transfer
            from_agent: Current owner
            to_agent: New owner

        Returns:
            The updated reservation

        Raises:
            ValueError: If reservation not found or not owned by from_agent
        """
        async with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                raise ValueError(f"Reservation {reservation_id} not found")

            if reservation.agent_id != from_agent:
                raise ValueError(
                    f"Reservation {reservation_id} belongs to {reservation.agent_id}, not {from_agent}"
                )

            # Check target agent limits
            target_reservations = self._by_agent.get(to_agent, set())
            if len(target_reservations) >= self.config.max_reservations_per_agent:
                raise RuntimeError(
                    f"Agent {to_agent} has reached maximum reservations"
                )

            # Update ownership
            old_agent = reservation.agent_id
            reservation.agent_id = to_agent
            reservation.status = ReservationStatus.TRANSFERRED

            # Update indices
            if old_agent in self._by_agent:
                self._by_agent[old_agent].discard(reservation.id)

            if to_agent not in self._by_agent:
                self._by_agent[to_agent] = set()
            self._by_agent[to_agent].add(reservation.id)

            # Reset status to active
            reservation.status = ReservationStatus.ACTIVE

            logger.info(
                "reservation_transferred",
                reservation_id=reservation.id,
                from_agent=str(from_agent),
                to_agent=str(to_agent),
                resource=reservation.resource,
            )

            return reservation

    async def get(self, reservation_id: str) -> Reservation | None:
        """Get a reservation by ID.

        Args:
            reservation_id: The reservation ID

        Returns:
            The reservation if found
        """
        return self._reservations.get(reservation_id)

    async def get_by_resource(self, resource: str) -> Reservation | None:
        """Get the active reservation for a resource.

        Args:
            resource: The resource path

        Returns:
            The active reservation if exists
        """
        reservation_id = self._by_resource.get(resource)
        if reservation_id is None:
            return None

        reservation = self._reservations.get(reservation_id)
        if reservation is None or not reservation.is_active:
            return None

        return reservation

    async def get_by_agent(self, agent_id: AgentId) -> list[Reservation]:
        """Get all reservations for an agent.

        Args:
            agent_id: The agent ID

        Returns:
            List of active reservations
        """
        reservation_ids = self._by_agent.get(agent_id, set())
        reservations = [
            self._reservations[rid]
            for rid in reservation_ids
            if rid in self._reservations
        ]
        return [r for r in reservations if r.is_active]

    async def is_reserved(self, resource: str) -> bool:
        """Check if a resource is reserved.

        Args:
            resource: The resource path

        Returns:
            True if reserved, False otherwise
        """
        reservation = await self.get_by_resource(resource)
        return reservation is not None and reservation.is_active

    async def check_conflict(
        self,
        resource: str,
        resource_type: ReservationType = ReservationType.FILE,
    ) -> Reservation | None:
        """Check for reservation conflicts.

        Args:
            resource: The resource to check
            resource_type: Type of resource

        Returns:
            Conflicting reservation if exists, None otherwise
        """
        async with self._lock:
            return await self._find_conflict(resource, resource_type)

    async def _check_conflicts(
        self,
        resource: str,
        resource_type: ReservationType,
    ) -> None:
        """Check for conflicts and raise if found (internal, must hold lock)."""
        conflict = await self._find_conflict(resource, resource_type)
        if conflict is not None:
            raise ReservationConflictError(resource, conflict)

    async def _find_conflict(
        self,
        resource: str,
        resource_type: ReservationType,
    ) -> Reservation | None:
        """Find a conflicting reservation (internal)."""
        # Check direct resource conflict
        reservation_id = self._by_resource.get(resource)
        if reservation_id:
            reservation = self._reservations.get(reservation_id)
            if reservation and reservation.is_active:
                return reservation

        # For file reservations, check if parent directory is reserved
        if resource_type == ReservationType.FILE:
            # Check all directory reservations
            for reservation in self._reservations.values():
                if not reservation.is_active:
                    continue
                if reservation.resource_type != ReservationType.DIRECTORY:
                    continue
                # Check if file is under reserved directory
                if resource.startswith(reservation.resource.rstrip("/") + "/"):
                    return reservation

        # For directory reservations, check if any files inside are reserved
        if resource_type == ReservationType.DIRECTORY:
            dir_prefix = resource.rstrip("/") + "/"
            for reservation in self._reservations.values():
                if not reservation.is_active:
                    continue
                if reservation.resource.startswith(dir_prefix):
                    return reservation

        return None

    async def release_all_for_agent(self, agent_id: AgentId) -> int:
        """Release all reservations for an agent.

        Args:
            agent_id: The agent ID

        Returns:
            Number of reservations released
        """
        async with self._lock:
            reservation_ids = list(self._by_agent.get(agent_id, set()))
            count = 0

            for reservation_id in reservation_ids:
                reservation = self._reservations.get(reservation_id)
                if reservation:
                    await self._release_internal(reservation)
                    count += 1

            return count

    async def extend(
        self,
        reservation_id: str,
        additional_seconds: float,
        agent_id: AgentId | None = None,
    ) -> Reservation:
        """Extend a reservation's TTL.

        Args:
            reservation_id: The reservation to extend
            additional_seconds: Seconds to add
            agent_id: Optional agent ID for verification

        Returns:
            The updated reservation
        """
        async with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                raise ValueError(f"Reservation {reservation_id} not found")

            if agent_id is not None and reservation.agent_id != agent_id:
                raise ValueError(
                    f"Reservation {reservation_id} belongs to {reservation.agent_id}"
                )

            if reservation.expires_at is not None:
                reservation.expires_at += timedelta(seconds=additional_seconds)
            else:
                reservation.expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=additional_seconds
                )

            logger.info(
                "reservation_extended",
                reservation_id=reservation_id,
                new_expires_at=reservation.expires_at.isoformat(),
            )

            return reservation

    async def get_stats(self) -> dict[str, Any]:
        """Get reservation statistics.

        Returns:
            Statistics about reservations
        """
        total = len(self._reservations)
        active = sum(1 for r in self._reservations.values() if r.is_active)
        by_type = {}
        by_agent = {}

        for reservation in self._reservations.values():
            if not reservation.is_active:
                continue

            rt = reservation.resource_type.value
            by_type[rt] = by_type.get(rt, 0) + 1

            agent = str(reservation.agent_id)
            by_agent[agent] = by_agent.get(agent, 0) + 1

        return {
            "total_reservations": total,
            "active_reservations": active,
            "expired_reservations": total - active,
            "by_type": by_type,
            "by_agent": by_agent,
        }

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired reservations."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)

                async with self._lock:
                    # Find expired reservations
                    expired = [
                        r for r in self._reservations.values()
                        if r.is_expired
                    ]

                    # Release expired
                    for reservation in expired:
                        reservation.status = ReservationStatus.EXPIRED
                        await self._release_internal(reservation)

                    if expired:
                        logger.info(
                            "reservations_cleanup",
                            expired_count=len(expired),
                            remaining=len(self._reservations),
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("reservation_cleanup_error", error=str(e))
