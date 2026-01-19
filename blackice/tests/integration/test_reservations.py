"""Integration test IT-008: Reservation conflicts.

Tests the file/directory reservation system that prevents agents from
conflicting on the same resources. Reservations have TTL and can be
released or transferred.

Per FR-017: Agents use reservations to claim exclusive access to files
during implementation to prevent conflicts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from blackice.primitives.types import AgentId, AgentRole


# =============================================================================
# Mock Components (for initial test writing - will be replaced by real impl)
# =============================================================================


class ReservationType(str, Enum):
    """Type of reservation."""

    FILE = "file"
    DIRECTORY = "directory"
    PATTERN = "pattern"  # e.g., "src/*.py"


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
    resource: str  # File path or directory or pattern
    resource_type: ReservationType
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: datetime | None = None
    status: ReservationStatus = ReservationStatus.ACTIVE
    reason: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if reservation has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_active(self) -> bool:
        """Check if reservation is currently active."""
        return self.status == ReservationStatus.ACTIVE and not self.is_expired


class ReservationConflictError(Exception):
    """Raised when a reservation conflicts with an existing one."""

    def __init__(self, resource: str, held_by: AgentId, message: str = ""):
        self.resource = resource
        self.held_by = held_by
        self.message = message or f"Resource {resource} is reserved by {held_by}"
        super().__init__(self.message)


class MockReservationSystem:
    """Mock reservation system for testing.

    Will be replaced by real Reservation class from blackice.colony.reservation.
    """

    def __init__(self, default_ttl_seconds: int = 300) -> None:
        self.reservations: dict[str, Reservation] = {}
        self._reservation_id = 0
        self.default_ttl_seconds = default_ttl_seconds
        self._conflict_log: list[tuple[str, AgentId, AgentId]] = []

    async def reserve(
        self,
        agent_id: AgentId,
        resource: str,
        resource_type: ReservationType = ReservationType.FILE,
        ttl_seconds: int | None = None,
        reason: str = "",
    ) -> Reservation:
        """Reserve a resource for an agent.

        Raises:
            ReservationConflictError: If resource is already reserved
        """
        # Check for conflicts
        existing = self._find_active_reservation(resource)
        if existing and existing.agent_id != agent_id:
            self._conflict_log.append((resource, agent_id, existing.agent_id))
            raise ReservationConflictError(resource, existing.agent_id)

        # Update existing reservation if same agent
        if existing and existing.agent_id == agent_id:
            # Extend TTL
            ttl = ttl_seconds or self.default_ttl_seconds
            existing.expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            return existing

        # Create new reservation
        self._reservation_id += 1
        ttl = ttl_seconds or self.default_ttl_seconds
        expires = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        reservation = Reservation(
            id=f"res-{self._reservation_id}",
            agent_id=agent_id,
            resource=resource,
            resource_type=resource_type,
            expires_at=expires,
            reason=reason,
        )
        self.reservations[reservation.id] = reservation
        return reservation

    async def release(self, reservation_id: str) -> None:
        """Release a reservation."""
        if reservation_id in self.reservations:
            self.reservations[reservation_id].status = ReservationStatus.RELEASED

    async def transfer(
        self,
        reservation_id: str,
        new_agent_id: AgentId,
    ) -> Reservation:
        """Transfer a reservation to another agent."""
        if reservation_id not in self.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")

        old_reservation = self.reservations[reservation_id]
        old_reservation.status = ReservationStatus.TRANSFERRED

        # Create new reservation for new agent
        self._reservation_id += 1
        new_reservation = Reservation(
            id=f"res-{self._reservation_id}",
            agent_id=new_agent_id,
            resource=old_reservation.resource,
            resource_type=old_reservation.resource_type,
            expires_at=old_reservation.expires_at,
            reason=f"Transferred from {old_reservation.agent_id}",
        )
        self.reservations[new_reservation.id] = new_reservation
        return new_reservation

    async def check(
        self,
        resource: str,
    ) -> Reservation | None:
        """Check if a resource is reserved."""
        return self._find_active_reservation(resource)

    async def get_agent_reservations(
        self,
        agent_id: AgentId,
    ) -> list[Reservation]:
        """Get all active reservations for an agent."""
        return [
            r for r in self.reservations.values()
            if r.agent_id == agent_id and r.is_active
        ]

    async def cleanup_expired(self) -> int:
        """Clean up expired reservations. Returns count of cleaned."""
        count = 0
        for res in self.reservations.values():
            if res.status == ReservationStatus.ACTIVE and res.is_expired:
                res.status = ReservationStatus.EXPIRED
                count += 1
        return count

    def _find_active_reservation(self, resource: str) -> Reservation | None:
        """Find an active reservation for a resource."""
        for res in self.reservations.values():
            if res.is_active and self._matches_resource(res.resource, resource):
                return res
        return None

    def _matches_resource(self, pattern: str, resource: str) -> bool:
        """Check if a resource matches a reservation pattern."""
        # Exact match
        if pattern == resource:
            return True

        # Directory match (reservation covers files inside)
        if pattern.endswith("/") and resource.startswith(pattern):
            return True

        # Pattern match (simple wildcard)
        if "*" in pattern:
            import fnmatch
            return fnmatch.fnmatch(resource, pattern)

        return False

    def get_conflicts(self) -> list[tuple[str, AgentId, AgentId]]:
        """Get log of conflicts that occurred."""
        return self._conflict_log.copy()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def reservation_system() -> MockReservationSystem:
    """Create mock reservation system."""
    return MockReservationSystem(default_ttl_seconds=60)


# =============================================================================
# Integration Tests - IT-008: Reservation Conflicts
# =============================================================================


class TestBasicReservations:
    """Tests for basic reservation functionality."""

    @pytest.mark.asyncio
    async def test_reserve_file(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test reserving a single file."""
        agent_id = AgentId("implementer-001")

        res = await reservation_system.reserve(
            agent_id=agent_id,
            resource="src/main.py",
            reason="Implementing main entry point",
        )

        assert res.agent_id == agent_id
        assert res.resource == "src/main.py"
        assert res.is_active is True

    @pytest.mark.asyncio
    async def test_reserve_directory(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test reserving a directory."""
        agent_id = AgentId("implementer-001")

        res = await reservation_system.reserve(
            agent_id=agent_id,
            resource="src/models/",
            resource_type=ReservationType.DIRECTORY,
            reason="Implementing data models",
        )

        assert res.resource == "src/models/"
        assert res.resource_type == ReservationType.DIRECTORY
        assert res.is_active is True

    @pytest.mark.asyncio
    async def test_check_reservation(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test checking if a resource is reserved."""
        agent_id = AgentId("implementer-001")

        # Initially not reserved
        existing = await reservation_system.check("src/main.py")
        assert existing is None

        # Reserve it
        await reservation_system.reserve(
            agent_id=agent_id,
            resource="src/main.py",
        )

        # Now it's reserved
        existing = await reservation_system.check("src/main.py")
        assert existing is not None
        assert existing.agent_id == agent_id


class TestReservationConflicts:
    """Tests for reservation conflicts between agents."""

    @pytest.mark.asyncio
    async def test_conflict_on_same_file(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test that two agents cannot reserve the same file."""
        agent1 = AgentId("implementer-001")
        agent2 = AgentId("implementer-002")

        # First agent reserves
        await reservation_system.reserve(
            agent_id=agent1,
            resource="src/main.py",
        )

        # Second agent cannot reserve same file
        with pytest.raises(ReservationConflictError) as exc_info:
            await reservation_system.reserve(
                agent_id=agent2,
                resource="src/main.py",
            )

        assert exc_info.value.resource == "src/main.py"
        assert exc_info.value.held_by == agent1

    @pytest.mark.asyncio
    async def test_conflict_on_file_within_directory(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test that file reservation conflicts with directory reservation."""
        agent1 = AgentId("implementer-001")
        agent2 = AgentId("implementer-002")

        # First agent reserves directory
        await reservation_system.reserve(
            agent_id=agent1,
            resource="src/models/",
            resource_type=ReservationType.DIRECTORY,
        )

        # Second agent cannot reserve file within that directory
        with pytest.raises(ReservationConflictError):
            await reservation_system.reserve(
                agent_id=agent2,
                resource="src/models/user.py",
            )

    @pytest.mark.asyncio
    async def test_conflict_logged(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test that conflicts are logged."""
        agent1 = AgentId("implementer-001")
        agent2 = AgentId("implementer-002")

        await reservation_system.reserve(
            agent_id=agent1,
            resource="src/main.py",
        )

        try:
            await reservation_system.reserve(
                agent_id=agent2,
                resource="src/main.py",
            )
        except ReservationConflictError:
            pass

        conflicts = reservation_system.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0] == ("src/main.py", agent2, agent1)

    @pytest.mark.asyncio
    async def test_same_agent_can_extend_reservation(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test that same agent can extend their own reservation."""
        agent_id = AgentId("implementer-001")

        # Initial reservation
        res1 = await reservation_system.reserve(
            agent_id=agent_id,
            resource="src/main.py",
            ttl_seconds=30,
        )
        initial_expiry = res1.expires_at

        # Same agent reserves again - should extend
        await asyncio.sleep(0.1)  # Small delay
        res2 = await reservation_system.reserve(
            agent_id=agent_id,
            resource="src/main.py",
            ttl_seconds=60,
        )

        # Should extend expiry
        assert res2.expires_at > initial_expiry


class TestReservationLifecycle:
    """Tests for reservation lifecycle (expiry, release, transfer)."""

    @pytest.mark.asyncio
    async def test_reservation_expires(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test that reservations expire after TTL."""
        agent_id = AgentId("implementer-001")

        # Reserve with very short TTL
        res = await reservation_system.reserve(
            agent_id=agent_id,
            resource="src/main.py",
            ttl_seconds=1,  # 1 second expiry
        )

        # Initially active
        assert res.is_active is True

        # Manually set expires_at to past to simulate expiry
        res.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        # Should now be expired
        assert res.is_expired is True
        assert res.is_active is False

    @pytest.mark.asyncio
    async def test_release_reservation(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test releasing a reservation."""
        agent1 = AgentId("implementer-001")
        agent2 = AgentId("implementer-002")

        # Agent 1 reserves
        res = await reservation_system.reserve(
            agent_id=agent1,
            resource="src/main.py",
        )

        # Release it
        await reservation_system.release(res.id)
        assert res.status == ReservationStatus.RELEASED

        # Now agent 2 can reserve
        res2 = await reservation_system.reserve(
            agent_id=agent2,
            resource="src/main.py",
        )
        assert res2.agent_id == agent2

    @pytest.mark.asyncio
    async def test_transfer_reservation(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test transferring a reservation between agents."""
        agent1 = AgentId("implementer-001")
        agent2 = AgentId("reviewer-001")

        # Agent 1 reserves
        res1 = await reservation_system.reserve(
            agent_id=agent1,
            resource="src/main.py",
            reason="Implementation",
        )

        # Transfer to agent 2
        res2 = await reservation_system.transfer(res1.id, agent2)

        assert res1.status == ReservationStatus.TRANSFERRED
        assert res2.agent_id == agent2
        assert res2.is_active is True

        # Agent 2 now has the reservation
        existing = await reservation_system.check("src/main.py")
        assert existing.agent_id == agent2

    @pytest.mark.asyncio
    async def test_cleanup_expired_reservations(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test cleaning up expired reservations."""
        agent_id = AgentId("implementer-001")

        # Create reservation
        res = await reservation_system.reserve(
            agent_id=agent_id,
            resource="src/main.py",
            ttl_seconds=60,
        )

        # Manually expire it
        res.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        # Cleanup
        count = await reservation_system.cleanup_expired()
        assert count == 1
        assert res.status == ReservationStatus.EXPIRED


class TestPatternReservations:
    """Tests for pattern-based reservations."""

    @pytest.mark.asyncio
    async def test_wildcard_pattern_reservation(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test wildcard pattern reservations."""
        agent1 = AgentId("implementer-001")
        agent2 = AgentId("implementer-002")

        # Reserve all Python files in src
        await reservation_system.reserve(
            agent_id=agent1,
            resource="src/*.py",
            resource_type=ReservationType.PATTERN,
        )

        # Cannot reserve specific Python file
        with pytest.raises(ReservationConflictError):
            await reservation_system.reserve(
                agent_id=agent2,
                resource="src/main.py",
            )

        # Can reserve non-Python file
        res = await reservation_system.reserve(
            agent_id=agent2,
            resource="src/config.json",
        )
        assert res.is_active is True


class TestMultipleAgentScenarios:
    """Tests for realistic multi-agent scenarios."""

    @pytest.mark.asyncio
    async def test_parallel_implementation_different_files(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test multiple agents implementing different files in parallel."""
        agents = {
            "models": AgentId("implementer-models"),
            "services": AgentId("implementer-services"),
            "api": AgentId("implementer-api"),
        }

        # Each agent reserves their area
        reservations = {}
        for area, agent_id in agents.items():
            res = await reservation_system.reserve(
                agent_id=agent_id,
                resource=f"src/{area}/",
                resource_type=ReservationType.DIRECTORY,
            )
            reservations[area] = res

        # All should be active
        for res in reservations.values():
            assert res.is_active is True

        # Each agent should have exactly one reservation
        for area, agent_id in agents.items():
            agent_res = await reservation_system.get_agent_reservations(agent_id)
            assert len(agent_res) == 1
            assert agent_res[0].resource == f"src/{area}/"

    @pytest.mark.asyncio
    async def test_implement_review_handoff(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test handoff from implementer to reviewer."""
        implementer = AgentId("implementer-001")
        reviewer = AgentId("reviewer-001")

        # Implementer reserves file during implementation
        impl_res = await reservation_system.reserve(
            agent_id=implementer,
            resource="src/feature.py",
            reason="Implementing feature",
        )

        # Implementation complete - transfer to reviewer
        review_res = await reservation_system.transfer(impl_res.id, reviewer)

        # Reviewer now has exclusive access
        assert review_res.agent_id == reviewer
        current = await reservation_system.check("src/feature.py")
        assert current.agent_id == reviewer

        # Implementer cannot modify during review
        with pytest.raises(ReservationConflictError):
            await reservation_system.reserve(
                agent_id=implementer,
                resource="src/feature.py",
            )

        # Review complete - release
        await reservation_system.release(review_res.id)

        # Now anyone can reserve
        new_res = await reservation_system.reserve(
            agent_id=implementer,
            resource="src/feature.py",
        )
        assert new_res.is_active is True

    @pytest.mark.asyncio
    async def test_deadlock_prevention_with_timeout(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test that TTL prevents deadlocks from abandoned reservations."""
        agent1 = AgentId("agent-001")
        agent2 = AgentId("agent-002")

        # Agent 1 reserves (simulating crash/abandon)
        res1 = await reservation_system.reserve(
            agent_id=agent1,
            resource="src/shared.py",
            ttl_seconds=60,
        )

        # Manually expire to simulate abandoned reservation
        res1.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        # Clean up expired
        await reservation_system.cleanup_expired()

        # Agent 2 can now reserve
        res2 = await reservation_system.reserve(
            agent_id=agent2,
            resource="src/shared.py",
        )
        assert res2.is_active is True


class TestConcurrentReservations:
    """Tests for concurrent reservation attempts."""

    @pytest.mark.asyncio
    async def test_concurrent_reservation_attempts(
        self,
        reservation_system: MockReservationSystem,
    ) -> None:
        """Test handling of concurrent reservation attempts."""
        agents = [AgentId(f"agent-{i}") for i in range(5)]

        # Simulate concurrent reservation attempts
        async def try_reserve(agent_id: AgentId) -> tuple[AgentId, bool]:
            try:
                await reservation_system.reserve(
                    agent_id=agent_id,
                    resource="src/contested.py",
                )
                return (agent_id, True)
            except ReservationConflictError:
                return (agent_id, False)

        # Run all concurrently
        tasks = [try_reserve(a) for a in agents]
        results = await asyncio.gather(*tasks)

        # Exactly one should succeed
        successes = [r for r in results if r[1]]
        failures = [r for r in results if not r[1]]

        assert len(successes) == 1
        assert len(failures) == 4

        # Winner should hold the reservation
        winner_id = successes[0][0]
        current = await reservation_system.check("src/contested.py")
        assert current.agent_id == winner_id
