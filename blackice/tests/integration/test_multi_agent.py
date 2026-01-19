"""Integration test IT-003: Multi-agent consensus.

Tests that specialist agents coordinate through voting to reduce single-model
brittleness, as specified in FR-014, FR-015, FR-016, FR-017.

Per the spec:
- Multiple specialist agents (planner, implementer, reviewer, tester, security)
- Consensus voting on important decisions (architecture, approaches)
- Supervisor coordinates agent lifecycle
- Durable messaging with threaded conversations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from blackice.primitives.types import AgentId, AgentRole
from blackice.schemas.agent import (
    Agent,
    AgentExecution,
    ConsensusResult,
    ConsensusVote,
    create_agent,
)


# =============================================================================
# Mock Components (for initial test writing - will be replaced by real impl)
# =============================================================================


@dataclass
class MockMessage:
    """Mock message for agent communication."""

    id: str
    thread_id: str
    sender: AgentId
    receiver: AgentId | None  # None = broadcast
    content: str
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    correlation_id: str | None = None
    reply_to: str | None = None


class MockMessaging:
    """Mock messaging system for testing.

    Will be replaced by real Messaging class from blackice.colony.messaging.
    """

    def __init__(self) -> None:
        self.messages: list[MockMessage] = []
        self._message_id = 0
        self._thread_id = 0

    async def send(
        self,
        sender: AgentId,
        receiver: AgentId | None,
        content: str,
        thread_id: str | None = None,
        reply_to: str | None = None,
    ) -> MockMessage:
        """Send a message."""
        self._message_id += 1
        if thread_id is None:
            self._thread_id += 1
            thread_id = f"thread-{self._thread_id}"

        msg = MockMessage(
            id=f"msg-{self._message_id}",
            thread_id=thread_id,
            sender=sender,
            receiver=receiver,
            content=content,
            reply_to=reply_to,
        )
        self.messages.append(msg)
        return msg

    async def get_thread(self, thread_id: str) -> list[MockMessage]:
        """Get all messages in a thread."""
        return [m for m in self.messages if m.thread_id == thread_id]


class MockSupervisor:
    """Mock supervisor for testing agent lifecycle.

    Will be replaced by real Supervisor class from blackice.colony.supervisor.
    """

    def __init__(self) -> None:
        self.agents: dict[AgentId, Agent] = {}
        self.executions: list[AgentExecution] = []
        self._spawned: list[AgentId] = []
        self._terminated: list[AgentId] = []

    async def spawn(self, agent: Agent) -> AgentId:
        """Spawn an agent."""
        self.agents[agent.id] = agent
        agent.active = True
        self._spawned.append(agent.id)
        return agent.id

    async def terminate(self, agent_id: AgentId) -> None:
        """Terminate an agent."""
        if agent_id in self.agents:
            self.agents[agent_id].active = False
            self._terminated.append(agent_id)

    async def get_active_agents(self) -> list[Agent]:
        """Get all active agents."""
        return [a for a in self.agents.values() if a.active]

    async def record_execution(self, execution: AgentExecution) -> None:
        """Record an agent execution."""
        self.executions.append(execution)


class MockConsensus:
    """Mock consensus system for testing.

    Will be replaced by real Consensus class from blackice.colony.consensus.
    """

    def __init__(self, policy: str = "majority") -> None:
        self.policy = policy
        self.decisions: list[ConsensusResult] = []

    async def initiate(
        self,
        topic: str,
        options: list[str],
        voters: list[AgentId],
    ) -> str:
        """Initiate a consensus vote, return decision ID."""
        decision_id = f"decision-{len(self.decisions) + 1}"
        result = ConsensusResult(topic=topic)
        self.decisions.append(result)
        return decision_id

    async def vote(
        self,
        decision_id: str,
        agent_id: AgentId,
        role: AgentRole,
        choice: str,
        confidence: float = 0.8,
        reasoning: str = "",
    ) -> None:
        """Cast a vote."""
        idx = int(decision_id.split("-")[1]) - 1
        if 0 <= idx < len(self.decisions):
            vote = ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision=choice,
                confidence=confidence,
                reasoning=reasoning,
            )
            self.decisions[idx].add_vote(vote)

    async def finalize(
        self,
        decision_id: str,
        threshold: float = 0.6,
    ) -> ConsensusResult:
        """Finalize voting and determine outcome."""
        idx = int(decision_id.split("-")[1]) - 1
        result = self.decisions[idx]
        result.compute_decision(threshold)
        return result


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def messaging() -> MockMessaging:
    """Create mock messaging system."""
    return MockMessaging()


@pytest.fixture
def supervisor() -> MockSupervisor:
    """Create mock supervisor."""
    return MockSupervisor()


@pytest.fixture
def consensus() -> MockConsensus:
    """Create mock consensus system."""
    return MockConsensus()


@pytest.fixture
def agents() -> dict[AgentRole, Agent]:
    """Create specialist agents for testing."""
    return {
        AgentRole.ARCHITECT: create_agent(
            AgentRole.ARCHITECT,
            agent_id=AgentId("architect-001"),
        ),
        AgentRole.IMPLEMENTER: create_agent(
            AgentRole.IMPLEMENTER,
            agent_id=AgentId("implementer-001"),
        ),
        AgentRole.REVIEWER: create_agent(
            AgentRole.REVIEWER,
            agent_id=AgentId("reviewer-001"),
        ),
        AgentRole.TESTER: create_agent(
            AgentRole.TESTER,
            agent_id=AgentId("tester-001"),
        ),
        AgentRole.SECURITY: create_agent(
            AgentRole.SECURITY,
            agent_id=AgentId("security-001"),
        ),
    }


# =============================================================================
# Integration Tests - IT-003: Multi-Agent Consensus
# =============================================================================


class TestMultiAgentSpawning:
    """Tests for spawning and managing multiple specialist agents.

    FR-014: Multi-agent consensus
    """

    @pytest.mark.asyncio
    async def test_spawn_all_specialist_agents(
        self,
        supervisor: MockSupervisor,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test that all specialist agents can be spawned."""
        for agent in agents.values():
            await supervisor.spawn(agent)

        active = await supervisor.get_active_agents()
        assert len(active) == 5
        assert all(a.active for a in active)

        # Verify each role is represented
        roles = {a.role for a in active}
        assert AgentRole.ARCHITECT in roles
        assert AgentRole.IMPLEMENTER in roles
        assert AgentRole.REVIEWER in roles
        assert AgentRole.TESTER in roles
        assert AgentRole.SECURITY in roles

    @pytest.mark.asyncio
    async def test_agent_lifecycle_management(
        self,
        supervisor: MockSupervisor,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test spawning and terminating agents."""
        architect = agents[AgentRole.ARCHITECT]
        await supervisor.spawn(architect)

        assert architect.active is True
        active = await supervisor.get_active_agents()
        assert len(active) == 1

        await supervisor.terminate(architect.id)
        assert architect.active is False
        active = await supervisor.get_active_agents()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_execution_tracking(
        self,
        supervisor: MockSupervisor,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test that agent executions are tracked."""
        implementer = agents[AgentRole.IMPLEMENTER]
        await supervisor.spawn(implementer)

        from blackice.primitives.types import new_task_id

        execution = AgentExecution(
            agent_id=implementer.id,
            task_id=new_task_id(),
            role=implementer.role,
        )
        execution.complete(success=True, output="Implementation complete")

        await supervisor.record_execution(execution)

        assert len(supervisor.executions) == 1
        assert supervisor.executions[0].success is True


class TestAgentMessaging:
    """Tests for inter-agent messaging.

    FR-017: Durable messaging with threaded conversations
    """

    @pytest.mark.asyncio
    async def test_direct_message_between_agents(
        self,
        messaging: MockMessaging,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test sending a direct message between agents."""
        architect = agents[AgentRole.ARCHITECT]
        implementer = agents[AgentRole.IMPLEMENTER]

        msg = await messaging.send(
            sender=architect.id,
            receiver=implementer.id,
            content="Please implement the user authentication module",
        )

        assert msg.sender == architect.id
        assert msg.receiver == implementer.id
        assert "authentication" in msg.content.lower()

    @pytest.mark.asyncio
    async def test_broadcast_message_to_all_agents(
        self,
        messaging: MockMessaging,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test broadcasting a message to all agents."""
        orchestrator_id = AgentId("orchestrator-001")

        msg = await messaging.send(
            sender=orchestrator_id,
            receiver=None,  # Broadcast
            content="Build phase starting - please prepare",
        )

        assert msg.receiver is None
        assert "prepare" in msg.content.lower()

    @pytest.mark.asyncio
    async def test_threaded_conversation(
        self,
        messaging: MockMessaging,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test threaded conversation between agents."""
        architect = agents[AgentRole.ARCHITECT]
        reviewer = agents[AgentRole.REVIEWER]

        # Start a thread
        msg1 = await messaging.send(
            sender=architect.id,
            receiver=reviewer.id,
            content="I propose using microservices architecture",
        )

        # Reply in same thread
        msg2 = await messaging.send(
            sender=reviewer.id,
            receiver=architect.id,
            content="Concerns about complexity - suggest modular monolith",
            thread_id=msg1.thread_id,
            reply_to=msg1.id,
        )

        # Another reply
        msg3 = await messaging.send(
            sender=architect.id,
            receiver=reviewer.id,
            content="Agreed, modular monolith is simpler",
            thread_id=msg1.thread_id,
            reply_to=msg2.id,
        )

        # Get full thread
        thread = await messaging.get_thread(msg1.thread_id)
        assert len(thread) == 3
        assert all(m.thread_id == msg1.thread_id for m in thread)


class TestConsensusVoting:
    """Tests for consensus voting mechanisms.

    FR-015: Voting policies (majority, supermajority, unanimous, quorum, weighted)
    FR-016: Specialist agents participate in consensus
    """

    @pytest.mark.asyncio
    async def test_majority_consensus_reached(
        self,
        consensus: MockConsensus,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test that majority consensus is reached when >50% agree."""
        voter_ids = [a.id for a in agents.values()]

        decision_id = await consensus.initiate(
            topic="Database technology",
            options=["PostgreSQL", "MongoDB", "SQLite"],
            voters=voter_ids,
        )

        # 3 vote PostgreSQL, 2 vote MongoDB
        await consensus.vote(
            decision_id,
            agents[AgentRole.ARCHITECT].id,
            AgentRole.ARCHITECT,
            "PostgreSQL",
            confidence=0.9,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.IMPLEMENTER].id,
            AgentRole.IMPLEMENTER,
            "PostgreSQL",
            confidence=0.8,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.REVIEWER].id,
            AgentRole.REVIEWER,
            "PostgreSQL",
            confidence=0.7,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.TESTER].id,
            AgentRole.TESTER,
            "MongoDB",
            confidence=0.6,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.SECURITY].id,
            AgentRole.SECURITY,
            "MongoDB",
            confidence=0.5,
        )

        result = await consensus.finalize(decision_id, threshold=0.5)

        assert result.final_decision == "PostgreSQL"
        assert result.agreement_ratio > 0.5
        assert len(result.votes) == 5

    @pytest.mark.asyncio
    async def test_consensus_not_reached_below_threshold(
        self,
        consensus: MockConsensus,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test that consensus is not reached when below threshold."""
        voter_ids = [a.id for a in agents.values()]

        decision_id = await consensus.initiate(
            topic="Framework choice",
            options=["FastAPI", "Django", "Flask"],
            voters=voter_ids,
        )

        # Evenly split votes - no clear winner
        await consensus.vote(
            decision_id,
            agents[AgentRole.ARCHITECT].id,
            AgentRole.ARCHITECT,
            "FastAPI",
            confidence=0.5,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.IMPLEMENTER].id,
            AgentRole.IMPLEMENTER,
            "Django",
            confidence=0.5,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.REVIEWER].id,
            AgentRole.REVIEWER,
            "Flask",
            confidence=0.5,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.TESTER].id,
            AgentRole.TESTER,
            "FastAPI",
            confidence=0.3,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.SECURITY].id,
            AgentRole.SECURITY,
            "Django",
            confidence=0.3,
        )

        result = await consensus.finalize(decision_id, threshold=0.7)

        # No decision reached at 70% threshold
        assert result.final_decision is None
        assert result.agreement_ratio < 0.7

    @pytest.mark.asyncio
    async def test_weighted_voting_by_confidence(
        self,
        consensus: MockConsensus,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test that votes are weighted by confidence."""
        voter_ids = [a.id for a in agents.values()]

        decision_id = await consensus.initiate(
            topic="Authentication method",
            options=["JWT", "Session", "OAuth2"],
            voters=voter_ids,
        )

        # Security expert votes with high confidence for OAuth2
        await consensus.vote(
            decision_id,
            agents[AgentRole.SECURITY].id,
            AgentRole.SECURITY,
            "OAuth2",
            confidence=0.99,
            reasoning="Most secure for modern apps",
        )

        # Others vote JWT with lower confidence
        await consensus.vote(
            decision_id,
            agents[AgentRole.ARCHITECT].id,
            AgentRole.ARCHITECT,
            "JWT",
            confidence=0.3,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.IMPLEMENTER].id,
            AgentRole.IMPLEMENTER,
            "JWT",
            confidence=0.3,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.REVIEWER].id,
            AgentRole.REVIEWER,
            "JWT",
            confidence=0.3,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.TESTER].id,
            AgentRole.TESTER,
            "JWT",
            confidence=0.3,
        )

        result = await consensus.finalize(decision_id, threshold=0.4)

        # JWT wins due to total weight: 4 * 0.3 = 1.2 vs OAuth2: 0.99
        # Total = 2.19, JWT ratio = 1.2/2.19 = 55%
        assert result.final_decision == "JWT"


class TestSecurityReviewFlow:
    """Tests for security review workflow.

    Per spec: Initiate a build requiring security review, verify specialist
    agents are invoked, and consensus voting produces final decisions.
    """

    @pytest.mark.asyncio
    async def test_security_review_invokes_security_agent(
        self,
        supervisor: MockSupervisor,
        messaging: MockMessaging,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test that security-sensitive builds invoke the security agent."""
        security_agent = agents[AgentRole.SECURITY]
        await supervisor.spawn(security_agent)

        # Simulate orchestrator requesting security review
        orchestrator_id = AgentId("orchestrator-001")
        await messaging.send(
            sender=orchestrator_id,
            receiver=security_agent.id,
            content="Review authentication implementation for vulnerabilities",
        )

        # Security agent should receive the message
        all_messages = messaging.messages
        security_messages = [m for m in all_messages if m.receiver == security_agent.id]
        assert len(security_messages) == 1
        assert "authentication" in security_messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_security_consensus_on_sensitive_change(
        self,
        supervisor: MockSupervisor,
        consensus: MockConsensus,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test consensus voting for security-sensitive changes."""
        # Spawn all agents
        for agent in agents.values():
            await supervisor.spawn(agent)

        voter_ids = [a.id for a in agents.values()]

        # Initiate vote on security-sensitive decision
        decision_id = await consensus.initiate(
            topic="Should we allow password reset via email?",
            options=["Yes with 2FA", "Yes without 2FA", "No"],
            voters=voter_ids,
        )

        # Security agent strongly votes for 2FA
        await consensus.vote(
            decision_id,
            agents[AgentRole.SECURITY].id,
            AgentRole.SECURITY,
            "Yes with 2FA",
            confidence=0.95,
            reasoning="2FA is essential for password reset security",
        )

        # Other agents agree
        await consensus.vote(
            decision_id,
            agents[AgentRole.ARCHITECT].id,
            AgentRole.ARCHITECT,
            "Yes with 2FA",
            confidence=0.8,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.IMPLEMENTER].id,
            AgentRole.IMPLEMENTER,
            "Yes with 2FA",
            confidence=0.7,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.REVIEWER].id,
            AgentRole.REVIEWER,
            "Yes with 2FA",
            confidence=0.85,
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.TESTER].id,
            AgentRole.TESTER,
            "Yes with 2FA",
            confidence=0.75,
        )

        result = await consensus.finalize(decision_id, threshold=0.8)

        # Unanimous agreement on 2FA
        assert result.final_decision == "Yes with 2FA"
        assert result.agreement_ratio > 0.9


class TestMessageDurability:
    """Tests for durable messaging (FR-017).

    Messages are stored as events and can be replayed.
    """

    @pytest.mark.asyncio
    async def test_messages_are_persisted(
        self,
        messaging: MockMessaging,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test that messages are stored persistently."""
        architect = agents[AgentRole.ARCHITECT]
        reviewer = agents[AgentRole.REVIEWER]

        await messaging.send(
            sender=architect.id,
            receiver=reviewer.id,
            content="First message",
        )
        await messaging.send(
            sender=reviewer.id,
            receiver=architect.id,
            content="Reply",
        )

        # Messages should be stored
        assert len(messaging.messages) == 2

        # Messages should have unique IDs
        ids = {m.id for m in messaging.messages}
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_thread_can_be_reconstructed(
        self,
        messaging: MockMessaging,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test that message threads can be reconstructed."""
        architect = agents[AgentRole.ARCHITECT]
        implementer = agents[AgentRole.IMPLEMENTER]

        msg1 = await messaging.send(
            sender=architect.id,
            receiver=implementer.id,
            content="Start implementation",
        )

        # Multiple replies in same thread
        for i in range(5):
            await messaging.send(
                sender=implementer.id if i % 2 == 0 else architect.id,
                receiver=architect.id if i % 2 == 0 else implementer.id,
                content=f"Progress update {i + 1}",
                thread_id=msg1.thread_id,
            )

        thread = await messaging.get_thread(msg1.thread_id)
        assert len(thread) == 6

        # Messages should maintain order via timestamps
        for i in range(len(thread) - 1):
            assert thread[i].timestamp <= thread[i + 1].timestamp


class TestAgentCoordination:
    """Tests for multi-agent coordination scenarios."""

    @pytest.mark.asyncio
    async def test_implement_review_cycle(
        self,
        supervisor: MockSupervisor,
        messaging: MockMessaging,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test implement -> review -> fix cycle."""
        implementer = agents[AgentRole.IMPLEMENTER]
        reviewer = agents[AgentRole.REVIEWER]

        await supervisor.spawn(implementer)
        await supervisor.spawn(reviewer)

        # Implementer submits code
        msg1 = await messaging.send(
            sender=implementer.id,
            receiver=reviewer.id,
            content="Code for feature X complete, ready for review",
        )

        # Reviewer provides feedback
        msg2 = await messaging.send(
            sender=reviewer.id,
            receiver=implementer.id,
            content="Found issue: missing null check on line 42",
            thread_id=msg1.thread_id,
            reply_to=msg1.id,
        )

        # Implementer fixes
        msg3 = await messaging.send(
            sender=implementer.id,
            receiver=reviewer.id,
            content="Fixed null check, please re-review",
            thread_id=msg1.thread_id,
            reply_to=msg2.id,
        )

        # Reviewer approves
        msg4 = await messaging.send(
            sender=reviewer.id,
            receiver=implementer.id,
            content="LGTM, approved",
            thread_id=msg1.thread_id,
            reply_to=msg3.id,
        )

        thread = await messaging.get_thread(msg1.thread_id)
        assert len(thread) == 4
        assert "LGTM" in thread[-1].content

    @pytest.mark.asyncio
    async def test_architecture_decision_with_all_specialists(
        self,
        supervisor: MockSupervisor,
        consensus: MockConsensus,
        messaging: MockMessaging,
        agents: dict[AgentRole, Agent],
    ) -> None:
        """Test full architecture decision flow with all specialists."""
        # Spawn all agents
        for agent in agents.values():
            await supervisor.spawn(agent)

        # Architect proposes architecture
        orchestrator_id = AgentId("orchestrator-001")
        await messaging.send(
            sender=orchestrator_id,
            receiver=None,  # Broadcast
            content="Proposing: Event-driven microservices with message queue",
        )

        # Initiate consensus
        voter_ids = [a.id for a in agents.values()]
        decision_id = await consensus.initiate(
            topic="Architecture: Event-driven microservices",
            options=["Approve", "Reject", "Needs more research"],
            voters=voter_ids,
        )

        # Each specialist votes based on their perspective
        await consensus.vote(
            decision_id,
            agents[AgentRole.ARCHITECT].id,
            AgentRole.ARCHITECT,
            "Approve",
            confidence=0.9,
            reasoning="Scalable and maintainable",
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.IMPLEMENTER].id,
            AgentRole.IMPLEMENTER,
            "Approve",
            confidence=0.7,
            reasoning="Familiar with the pattern",
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.REVIEWER].id,
            AgentRole.REVIEWER,
            "Approve",
            confidence=0.8,
            reasoning="Clean separation of concerns",
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.TESTER].id,
            AgentRole.TESTER,
            "Needs more research",
            confidence=0.6,
            reasoning="Testing distributed systems is complex",
        )
        await consensus.vote(
            decision_id,
            agents[AgentRole.SECURITY].id,
            AgentRole.SECURITY,
            "Approve",
            confidence=0.75,
            reasoning="Message queue can be secured",
        )

        result = await consensus.finalize(decision_id, threshold=0.6)

        # Architecture approved by consensus
        assert result.final_decision == "Approve"
        assert len(result.votes) == 5
