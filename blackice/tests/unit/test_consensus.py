"""Unit tests for consensus voting mechanisms.

Tests the consensus module that implements voting policies:
- Majority (>50%)
- Supermajority (>2/3)
- Unanimous (100%)
- Quorum (minimum participants)
- Weighted (by confidence/role)

Per FR-015: Voting policies for multi-agent consensus.
"""

from __future__ import annotations

from typing import Any

import pytest

from blackice.primitives.types import AgentId, AgentRole
from blackice.schemas.agent import ConsensusResult, ConsensusVote


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def voters() -> dict[AgentRole, AgentId]:
    """Create a set of voter agent IDs by role."""
    return {
        AgentRole.ARCHITECT: AgentId("architect-001"),
        AgentRole.IMPLEMENTER: AgentId("implementer-001"),
        AgentRole.REVIEWER: AgentId("reviewer-001"),
        AgentRole.TESTER: AgentId("tester-001"),
        AgentRole.SECURITY: AgentId("security-001"),
    }


# =============================================================================
# ConsensusVote Tests
# =============================================================================


class TestConsensusVote:
    """Tests for individual votes."""

    def test_vote_creation(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test creating a valid vote."""
        vote = ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="Option A",
            confidence=0.85,
            reasoning="Best fit for requirements",
        )

        assert vote.agent_id == voters[AgentRole.ARCHITECT]
        assert vote.role == AgentRole.ARCHITECT
        assert vote.decision == "Option A"
        assert vote.confidence == 0.85
        assert vote.reasoning == "Best fit for requirements"

    def test_vote_confidence_bounds(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test that confidence must be between 0 and 1."""
        # Valid confidence
        vote = ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="A",
            confidence=0.0,
        )
        assert vote.confidence == 0.0

        vote = ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="A",
            confidence=1.0,
        )
        assert vote.confidence == 1.0

        # Invalid confidence should raise
        with pytest.raises(ValueError):
            ConsensusVote(
                agent_id=voters[AgentRole.ARCHITECT],
                role=AgentRole.ARCHITECT,
                decision="A",
                confidence=1.5,
            )

        with pytest.raises(ValueError):
            ConsensusVote(
                agent_id=voters[AgentRole.ARCHITECT],
                role=AgentRole.ARCHITECT,
                decision="A",
                confidence=-0.1,
            )

    def test_vote_has_timestamp(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test that votes automatically have timestamps."""
        vote = ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="A",
            confidence=0.8,
        )

        assert vote.timestamp is not None


# =============================================================================
# ConsensusResult Tests
# =============================================================================


class TestConsensusResult:
    """Tests for consensus result computation."""

    def test_empty_consensus(self) -> None:
        """Test consensus with no votes."""
        result = ConsensusResult(topic="Test topic")

        decision = result.compute_decision()

        assert decision is None
        assert result.final_decision is None
        assert len(result.votes) == 0

    def test_add_vote(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test adding votes to a result."""
        result = ConsensusResult(topic="Test topic")

        vote = ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="A",
            confidence=0.8,
        )
        result.add_vote(vote)

        assert len(result.votes) == 1
        assert result.votes[0] == vote

    def test_unanimous_decision(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test unanimous decision (all vote same)."""
        result = ConsensusResult(topic="Framework choice")

        for role, agent_id in voters.items():
            vote = ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision="FastAPI",
                confidence=0.8,
            )
            result.add_vote(vote)

        decision = result.compute_decision(threshold=0.5)

        assert decision == "FastAPI"
        assert result.agreement_ratio == 1.0

    def test_majority_decision(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test majority decision (>50% agree)."""
        result = ConsensusResult(topic="Database choice")

        # 3 vote for PostgreSQL
        for role in [AgentRole.ARCHITECT, AgentRole.IMPLEMENTER, AgentRole.REVIEWER]:
            vote = ConsensusVote(
                agent_id=voters[role],
                role=role,
                decision="PostgreSQL",
                confidence=0.8,
            )
            result.add_vote(vote)

        # 2 vote for MongoDB
        for role in [AgentRole.TESTER, AgentRole.SECURITY]:
            vote = ConsensusVote(
                agent_id=voters[role],
                role=role,
                decision="MongoDB",
                confidence=0.8,
            )
            result.add_vote(vote)

        decision = result.compute_decision(threshold=0.5)

        assert decision == "PostgreSQL"
        # 3 * 0.8 = 2.4 for PostgreSQL, 2 * 0.8 = 1.6 for MongoDB
        # Total = 4.0, PostgreSQL ratio = 2.4/4.0 = 0.6
        assert abs(result.agreement_ratio - 0.6) < 0.01

    def test_no_majority_at_high_threshold(
        self,
        voters: dict[AgentRole, AgentId],
    ) -> None:
        """Test that decision fails at high threshold without supermajority."""
        result = ConsensusResult(topic="Caching strategy")

        # 3 vote for Redis
        for role in [AgentRole.ARCHITECT, AgentRole.IMPLEMENTER, AgentRole.REVIEWER]:
            vote = ConsensusVote(
                agent_id=voters[role],
                role=role,
                decision="Redis",
                confidence=0.8,
            )
            result.add_vote(vote)

        # 2 vote for Memcached
        for role in [AgentRole.TESTER, AgentRole.SECURITY]:
            vote = ConsensusVote(
                agent_id=voters[role],
                role=role,
                decision="Memcached",
                confidence=0.8,
            )
            result.add_vote(vote)

        # Require supermajority (67%)
        decision = result.compute_decision(threshold=0.67)

        assert decision is None
        assert result.final_decision is None
        # 3 * 0.8 = 2.4 for Redis, 2 * 0.8 = 1.6 for Memcached
        # Total = 4.0, Redis ratio = 2.4/4.0 = 0.6
        assert abs(result.agreement_ratio - 0.6) < 0.01

    def test_weighted_voting_by_confidence(
        self,
        voters: dict[AgentRole, AgentId],
    ) -> None:
        """Test that votes are weighted by confidence."""
        result = ConsensusResult(topic="Authentication")

        # Security expert votes with high confidence
        result.add_vote(ConsensusVote(
            agent_id=voters[AgentRole.SECURITY],
            role=AgentRole.SECURITY,
            decision="OAuth2",
            confidence=0.95,
        ))

        # Others vote JWT with low confidence
        for role in [AgentRole.ARCHITECT, AgentRole.IMPLEMENTER,
                     AgentRole.REVIEWER, AgentRole.TESTER]:
            result.add_vote(ConsensusVote(
                agent_id=voters[role],
                role=role,
                decision="JWT",
                confidence=0.3,
            ))

        decision = result.compute_decision(threshold=0.4)

        # Despite 4 vs 1 count, OAuth2 wins due to confidence weighting
        # OAuth2: 0.95, JWT: 4 * 0.3 = 1.2
        # Total: 2.15, OAuth2 ratio: 0.95/2.15 = 44%
        # JWT ratio: 1.2/2.15 = 56%
        # JWT should win with weighted voting
        assert decision == "JWT"

    def test_confidence_weight_ties(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test behavior when weighted votes are tied."""
        result = ConsensusResult(topic="API style")

        # Equal weighted votes
        result.add_vote(ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="REST",
            confidence=0.5,
        ))
        result.add_vote(ConsensusVote(
            agent_id=voters[AgentRole.IMPLEMENTER],
            role=AgentRole.IMPLEMENTER,
            decision="GraphQL",
            confidence=0.5,
        ))

        decision = result.compute_decision(threshold=0.5)

        # Tie - one should win (depends on implementation, max selects first)
        assert decision in ["REST", "GraphQL"]
        assert result.agreement_ratio == 0.5

    def test_zero_confidence_votes_ignored(
        self,
        voters: dict[AgentRole, AgentId],
    ) -> None:
        """Test that zero-confidence votes have no weight."""
        result = ConsensusResult(topic="Test runner")

        # One confident vote
        result.add_vote(ConsensusVote(
            agent_id=voters[AgentRole.TESTER],
            role=AgentRole.TESTER,
            decision="pytest",
            confidence=1.0,
        ))

        # Rest have zero confidence
        for role in [AgentRole.ARCHITECT, AgentRole.IMPLEMENTER,
                     AgentRole.REVIEWER, AgentRole.SECURITY]:
            result.add_vote(ConsensusVote(
                agent_id=voters[role],
                role=role,
                decision="unittest",
                confidence=0.0,
            ))

        decision = result.compute_decision(threshold=0.5)

        # pytest wins because it has all the weight
        assert decision == "pytest"
        assert result.agreement_ratio == 1.0


class TestConsensusThresholds:
    """Tests for different consensus threshold policies."""

    def test_simple_majority_threshold(
        self,
        voters: dict[AgentRole, AgentId],
    ) -> None:
        """Test simple majority (50% + 1)."""
        result = ConsensusResult(topic="Test")

        # Add 3 votes for A, 2 for B
        for i, (role, agent_id) in enumerate(voters.items()):
            decision = "A" if i < 3 else "B"
            result.add_vote(ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision=decision,
                confidence=0.8,
            ))

        decision = result.compute_decision(threshold=0.5)
        assert decision == "A"

    def test_supermajority_threshold(
        self,
        voters: dict[AgentRole, AgentId],
    ) -> None:
        """Test supermajority (2/3)."""
        result = ConsensusResult(topic="Test")

        # Add 4 votes for A, 1 for B (80%)
        for i, (role, agent_id) in enumerate(voters.items()):
            decision = "A" if i < 4 else "B"
            result.add_vote(ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision=decision,
                confidence=0.8,
            ))

        # Should pass supermajority
        decision = result.compute_decision(threshold=0.67)
        assert decision == "A"

        # Same votes but 3 vs 2 (60%) should fail supermajority
        result2 = ConsensusResult(topic="Test2")
        for i, (role, agent_id) in enumerate(voters.items()):
            decision = "A" if i < 3 else "B"
            result2.add_vote(ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision=decision,
                confidence=0.8,
            ))

        decision2 = result2.compute_decision(threshold=0.67)
        assert decision2 is None

    def test_unanimous_threshold(
        self,
        voters: dict[AgentRole, AgentId],
    ) -> None:
        """Test unanimous requirement (100%)."""
        result = ConsensusResult(topic="Critical security decision")

        # All vote same
        for role, agent_id in voters.items():
            result.add_vote(ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision="Approve",
                confidence=0.9,
            ))

        decision = result.compute_decision(threshold=1.0)
        assert decision == "Approve"

        # One dissenting vote fails unanimous
        result2 = ConsensusResult(topic="Test")
        roles = list(voters.items())
        for i, (role, agent_id) in enumerate(roles):
            decision = "Approve" if i < 4 else "Reject"
            result2.add_vote(ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision=decision,
                confidence=0.9,
            ))

        decision2 = result2.compute_decision(threshold=1.0)
        assert decision2 is None


class TestConsensusMetadata:
    """Tests for consensus metadata and tracking."""

    def test_result_has_topic(self) -> None:
        """Test that result tracks the topic."""
        result = ConsensusResult(topic="API versioning strategy")
        assert result.topic == "API versioning strategy"

    def test_result_has_timestamp(self) -> None:
        """Test that result has a timestamp."""
        result = ConsensusResult(topic="Test")
        assert result.timestamp is not None

    def test_vote_count_tracking(
        self,
        voters: dict[AgentRole, AgentId],
    ) -> None:
        """Test tracking of vote counts."""
        result = ConsensusResult(topic="Test")

        for i, (role, agent_id) in enumerate(voters.items()):
            result.add_vote(ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision="A" if i % 2 == 0 else "B",
                confidence=0.8,
            ))

        assert len(result.votes) == 5

        # Can count by decision
        a_votes = [v for v in result.votes if v.decision == "A"]
        b_votes = [v for v in result.votes if v.decision == "B"]
        assert len(a_votes) == 3
        assert len(b_votes) == 2


class TestEdgeCases:
    """Tests for edge cases in consensus voting."""

    def test_single_voter(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test consensus with single voter."""
        result = ConsensusResult(topic="Solo decision")

        result.add_vote(ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="Approve",
            confidence=0.9,
        ))

        decision = result.compute_decision(threshold=0.5)
        assert decision == "Approve"
        assert result.agreement_ratio == 1.0

    def test_all_zero_confidence(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test when all votes have zero confidence."""
        result = ConsensusResult(topic="Uncertain")

        for role, agent_id in voters.items():
            result.add_vote(ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision="A",
                confidence=0.0,
            ))

        decision = result.compute_decision()

        # No decision when all confidence is zero (total weight = 0)
        assert decision is None

    def test_many_options(self, voters: dict[AgentRole, AgentId]) -> None:
        """Test with many different options."""
        result = ConsensusResult(topic="Many choices")

        options = ["A", "B", "C", "D", "E"]
        for (role, agent_id), option in zip(voters.items(), options):
            result.add_vote(ConsensusVote(
                agent_id=agent_id,
                role=role,
                decision=option,
                confidence=0.8,
            ))

        # No majority possible with 5 different choices
        decision = result.compute_decision(threshold=0.5)
        assert decision is None

    def test_duplicate_votes_counted(
        self,
        voters: dict[AgentRole, AgentId],
    ) -> None:
        """Test that duplicate votes from same agent are counted."""
        result = ConsensusResult(topic="Test")

        # Same agent votes twice (shouldn't happen but test behavior)
        result.add_vote(ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="A",
            confidence=0.8,
        ))
        result.add_vote(ConsensusVote(
            agent_id=voters[AgentRole.ARCHITECT],
            role=AgentRole.ARCHITECT,
            decision="B",
            confidence=0.8,
        ))

        # Both votes are counted
        assert len(result.votes) == 2
