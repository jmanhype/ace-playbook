"""Consensus module for BLACKICE 3.0 colony.

Implements multi-agent voting mechanisms for collaborative decisions:
- Majority voting (>50%)
- Supermajority voting (>2/3)
- Unanimous voting (100%)
- Weighted voting by confidence/role
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from blackice.primitives.types import AgentId, AgentRole
from blackice.schemas.agent import ConsensusResult, ConsensusVote

if TYPE_CHECKING:
    from blackice.colony.agents.base import BaseColonyAgent


logger = structlog.get_logger(__name__)


class VotingPolicy(str, Enum):
    """Voting policy types."""

    MAJORITY = "majority"  # >50% agreement
    SUPERMAJORITY = "supermajority"  # >2/3 agreement
    UNANIMOUS = "unanimous"  # 100% agreement
    QUORUM = "quorum"  # Minimum participants
    WEIGHTED = "weighted"  # By confidence/role


@dataclass
class VotingConfig:
    """Configuration for a voting session."""

    policy: VotingPolicy = VotingPolicy.MAJORITY
    threshold: float = 0.5  # Minimum agreement ratio
    min_voters: int = 1  # Minimum required voters (quorum)
    timeout: float = 60.0  # Voting timeout in seconds
    role_weights: dict[AgentRole, float] = field(default_factory=dict)


class ConsensusSession:
    """A single consensus voting session.

    Manages the lifecycle of a voting round:
    - Collecting votes from agents
    - Computing results based on policy
    - Tracking voting history
    """

    def __init__(
        self,
        session_id: str,
        topic: str,
        options: list[str],
        config: VotingConfig | None = None,
    ) -> None:
        """Initialize a consensus session.

        Args:
            session_id: Unique session identifier
            topic: The topic being voted on
            options: Available options to vote for
            config: Voting configuration
        """
        self.session_id = session_id
        self.topic = topic
        self.options = options
        self.config = config or VotingConfig()
        self.result = ConsensusResult(topic=topic)
        self.started_at = datetime.now(timezone.utc)
        self.completed_at: datetime | None = None
        self._voters: set[AgentId] = set()
        self._lock = asyncio.Lock()

    @property
    def is_complete(self) -> bool:
        """Check if voting is complete."""
        return self.completed_at is not None

    @property
    def vote_count(self) -> int:
        """Get the number of votes cast."""
        return len(self.result.votes)

    async def add_vote(self, vote: ConsensusVote) -> bool:
        """Add a vote to the session.

        Args:
            vote: The vote to add

        Returns:
            True if vote was added, False if already voted
        """
        async with self._lock:
            if vote.agent_id in self._voters:
                logger.warning(
                    "duplicate_vote_rejected",
                    session_id=self.session_id,
                    agent_id=str(vote.agent_id),
                )
                return False

            if vote.decision not in self.options:
                logger.warning(
                    "invalid_vote_option",
                    session_id=self.session_id,
                    decision=vote.decision,
                    valid_options=self.options,
                )
                return False

            self._voters.add(vote.agent_id)
            self.result.add_vote(vote)

            logger.info(
                "vote_added",
                session_id=self.session_id,
                agent_id=str(vote.agent_id),
                decision=vote.decision,
                confidence=vote.confidence,
            )

            return True

    async def compute_result(self) -> str | None:
        """Compute the consensus result.

        Returns:
            The winning decision if consensus reached, None otherwise
        """
        async with self._lock:
            # Check quorum
            if len(self.result.votes) < self.config.min_voters:
                logger.warning(
                    "quorum_not_met",
                    session_id=self.session_id,
                    votes=len(self.result.votes),
                    required=self.config.min_voters,
                )
                return None

            # Apply role weights if configured
            if self.config.role_weights:
                self._apply_role_weights()

            # Determine threshold based on policy
            threshold = self._get_threshold()

            # Compute decision
            decision = self.result.compute_decision(threshold)

            self.completed_at = datetime.now(timezone.utc)

            if decision:
                logger.info(
                    "consensus_reached",
                    session_id=self.session_id,
                    decision=decision,
                    agreement_ratio=self.result.agreement_ratio,
                )
            else:
                logger.info(
                    "consensus_not_reached",
                    session_id=self.session_id,
                    agreement_ratio=self.result.agreement_ratio,
                    threshold=threshold,
                )

            return decision

    def _apply_role_weights(self) -> None:
        """Apply role-based weights to votes."""
        for vote in self.result.votes:
            if vote.role in self.config.role_weights:
                # Multiply confidence by role weight
                weight = self.config.role_weights[vote.role]
                vote.confidence *= weight

    def _get_threshold(self) -> float:
        """Get the threshold based on voting policy."""
        if self.config.policy == VotingPolicy.MAJORITY:
            return 0.5
        elif self.config.policy == VotingPolicy.SUPERMAJORITY:
            return 0.67
        elif self.config.policy == VotingPolicy.UNANIMOUS:
            return 1.0
        else:
            return self.config.threshold


class Consensus:
    """Multi-agent consensus system.

    Coordinates voting across multiple agents:
    - Creating voting sessions
    - Collecting votes with timeout
    - Computing consensus results
    - Maintaining voting history
    """

    def __init__(self) -> None:
        """Initialize the consensus system."""
        self._sessions: dict[str, ConsensusSession] = {}
        self._session_counter = 0
        self._lock = asyncio.Lock()

    async def propose(
        self,
        topic: str,
        options: list[str],
        config: VotingConfig | None = None,
    ) -> ConsensusSession:
        """Create a new voting session for a proposal.

        Args:
            topic: The topic to vote on
            options: Available options
            config: Voting configuration

        Returns:
            The created ConsensusSession
        """
        async with self._lock:
            self._session_counter += 1
            session_id = f"session-{self._session_counter}"

            session = ConsensusSession(
                session_id=session_id,
                topic=topic,
                options=options,
                config=config,
            )

            self._sessions[session_id] = session

            logger.info(
                "voting_session_created",
                session_id=session_id,
                topic=topic,
                options=options,
            )

            return session

    async def add_vote(
        self,
        session_id: str,
        vote: ConsensusVote,
    ) -> bool:
        """Add a vote to a session.

        Args:
            session_id: The session to vote in
            vote: The vote to add

        Returns:
            True if vote was added

        Raises:
            ValueError: If session not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        if session.is_complete:
            raise ValueError(f"Session {session_id} is already complete")

        return await session.add_vote(vote)

    async def compute_decision(
        self,
        session_id: str,
    ) -> str | None:
        """Compute the consensus decision for a session.

        Args:
            session_id: The session to compute

        Returns:
            The winning decision if consensus reached

        Raises:
            ValueError: If session not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        return await session.compute_result()

    async def vote_with_agents(
        self,
        topic: str,
        options: list[str],
        agents: list[BaseColonyAgent],
        context: str | None = None,
        config: VotingConfig | None = None,
    ) -> ConsensusResult:
        """Run a complete voting session with agents.

        Args:
            topic: The topic to vote on
            options: Available options
            agents: Agents to vote
            context: Optional context for voting
            config: Voting configuration

        Returns:
            The consensus result
        """
        config = config or VotingConfig()
        session = await self.propose(topic, options, config)

        # Collect votes with timeout
        async def collect_vote(agent: BaseColonyAgent) -> ConsensusVote | None:
            try:
                vote = await asyncio.wait_for(
                    agent.vote(topic, options, context),
                    timeout=config.timeout,
                )
                await session.add_vote(vote)
                return vote
            except asyncio.TimeoutError:
                logger.warning(
                    "vote_timeout",
                    agent_id=str(agent.id),
                    session_id=session.session_id,
                )
                return None
            except Exception as e:
                logger.error(
                    "vote_error",
                    agent_id=str(agent.id),
                    error=str(e),
                )
                return None

        # Collect all votes concurrently
        await asyncio.gather(*[collect_vote(agent) for agent in agents])

        # Compute final result
        await session.compute_result()

        return session.result

    async def get_session(self, session_id: str) -> ConsensusSession | None:
        """Get a session by ID.

        Args:
            session_id: The session ID

        Returns:
            The session if found
        """
        return self._sessions.get(session_id)

    async def get_active_sessions(self) -> list[ConsensusSession]:
        """Get all active (incomplete) sessions.

        Returns:
            List of active sessions
        """
        return [s for s in self._sessions.values() if not s.is_complete]

    async def get_history(self, limit: int = 10) -> list[ConsensusResult]:
        """Get recent voting history.

        Args:
            limit: Maximum results to return

        Returns:
            List of recent results
        """
        completed = [
            s.result
            for s in self._sessions.values()
            if s.is_complete
        ]
        # Sort by completion time (most recent first)
        completed.sort(
            key=lambda r: r.timestamp.value if hasattr(r, 'timestamp') else datetime.min,
            reverse=True,
        )
        return completed[:limit]
