"""Colony module for BLACKICE 3.0.

The colony is the multi-agent system that executes the vision-to-software
pipeline through coordinated specialist agents.

Components:
- Agents: Specialist LLM instances (Planner, Implementer, Reviewer, Tester, Security)
- Supervisor: Manages agent lifecycle and task assignment
- Consensus: Multi-agent voting and decision making
- Messaging: Durable message passing between agents
- Registry: Agent registration and discovery
- Reservation: Resource reservation to prevent conflicts
"""

from blackice.colony.agents import (
    BaseColonyAgent,
    ImplementerAgent,
    PlannerAgent,
    ReviewerAgent,
    SecurityAgent,
    TesterAgent,
    create_colony_agent,
    create_implementer,
    create_planner,
    create_reviewer,
    create_security,
    create_tester,
)
from blackice.colony.consensus import (
    Consensus,
    ConsensusSession,
    VotingConfig,
    VotingPolicy,
)
from blackice.colony.messaging import (
    Message,
    MessagePriority,
    MessageStatus,
    MessageStore,
    Messaging,
    Thread,
)
from blackice.colony.registry import (
    AgentRegistration,
    Registry,
    RegistryConfig,
)
from blackice.colony.reservation import (
    Reservation,
    ReservationConfig,
    ReservationConflictError,
    ReservationStatus,
    ReservationSystem,
    ReservationType,
)
from blackice.colony.supervisor import (
    AgentState,
    Supervisor,
    SupervisorConfig,
)

__all__ = [
    # Agents
    "BaseColonyAgent",
    "PlannerAgent",
    "ImplementerAgent",
    "ReviewerAgent",
    "TesterAgent",
    "SecurityAgent",
    "create_colony_agent",
    "create_planner",
    "create_implementer",
    "create_reviewer",
    "create_tester",
    "create_security",
    # Supervisor
    "Supervisor",
    "SupervisorConfig",
    "AgentState",
    # Consensus
    "Consensus",
    "ConsensusSession",
    "VotingConfig",
    "VotingPolicy",
    # Messaging
    "Messaging",
    "Message",
    "MessagePriority",
    "MessageStatus",
    "MessageStore",
    "Thread",
    # Registry
    "Registry",
    "RegistryConfig",
    "AgentRegistration",
    # Reservation
    "ReservationSystem",
    "ReservationConfig",
    "Reservation",
    "ReservationType",
    "ReservationStatus",
    "ReservationConflictError",
]
