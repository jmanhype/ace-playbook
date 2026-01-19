"""Agent schema for BLACKICE 3.0.

Agents are specialist LLM instances with specific roles in the
multi-agent colony that executes the vision-to-software pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from blackice.primitives.types import (
    AgentId,
    AgentRole,
    TaskId,
    Timestamp,
)


class AgentCapabilities(BaseModel):
    """Capabilities that an agent can provide."""

    can_write_code: bool = Field(default=True)
    can_execute_commands: bool = Field(default=True)
    can_read_files: bool = Field(default=True)
    can_write_files: bool = Field(default=True)
    can_search_web: bool = Field(default=False)
    can_use_tools: bool = Field(default=True)

    # Specialized capabilities
    languages: list[str] = Field(default_factory=lambda: ["python", "javascript", "typescript"])
    frameworks: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class Agent(BaseModel):
    """A specialist agent in the BLACKICE colony.

    Agents are LLM instances configured for specific roles like
    architecture, implementation, review, testing, or documentation.

    Attributes:
        id: Unique identifier for this agent
        role: The agent's specialist role
        model_provider: LLM provider to use
        model_name: Specific model to use
        system_prompt: Role-specific system prompt
        capabilities: What this agent can do
        temperature: Model temperature for generation
    """

    id: AgentId
    role: AgentRole
    model_provider: str = Field(default="claude")
    model_name: str | None = Field(default=None)

    # Configuration
    system_prompt: str = Field(..., min_length=10, max_length=10_000)
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=100, le=100_000)

    # State
    active: bool = Field(default=False)
    created_at: Timestamp = Field(default_factory=Timestamp.now)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = False


class AgentExecution(BaseModel):
    """Record of an agent's execution of a task.

    Tracks the agent's work on a specific task including
    tokens used, duration, and outcomes.
    """

    agent_id: AgentId
    task_id: TaskId
    role: AgentRole

    # Timing
    started_at: Timestamp = Field(default_factory=Timestamp.now)
    completed_at: Timestamp | None = Field(default=None)

    # Token usage
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    # Cost tracking (in USD)
    cost_usd: float = Field(default=0.0, ge=0.0)

    # Results
    success: bool = Field(default=False)
    output: str | None = Field(default=None)
    error: str | None = Field(default=None)

    # Tool usage
    tool_calls: int = Field(default=0, ge=0)
    commands_executed: int = Field(default=0, ge=0)
    files_modified: list[str] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate execution duration in seconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at.value - self.started_at.value).total_seconds()

    def complete(self, success: bool, output: str | None = None, error: str | None = None) -> None:
        """Mark execution as complete."""
        self.completed_at = Timestamp.now()
        self.success = success
        self.output = output
        self.error = error


class ConsensusVote(BaseModel):
    """A vote from an agent in a consensus decision."""

    agent_id: AgentId
    role: AgentRole
    decision: str  # The agent's choice
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=1000)
    timestamp: Timestamp = Field(default_factory=Timestamp.now)


class ConsensusResult(BaseModel):
    """Result of a multi-agent consensus process.

    When consensus voting is enabled, multiple agents vote
    on important decisions (architecture, approaches, etc.).
    """

    topic: str
    votes: list[ConsensusVote] = Field(default_factory=list)
    final_decision: str | None = Field(default=None)
    agreement_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: Timestamp = Field(default_factory=Timestamp.now)

    def add_vote(self, vote: ConsensusVote) -> None:
        """Add a vote to the consensus."""
        self.votes.append(vote)

    def compute_decision(self, threshold: float = 0.6) -> str | None:
        """Compute the final decision based on votes.

        Args:
            threshold: Minimum agreement ratio required

        Returns:
            The winning decision if threshold met, None otherwise
        """
        if not self.votes:
            return None

        # Count weighted votes (by confidence)
        vote_scores: dict[str, float] = {}
        total_weight = 0.0

        for vote in self.votes:
            vote_scores[vote.decision] = vote_scores.get(vote.decision, 0.0) + vote.confidence
            total_weight += vote.confidence

        if total_weight == 0:
            return None

        # Find winner
        winner = max(vote_scores.items(), key=lambda x: x[1])
        self.agreement_ratio = winner[1] / total_weight

        if self.agreement_ratio >= threshold:
            self.final_decision = winner[0]
            return self.final_decision

        return None


# Standard system prompts for each role
AGENT_PROMPTS: dict[AgentRole, str] = {
    AgentRole.ARCHITECT: """You are a software architect agent. Your role is to:
- Analyze requirements and design system architecture
- Make technology choices and document decisions
- Define interfaces and component boundaries
- Ensure scalability, security, and maintainability
Focus on clear, pragmatic designs that balance complexity with functionality.""",
    AgentRole.IMPLEMENTER: """You are an implementation agent. Your role is to:
- Write clean, well-structured code following best practices
- Implement features according to the architectural design
- Handle edge cases and error conditions
- Write self-documenting code with appropriate comments
Focus on correctness, readability, and maintainability.""",
    AgentRole.REVIEWER: """You are a code review agent. Your role is to:
- Review code for correctness, security, and best practices
- Identify potential bugs, vulnerabilities, and improvements
- Ensure code follows project conventions
- Provide constructive feedback with specific suggestions
Focus on catching issues early and improving code quality.""",
    AgentRole.TESTER: """You are a testing agent. Your role is to:
- Write comprehensive unit and integration tests
- Identify edge cases and boundary conditions
- Ensure test coverage meets requirements
- Validate that code behaves correctly
Focus on thorough testing that catches real bugs.""",
    AgentRole.DOCUMENTER: """You are a documentation agent. Your role is to:
- Write clear, comprehensive documentation
- Document APIs, interfaces, and usage patterns
- Create examples and tutorials
- Maintain consistency in documentation style
Focus on helping users understand and use the software effectively.""",
    AgentRole.SECURITY: """You are a security agent. Your role is to:
- Review code for security vulnerabilities
- Ensure secure coding practices are followed
- Validate input handling and authentication
- Check for OWASP top 10 vulnerabilities
Focus on protecting the system from security threats.""",
    AgentRole.ORCHESTRATOR: """You are the orchestrator agent. Your role is to:
- Coordinate work between other agents
- Manage task dependencies and ordering
- Handle failures and recovery
- Ensure overall progress toward completion
Focus on efficient execution and successful completion.""",
}


def create_agent(role: AgentRole, agent_id: AgentId | None = None, **kwargs: Any) -> Agent:
    """Factory function to create a properly configured agent.

    Args:
        role: The agent's specialist role
        agent_id: Optional custom ID (generated if not provided)
        **kwargs: Additional agent configuration

    Returns:
        Configured Agent instance
    """
    if agent_id is None:
        agent_id = AgentId(f"{role.value}-{Timestamp.now().value.timestamp():.0f}")

    return Agent(
        id=agent_id,
        role=role,
        system_prompt=kwargs.pop("system_prompt", AGENT_PROMPTS[role]),
        **kwargs,
    )
