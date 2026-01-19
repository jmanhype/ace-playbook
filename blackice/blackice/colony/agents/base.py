"""Base agent class for BLACKICE 3.0 colony.

Provides common functionality for all specialist agents in the
multi-agent colony that executes the vision-to-software pipeline.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import structlog

from blackice.adapters.models.base import Message
from blackice.primitives.types import AgentId, AgentRole, TaskId, Timestamp
from blackice.schemas.agent import (
    Agent,
    AgentCapabilities,
    AgentExecution,
    ConsensusVote,
    create_agent,
)

if TYPE_CHECKING:
    from blackice.adapters.models.base import ModelProvider
    from blackice.adapters.memory.base import MemoryProvider


logger = structlog.get_logger(__name__)


class BaseColonyAgent(ABC):
    """Base class for all colony agents.

    Provides common functionality for:
    - Task execution with model providers
    - Memory integration for learning
    - Consensus voting participation
    - Execution tracking and metrics
    """

    def __init__(
        self,
        agent: Agent,
        model_provider: ModelProvider | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        """Initialize the colony agent.

        Args:
            agent: The underlying Agent schema
            model_provider: LLM provider for inference
            memory_provider: Memory provider for learning
        """
        self.agent = agent
        self.model_provider = model_provider
        self.memory_provider = memory_provider
        self._current_execution: AgentExecution | None = None
        self._lock = asyncio.Lock()

    @property
    def id(self) -> AgentId:
        """Get the agent ID."""
        return self.agent.id

    @property
    def role(self) -> AgentRole:
        """Get the agent's role."""
        return self.agent.role

    @property
    def active(self) -> bool:
        """Check if agent is active."""
        return self.agent.active

    @active.setter
    def active(self, value: bool) -> None:
        """Set agent active state."""
        self.agent.active = value

    @property
    def capabilities(self) -> AgentCapabilities:
        """Get agent capabilities."""
        return self.agent.capabilities

    @abstractmethod
    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a task assigned to this agent.

        Args:
            task: Task definition with description, requirements, etc.

        Returns:
            Task result with output, artifacts, etc.
        """
        ...

    async def think(
        self,
        prompt: str,
        *,
        context: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Use the model provider to think about a problem.

        Args:
            prompt: The prompt to send to the model
            context: Additional context to include
            max_tokens: Maximum tokens for the response

        Returns:
            The model's response content
        """
        if self.model_provider is None:
            raise RuntimeError("No model provider configured for this agent")

        messages = []

        # Add system prompt
        messages.append(Message(
            role="system",
            content=self.agent.system_prompt,
        ))

        # Add context if provided
        if context:
            messages.append(Message(
                role="user",
                content=f"Context:\n{context}",
            ))

        # Add the main prompt
        messages.append(Message(
            role="user",
            content=prompt,
        ))

        result = await self.model_provider.chat(
            messages,
            max_tokens=max_tokens,
            temperature=self.agent.temperature,
        )

        return result.content

    async def vote(
        self,
        topic: str,
        options: list[str],
        context: str | None = None,
    ) -> ConsensusVote:
        """Cast a vote on a consensus decision.

        Args:
            topic: The topic being voted on
            options: Available options to choose from
            context: Additional context for the decision

        Returns:
            ConsensusVote with the agent's decision
        """
        # Format the prompt for voting
        options_str = "\n".join(f"- {opt}" for opt in options)
        prompt = f"""You are voting on: {topic}

Available options:
{options_str}

Analyze each option and choose the best one for this situation.
Respond with:
1. Your decision (exactly one of the options)
2. Your confidence level (0.0 to 1.0)
3. Brief reasoning (1-2 sentences)

Format your response as:
DECISION: <your choice>
CONFIDENCE: <0.0-1.0>
REASONING: <your explanation>"""

        response = await self.think(prompt, context=context)

        # Parse the response
        decision = options[0]  # Default
        confidence = 0.5
        reasoning = ""

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return ConsensusVote(
            agent_id=self.id,
            role=self.role,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
        )

    async def start_execution(self, task_id: TaskId) -> AgentExecution:
        """Start tracking execution of a task.

        Args:
            task_id: The task being executed

        Returns:
            AgentExecution tracking object
        """
        async with self._lock:
            execution = AgentExecution(
                agent_id=self.id,
                task_id=task_id,
                role=self.role,
            )
            self._current_execution = execution
            logger.info(
                "agent_started_execution",
                agent_id=str(self.id),
                task_id=str(task_id),
                role=self.role.value,
            )
            return execution

    async def complete_execution(
        self,
        success: bool,
        output: str | None = None,
        error: str | None = None,
    ) -> AgentExecution | None:
        """Complete the current execution.

        Args:
            success: Whether execution succeeded
            output: Execution output/result
            error: Error message if failed

        Returns:
            The completed AgentExecution or None
        """
        async with self._lock:
            if self._current_execution is None:
                return None

            self._current_execution.complete(success, output, error)
            execution = self._current_execution
            self._current_execution = None

            logger.info(
                "agent_completed_execution",
                agent_id=str(self.id),
                task_id=str(execution.task_id),
                success=success,
                duration_seconds=execution.duration_seconds,
            )

            return execution

    async def remember(
        self,
        content: str,
        *,
        memory_type: str = "pattern",
        tags: list[str] | None = None,
    ) -> str | None:
        """Store something in memory for future reference.

        Args:
            content: Content to remember
            memory_type: Type of memory (pattern, error, decision, etc.)
            tags: Optional tags for categorization

        Returns:
            Memory entry ID if stored, None otherwise
        """
        if self.memory_provider is None:
            return None

        from blackice.adapters.memory.base import MemoryEntry, MemoryType

        try:
            entry = MemoryEntry(
                id=f"mem-{self.id}-{Timestamp.now().value.timestamp():.0f}",
                memory_type=MemoryType(memory_type),
                content=content,
                source=f"agent:{self.id}",
                tags=tags or [],
            )
            return await self.memory_provider.put(entry)
        except Exception as e:
            logger.warning(
                "failed_to_store_memory",
                agent_id=str(self.id),
                error=str(e),
            )
            return None

    async def recall(
        self,
        query: str,
        limit: int = 5,
    ) -> list[str]:
        """Recall relevant memories.

        Args:
            query: What to search for
            limit: Maximum memories to return

        Returns:
            List of relevant memory contents
        """
        if self.memory_provider is None:
            return []

        try:
            results = await self.memory_provider.search(query, limit=limit)
            return [r.entry.content for r in results]
        except Exception as e:
            logger.warning(
                "failed_to_recall_memory",
                agent_id=str(self.id),
                error=str(e),
            )
            return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} role={self.role.value}>"


def create_colony_agent(
    agent_class: type[BaseColonyAgent],
    role: AgentRole,
    agent_id: AgentId | None = None,
    model_provider: ModelProvider | None = None,
    memory_provider: MemoryProvider | None = None,
    **kwargs: Any,
) -> BaseColonyAgent:
    """Factory function to create a colony agent.

    Args:
        agent_class: The specific agent class to instantiate
        role: The agent's role
        agent_id: Optional custom ID
        model_provider: LLM provider
        memory_provider: Memory provider
        **kwargs: Additional agent configuration

    Returns:
        Configured colony agent instance
    """
    agent = create_agent(role, agent_id, **kwargs)
    return agent_class(agent, model_provider, memory_provider)
