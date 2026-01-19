"""Supervisor for BLACKICE 3.0 colony.

The Supervisor manages agent lifecycle and coordination:
- Spawning and terminating agents
- Task assignment and monitoring
- Agent health checks
- Work distribution
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from blackice.primitives.types import AgentId, AgentRole, TaskId
from blackice.schemas.agent import Agent, AgentExecution

if TYPE_CHECKING:
    from blackice.colony.agents.base import BaseColonyAgent


logger = structlog.get_logger(__name__)


@dataclass
class SupervisorConfig:
    """Configuration for the Supervisor."""

    max_agents: int = 10
    health_check_interval: float = 30.0
    task_timeout: float = 300.0
    max_retries: int = 3


@dataclass
class AgentState:
    """State of a managed agent."""

    agent: BaseColonyAgent
    spawned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_health_check: datetime | None = None
    current_task: TaskId | None = None
    task_count: int = 0
    error_count: int = 0


class Supervisor:
    """Supervisor manages the agent colony lifecycle.

    Responsibilities:
    - Spawn agents for specific roles
    - Assign tasks to appropriate agents
    - Monitor agent health
    - Handle agent failures
    - Terminate agents when done
    """

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        """Initialize the Supervisor.

        Args:
            config: Supervisor configuration
        """
        self.config = config or SupervisorConfig()
        self._agents: dict[AgentId, AgentState] = {}
        self._executions: list[AgentExecution] = []
        self._lock = asyncio.Lock()
        self._health_task: asyncio.Task | None = None
        self._running = False

    @property
    def agents(self) -> dict[AgentId, Agent]:
        """Get all managed agents."""
        return {aid: state.agent.agent for aid, state in self._agents.items()}

    @property
    def executions(self) -> list[AgentExecution]:
        """Get all recorded executions."""
        return self._executions.copy()

    @property
    def active_agents(self) -> list[AgentId]:
        """Get IDs of active agents."""
        return [aid for aid, state in self._agents.items() if state.agent.active]

    async def start(self) -> None:
        """Start the supervisor and health monitoring."""
        if self._running:
            return

        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info("supervisor_started", max_agents=self.config.max_agents)

    async def stop(self) -> None:
        """Stop the supervisor and terminate all agents."""
        self._running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        # Terminate all agents
        for agent_id in list(self._agents.keys()):
            await self.terminate(agent_id)

        logger.info("supervisor_stopped")

    async def spawn(self, agent: BaseColonyAgent) -> AgentId:
        """Spawn a new agent in the colony.

        Args:
            agent: The agent to spawn

        Returns:
            The agent's ID

        Raises:
            RuntimeError: If max agents reached
        """
        async with self._lock:
            if len(self._agents) >= self.config.max_agents:
                raise RuntimeError(
                    f"Maximum agents ({self.config.max_agents}) reached"
                )

            agent.active = True
            self._agents[agent.id] = AgentState(agent=agent)

            logger.info(
                "agent_spawned",
                agent_id=str(agent.id),
                role=agent.role.value,
                total_agents=len(self._agents),
            )

            return agent.id

    async def terminate(self, agent_id: AgentId) -> bool:
        """Terminate an agent.

        Args:
            agent_id: ID of the agent to terminate

        Returns:
            True if terminated, False if not found
        """
        async with self._lock:
            state = self._agents.pop(agent_id, None)
            if state is None:
                return False

            state.agent.active = False

            logger.info(
                "agent_terminated",
                agent_id=str(agent_id),
                role=state.agent.role.value,
                task_count=state.task_count,
                error_count=state.error_count,
            )

            return True

    async def get_agent(self, agent_id: AgentId) -> BaseColonyAgent | None:
        """Get an agent by ID.

        Args:
            agent_id: ID of the agent

        Returns:
            The agent if found, None otherwise
        """
        state = self._agents.get(agent_id)
        return state.agent if state else None

    async def get_agents_by_role(self, role: AgentRole) -> list[BaseColonyAgent]:
        """Get all agents with a specific role.

        Args:
            role: The role to filter by

        Returns:
            List of agents with that role
        """
        return [
            state.agent
            for state in self._agents.values()
            if state.agent.role == role and state.agent.active
        ]

    async def assign_task(
        self,
        agent_id: AgentId,
        task: dict[str, Any],
    ) -> AgentExecution:
        """Assign a task to an agent.

        Args:
            agent_id: ID of the agent
            task: Task definition

        Returns:
            AgentExecution tracking object

        Raises:
            ValueError: If agent not found or not active
        """
        state = self._agents.get(agent_id)
        if state is None:
            raise ValueError(f"Agent {agent_id} not found")

        if not state.agent.active:
            raise ValueError(f"Agent {agent_id} is not active")

        task_id = TaskId(task.get("id", f"task-{len(self._executions) + 1}"))

        async with self._lock:
            state.current_task = task_id
            state.task_count += 1

        try:
            # Execute the task
            result = await asyncio.wait_for(
                state.agent.execute_task(task),
                timeout=self.config.task_timeout,
            )

            execution = AgentExecution(
                agent_id=agent_id,
                task_id=task_id,
                role=state.agent.role,
                success=True,
                output=str(result),
            )

        except asyncio.TimeoutError:
            execution = AgentExecution(
                agent_id=agent_id,
                task_id=task_id,
                role=state.agent.role,
                success=False,
                error=f"Task timed out after {self.config.task_timeout}s",
            )
            async with self._lock:
                state.error_count += 1

        except Exception as e:
            execution = AgentExecution(
                agent_id=agent_id,
                task_id=task_id,
                role=state.agent.role,
                success=False,
                error=str(e),
            )
            async with self._lock:
                state.error_count += 1

        finally:
            async with self._lock:
                state.current_task = None

        execution.complete(execution.success, execution.output, execution.error)
        self._executions.append(execution)

        return execution

    async def find_available_agent(self, role: AgentRole) -> AgentId | None:
        """Find an available agent for a role.

        Args:
            role: The required role

        Returns:
            Agent ID if available, None otherwise
        """
        for agent_id, state in self._agents.items():
            if (
                state.agent.role == role
                and state.agent.active
                and state.current_task is None
            ):
                return agent_id
        return None

    async def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Statistics about executions
        """
        total = len(self._executions)
        successful = sum(1 for e in self._executions if e.success)
        failed = total - successful

        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "active_agents": len(self.active_agents),
            "total_agents": len(self._agents),
        }

    async def _health_check_loop(self) -> None:
        """Periodic health check for agents."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                for agent_id, state in list(self._agents.items()):
                    if not state.agent.active:
                        continue

                    state.last_health_check = datetime.now(timezone.utc)

                    # Check for stuck tasks
                    if state.current_task is not None:
                        # Could add timeout handling here
                        pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))
