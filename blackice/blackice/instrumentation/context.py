"""Context management for instrumentation in BLACKICE 3.0.

Provides thread-local and async-safe context for correlation IDs,
run context, and other instrumentation data.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any
from uuid import UUID

# Context variables for instrumentation
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_run_id: ContextVar[UUID | None] = ContextVar("run_id", default=None)
_task_id: ContextVar[UUID | None] = ContextVar("task_id", default=None)
_agent_id: ContextVar[str | None] = ContextVar("agent_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the current correlation ID."""
    _correlation_id.set(correlation_id)


def get_run_context() -> dict[str, Any]:
    """Get the current run context."""
    return {
        "run_id": _run_id.get(),
        "task_id": _task_id.get(),
        "agent_id": _agent_id.get(),
    }


def set_run_id(run_id: UUID) -> None:
    """Set the current run ID."""
    _run_id.set(run_id)


def set_task_id(task_id: UUID) -> None:
    """Set the current task ID."""
    _task_id.set(task_id)


def set_agent_id(agent_id: str) -> None:
    """Set the current agent ID."""
    _agent_id.set(agent_id)


def clear_context() -> None:
    """Clear all context variables."""
    _correlation_id.set(None)
    _run_id.set(None)
    _task_id.set(None)
    _agent_id.set(None)


class InstrumentationContext:
    """Context manager for setting instrumentation context.

    Example:
        with InstrumentationContext(run_id=run.id, correlation_id="abc123"):
            # All logs and traces will include run_id and correlation_id
            await execute_task()
    """

    def __init__(
        self,
        correlation_id: str | None = None,
        run_id: UUID | None = None,
        task_id: UUID | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.correlation_id = correlation_id
        self.run_id = run_id
        self.task_id = task_id
        self.agent_id = agent_id
        self._tokens: dict[str, Any] = {}

    def __enter__(self) -> InstrumentationContext:
        if self.correlation_id:
            self._tokens["correlation_id"] = _correlation_id.set(self.correlation_id)
        if self.run_id:
            self._tokens["run_id"] = _run_id.set(self.run_id)
        if self.task_id:
            self._tokens["task_id"] = _task_id.set(self.task_id)
        if self.agent_id:
            self._tokens["agent_id"] = _agent_id.set(self.agent_id)
        return self

    def __exit__(self, *args: Any) -> None:
        for key, token in self._tokens.items():
            var = {
                "correlation_id": _correlation_id,
                "run_id": _run_id,
                "task_id": _task_id,
                "agent_id": _agent_id,
            }[key]
            var.reset(token)

    async def __aenter__(self) -> InstrumentationContext:
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        self.__exit__(*args)
