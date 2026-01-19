"""Base interface for Execution Providers in BLACKICE 3.0.

Execution providers abstract command execution, supporting multiple
environments (local, container, sandbox) with a unified interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, runtime_checkable


class ExecutionEnvironment(str, Enum):
    """Types of execution environments."""

    LOCAL = "local"
    CONTAINER = "container"
    SANDBOX = "sandbox"
    REMOTE = "remote"


@dataclass
class ExecutionConfig:
    """Configuration for command execution."""

    working_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 300.0  # 5 minutes default
    max_output_bytes: int = 10_000_000  # 10MB
    capture_stderr: bool = True
    shell: bool = False  # Use shell interpretation


@dataclass
class ExecutionResult:
    """Result of a command execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    command: str
    timed_out: bool = False
    killed: bool = False
    # Resource usage
    max_memory_bytes: int | None = None
    cpu_time_seconds: float | None = None


@dataclass
class HealthStatus:
    """Health status of an execution provider."""

    healthy: bool
    environment: ExecutionEnvironment
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionCapabilities:
    """Capabilities of an execution provider."""

    environment: ExecutionEnvironment
    supports_streaming: bool = True
    supports_attach: bool = False
    supports_isolation: bool = False
    max_timeout: float = 3600.0
    max_concurrent: int = 10


@runtime_checkable
class ExecutionProvider(Protocol):
    """Protocol for execution providers.

    Implementations must provide methods for:
    - Command execution (execute)
    - Health checks (health)
    - Optional: streaming output (stream)
    - Optional: interactive attach (attach)
    """

    @property
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Get provider capabilities."""
        ...

    async def execute(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> ExecutionResult:
        """Execute a command.

        Args:
            command: Command string or list of arguments
            config: Execution configuration

        Returns:
            ExecutionResult with output and status
        """
        ...

    async def stream(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream command output line by line.

        Args:
            command: Command string or list of arguments
            config: Execution configuration

        Yields:
            Output lines as they're produced
        """
        ...

    async def health(self) -> HealthStatus:
        """Check provider health.

        Returns:
            HealthStatus indicating if provider is operational
        """
        ...

    async def attach(
        self,
        session_id: str,
    ) -> tuple[AsyncIterator[str], Any]:
        """Attach to an interactive session.

        Args:
            session_id: ID of session to attach to

        Returns:
            Tuple of (output iterator, write function)
        """
        ...


class BaseExecutionProvider(ABC):
    """Abstract base class for execution providers.

    Provides common functionality and default implementations
    for the ExecutionProvider protocol.
    """

    def __init__(
        self,
        default_timeout: float = 300.0,
        default_working_dir: Path | None = None,
    ) -> None:
        self.default_timeout = default_timeout
        self.default_working_dir = default_working_dir

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ExecutionCapabilities:
        """Get provider capabilities."""
        ...

    @abstractmethod
    async def execute(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> ExecutionResult:
        """Execute a command."""
        ...

    async def stream(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> AsyncIterator[str]:
        """Default streaming implementation via execute."""
        result = await self.execute(command, config=config)
        for line in result.stdout.splitlines():
            yield line
        for line in result.stderr.splitlines():
            yield f"[stderr] {line}"

    async def health(self) -> HealthStatus:
        """Default health check via simple command."""
        import time

        start = time.monotonic()
        try:
            result = await self.execute("echo health_check")
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=result.exit_code == 0,
                environment=self.capabilities.environment,
                latency_ms=latency,
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                environment=self.capabilities.environment,
                error=str(e),
            )

    async def attach(
        self,
        session_id: str,
    ) -> tuple[AsyncIterator[str], Any]:
        """Default attach raises NotImplementedError."""
        raise NotImplementedError(f"{self.name} does not support attach")

    def _normalize_command(self, command: str | list[str]) -> list[str]:
        """Normalize command to list of arguments."""
        if isinstance(command, str):
            import shlex

            return shlex.split(command)
        return list(command)

    def _merge_config(self, config: ExecutionConfig | None) -> ExecutionConfig:
        """Merge provided config with defaults."""
        if config is None:
            config = ExecutionConfig()

        if config.working_dir is None and self.default_working_dir:
            config.working_dir = self.default_working_dir

        if config.timeout == 300.0:  # Default value
            config.timeout = self.default_timeout

        return config
