"""Base interface for Connectivity Providers in BLACKICE 3.0.

Connectivity providers abstract remote environment connections,
supporting SSH, WireGuard, and other tunneling protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Protocol, runtime_checkable


class ConnectionType(str, Enum):
    """Types of connections."""

    SSH = "ssh"
    WIREGUARD = "wireguard"
    TAILSCALE = "tailscale"
    DIRECT = "direct"


class ConnectionStatus(str, Enum):
    """Status of a connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ConnectionConfig:
    """Configuration for a connection."""

    host: str
    port: int = 22
    username: str | None = None
    key_path: str | None = None
    password: str | None = None
    timeout: float = 30.0
    keep_alive_interval: float = 60.0
    retry_count: int = 3
    retry_delay: float = 5.0


@dataclass
class PortForward:
    """Port forwarding configuration."""

    local_port: int
    remote_host: str
    remote_port: int
    bind_address: str = "127.0.0.1"


@dataclass
class Connection:
    """An active connection."""

    id: str
    connection_type: ConnectionType
    status: ConnectionStatus
    host: str
    port: int
    established_at: float | None = None
    last_activity: float | None = None
    forwards: list[PortForward] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Health status of a connectivity provider."""

    healthy: bool
    connection_type: ConnectionType
    active_connections: int = 0
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ConnectivityProvider(Protocol):
    """Protocol for connectivity providers.

    Implementations must provide methods for:
    - Connection management (attach, detach)
    - Rescue operations (rescue)
    - Port forwarding (port_forward)
    - Health checks (health)
    """

    @property
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @property
    def connection_type(self) -> ConnectionType:
        """Type of connection this provider handles."""
        ...

    async def attach(
        self,
        config: ConnectionConfig,
    ) -> Connection:
        """Establish a connection to a remote environment.

        Args:
            config: Connection configuration

        Returns:
            Active Connection object
        """
        ...

    async def detach(
        self,
        connection_id: str,
    ) -> bool:
        """Close a connection.

        Args:
            connection_id: ID of connection to close

        Returns:
            True if closed, False if not found
        """
        ...

    async def rescue(
        self,
        connection_id: str,
    ) -> Connection:
        """Attempt to rescue a failed connection.

        Args:
            connection_id: ID of connection to rescue

        Returns:
            Restored Connection object
        """
        ...

    async def port_forward(
        self,
        connection_id: str,
        forward: PortForward,
    ) -> bool:
        """Set up port forwarding on a connection.

        Args:
            connection_id: Connection to use
            forward: Port forwarding configuration

        Returns:
            True if successful
        """
        ...

    async def execute(
        self,
        connection_id: str,
        command: str,
        *,
        timeout: float = 300.0,
    ) -> tuple[int, str, str]:
        """Execute a command on the remote environment.

        Args:
            connection_id: Connection to use
            command: Command to execute
            timeout: Command timeout

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        ...

    async def stream_execute(
        self,
        connection_id: str,
        command: str,
        *,
        timeout: float = 300.0,
    ) -> AsyncIterator[str]:
        """Execute a command and stream output.

        Args:
            connection_id: Connection to use
            command: Command to execute
            timeout: Command timeout

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


class BaseConnectivityProvider(ABC):
    """Abstract base class for connectivity providers.

    Provides common functionality and default implementations
    for the ConnectivityProvider protocol.
    """

    def __init__(self) -> None:
        self._connections: dict[str, Connection] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @property
    @abstractmethod
    def connection_type(self) -> ConnectionType:
        """Type of connection this provider handles."""
        ...

    @abstractmethod
    async def attach(
        self,
        config: ConnectionConfig,
    ) -> Connection:
        """Establish a connection to a remote environment."""
        ...

    async def detach(
        self,
        connection_id: str,
    ) -> bool:
        """Close a connection."""
        if connection_id in self._connections:
            conn = self._connections.pop(connection_id)
            conn.status = ConnectionStatus.DISCONNECTED
            return True
        return False

    async def rescue(
        self,
        connection_id: str,
    ) -> Connection:
        """Default rescue: re-attach with same config."""
        if connection_id not in self._connections:
            raise ValueError(f"Connection {connection_id} not found")

        conn = self._connections[connection_id]
        conn.status = ConnectionStatus.RECONNECTING

        # Re-establish connection
        config = ConnectionConfig(
            host=conn.host,
            port=conn.port,
        )
        return await self.attach(config)

    @abstractmethod
    async def port_forward(
        self,
        connection_id: str,
        forward: PortForward,
    ) -> bool:
        """Set up port forwarding on a connection."""
        ...

    @abstractmethod
    async def execute(
        self,
        connection_id: str,
        command: str,
        *,
        timeout: float = 300.0,
    ) -> tuple[int, str, str]:
        """Execute a command on the remote environment."""
        ...

    async def stream_execute(
        self,
        connection_id: str,
        command: str,
        *,
        timeout: float = 300.0,
    ) -> AsyncIterator[str]:
        """Default streaming: execute and yield lines."""
        exit_code, stdout, stderr = await self.execute(
            connection_id, command, timeout=timeout
        )
        for line in stdout.splitlines():
            yield line
        for line in stderr.splitlines():
            yield f"[stderr] {line}"

    async def health(self) -> HealthStatus:
        """Default health check."""
        return HealthStatus(
            healthy=True,
            connection_type=self.connection_type,
            active_connections=len(self._connections),
        )

    def _generate_connection_id(self) -> str:
        """Generate a unique connection ID."""
        import uuid

        return str(uuid.uuid4())[:8]
