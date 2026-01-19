"""Connectivity provider adapters for BLACKICE 3.0."""

from blackice.adapters.connectivity.base import (
    BaseConnectivityProvider,
    Connection,
    ConnectionConfig,
    ConnectionStatus,
    ConnectionType,
    ConnectivityProvider,
    HealthStatus,
    PortForward,
)
from blackice.adapters.connectivity.ssh import SSHConnectivityProvider
from blackice.adapters.connectivity.wireguard import (
    WireGuardConfig,
    WireGuardConnectivityProvider,
)

__all__ = [
    "ConnectivityProvider",
    "BaseConnectivityProvider",
    "ConnectionType",
    "ConnectionStatus",
    "ConnectionConfig",
    "Connection",
    "PortForward",
    "HealthStatus",
    "SSHConnectivityProvider",
    "WireGuardConnectivityProvider",
    "WireGuardConfig",
]
