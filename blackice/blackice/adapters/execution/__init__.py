"""Execution provider adapters for BLACKICE 3.0."""

from blackice.adapters.execution.base import (
    BaseExecutionProvider,
    ExecutionCapabilities,
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionProvider,
    ExecutionResult,
    HealthStatus,
)
from blackice.adapters.execution.container import ContainerExecutionProvider
from blackice.adapters.execution.local import LocalExecutionProvider
from blackice.adapters.execution.safety import (
    CommandAnalysis,
    RiskLevel,
    SafeExecutor,
    SafetyPipeline,
    SafetyPolicy,
)
from blackice.adapters.execution.sandbox import SandboxExecutionProvider

__all__ = [
    # Protocol and base
    "ExecutionProvider",
    "BaseExecutionProvider",
    # Data classes
    "ExecutionEnvironment",
    "ExecutionConfig",
    "ExecutionResult",
    "ExecutionCapabilities",
    "HealthStatus",
    # Providers
    "LocalExecutionProvider",
    "ContainerExecutionProvider",
    "SandboxExecutionProvider",
    # Safety
    "SafetyPipeline",
    "SafetyPolicy",
    "SafeExecutor",
    "CommandAnalysis",
    "RiskLevel",
]
