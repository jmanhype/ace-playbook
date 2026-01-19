"""BLACKICE adapters - Provider interfaces and implementations.

This module exports the provider protocols and base classes for:
- Model providers (LLM backends)
- Execution providers (command execution environments)
- Memory providers (persistent memory storage)
- Connectivity providers (remote environment connections)
- Secrets providers (credential management)
"""

from blackice.adapters.connectivity import (
    BaseConnectivityProvider,
    Connection,
    ConnectionConfig,
    ConnectionStatus,
    ConnectionType,
    ConnectivityProvider,
    PortForward,
)
from blackice.adapters.execution import (
    BaseExecutionProvider,
    ExecutionCapabilities,
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionProvider,
    ExecutionResult,
)
from blackice.adapters.memory import (
    BaseMemoryProvider,
    ContextWindow,
    MemoryEntry,
    MemoryProvider,
    MemoryType,
    RetentionPolicy,
    SearchResult,
)
from blackice.adapters.models import (
    BaseModelProvider,
    EmbeddingResult,
    GenerationResult,
    Message,
    ModelCapabilities,
    ModelProvider,
    ToolCall,
    ToolResult,
)
from blackice.adapters.secrets import (
    BaseSecretsProvider,
    InjectionResult,
    RedactionResult,
    Secret,
    SecretReference,
    SecretsProvider,
    SecretType,
)

__all__ = [
    # Models
    "ModelProvider",
    "BaseModelProvider",
    "Message",
    "ToolCall",
    "ToolResult",
    "GenerationResult",
    "EmbeddingResult",
    "ModelCapabilities",
    # Execution
    "ExecutionProvider",
    "BaseExecutionProvider",
    "ExecutionEnvironment",
    "ExecutionConfig",
    "ExecutionResult",
    "ExecutionCapabilities",
    # Memory
    "MemoryProvider",
    "BaseMemoryProvider",
    "MemoryType",
    "MemoryEntry",
    "RetentionPolicy",
    "SearchResult",
    "ContextWindow",
    # Connectivity
    "ConnectivityProvider",
    "BaseConnectivityProvider",
    "ConnectionType",
    "ConnectionStatus",
    "ConnectionConfig",
    "Connection",
    "PortForward",
    # Secrets
    "SecretsProvider",
    "BaseSecretsProvider",
    "SecretType",
    "Secret",
    "SecretReference",
    "InjectionResult",
    "RedactionResult",
]
