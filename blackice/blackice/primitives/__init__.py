"""BLACKICE primitives - foundational types, errors, and patterns.

This module exports all primitives for use throughout the BLACKICE system.
"""

from blackice.primitives.errors import (
    BlackiceError,
    CheckpointError,
    CommandBlockedError,
    CommandError,
    ConfigError,
    ConnectivityProviderError,
    EditionError,
    EventLogError,
    ExecutionError,
    ExecutionProviderError,
    HashChainError,
    InvalidConfigError,
    MemoryProviderError,
    MissingConfigError,
    ModelProviderError,
    PolicyViolationError,
    ProviderError,
    RecoveryError,
    RunError,
    SecretExposureError,
    SecretsProviderError,
    SecurityError,
    TaskError,
    TaskSpecValidationError,
    TimeoutError,
    ValidationError,
    VisionValidationError,
)
from blackice.primitives.patterns import (
    CircuitBreaker,
    Either,
    Err,
    Left,
    Ok,
    Result,
    Right,
    async_retry,
    err,
    left,
    ok,
    retry,
    right,
)
from blackice.primitives.types import (
    AgentId,
    AgentRole,
    BeadId,
    CorrelationId,
    Edition,
    EventId,
    EventType,
    Hash,
    PIIPolicy,
    RunId,
    RunStatus,
    StrictnessLevel,
    TaskId,
    TaskStatus,
    Timestamp,
    deserialize_json,
    new_event_id,
    new_run_id,
    new_task_id,
    serialize_json,
)

__all__ = [
    # Types - Identifiers
    "RunId",
    "TaskId",
    "EventId",
    "AgentId",
    "BeadId",
    "new_run_id",
    "new_task_id",
    "new_event_id",
    # Types - Enumerations
    "Edition",
    "AgentRole",
    "RunStatus",
    "TaskStatus",
    "EventType",
    "StrictnessLevel",
    "PIIPolicy",
    # Types - Value Objects
    "Timestamp",
    "Hash",
    "CorrelationId",
    # Types - Serialization
    "serialize_json",
    "deserialize_json",
    # Errors - Base
    "BlackiceError",
    # Errors - Configuration
    "ConfigError",
    "MissingConfigError",
    "InvalidConfigError",
    "EditionError",
    # Errors - Execution
    "ExecutionError",
    "CommandError",
    "TimeoutError",
    "TaskError",
    "RunError",
    # Errors - Providers
    "ProviderError",
    "ModelProviderError",
    "MemoryProviderError",
    "ExecutionProviderError",
    "ConnectivityProviderError",
    "SecretsProviderError",
    # Errors - Security
    "SecurityError",
    "CommandBlockedError",
    "PolicyViolationError",
    "SecretExposureError",
    # Errors - Recovery
    "RecoveryError",
    "CheckpointError",
    "EventLogError",
    "HashChainError",
    # Errors - Validation
    "ValidationError",
    "VisionValidationError",
    "TaskSpecValidationError",
    # Patterns - Result
    "Result",
    "Ok",
    "Err",
    "ok",
    "err",
    # Patterns - Either
    "Either",
    "Left",
    "Right",
    "left",
    "right",
    # Patterns - Retry
    "retry",
    "async_retry",
    # Patterns - Circuit Breaker
    "CircuitBreaker",
]
