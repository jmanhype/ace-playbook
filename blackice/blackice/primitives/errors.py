"""Exception hierarchy for BLACKICE 3.0.

This module defines a structured exception hierarchy:
- BlackiceError: Base for all BLACKICE exceptions
- ConfigError: Configuration and validation errors
- ExecutionError: Runtime execution errors
- ProviderError: External provider failures
- SecurityError: Security policy violations
- RecoveryError: State recovery failures
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from blackice.primitives.types import RunId, TaskId


class BlackiceError(Exception):
    """Base exception for all BLACKICE errors."""

    code: str = "E0000"
    recoverable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        recoverable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.context = context or {}
        if recoverable is not None:
            self.recoverable = recoverable

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable,
        }


# === Configuration Errors ===


class ConfigError(BlackiceError):
    """Configuration or validation error."""

    code = "E1000"


class MissingConfigError(ConfigError):
    """Required configuration is missing."""

    code = "E1001"

    def __init__(self, key: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"Missing required configuration: {key}",
            context={"key": key, **(context or {})},
        )


class InvalidConfigError(ConfigError):
    """Configuration value is invalid."""

    code = "E1002"

    def __init__(
        self, key: str, value: Any, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Invalid configuration for '{key}': {reason}",
            context={"key": key, "value": str(value), "reason": reason, **(context or {})},
        )


class EditionError(ConfigError):
    """Feature requires a higher edition tier."""

    code = "E1003"

    def __init__(
        self, feature: str, required: str, current: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Feature '{feature}' requires {required} edition (current: {current})",
            context={
                "feature": feature,
                "required_edition": required,
                "current_edition": current,
                **(context or {}),
            },
        )


# === Execution Errors ===


class ExecutionError(BlackiceError):
    """Runtime execution error."""

    code = "E2000"
    recoverable = True


class CommandError(ExecutionError):
    """Command execution failed."""

    code = "E2001"

    def __init__(
        self, command: str, exit_code: int, stderr: str = "", *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Command failed with exit code {exit_code}: {command[:100]}",
            context={
                "command": command,
                "exit_code": exit_code,
                "stderr": stderr[:500],
                **(context or {}),
            },
        )


class TimeoutError(ExecutionError):
    """Operation timed out."""

    code = "E2002"

    def __init__(
        self, operation: str, timeout_seconds: float, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s",
            context={"operation": operation, "timeout_seconds": timeout_seconds, **(context or {})},
        )


class TaskError(ExecutionError):
    """Task execution failed."""

    code = "E2003"

    def __init__(
        self, task_id: TaskId, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Task {task_id} failed: {reason}",
            context={"task_id": str(task_id), "reason": reason, **(context or {})},
        )


class RunError(ExecutionError):
    """Run-level execution error."""

    code = "E2004"

    def __init__(
        self, run_id: RunId, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Run {run_id} failed: {reason}",
            context={"run_id": str(run_id), "reason": reason, **(context or {})},
        )


# === Provider Errors ===


class ProviderError(BlackiceError):
    """External provider failure."""

    code = "E3000"
    recoverable = True


class ModelProviderError(ProviderError):
    """Model provider (LLM) error."""

    code = "E3001"

    def __init__(
        self,
        provider: str,
        reason: str,
        *,
        retryable: bool = True,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Model provider '{provider}' error: {reason}",
            recoverable=retryable,
            context={"provider": provider, "reason": reason, **(context or {})},
        )


class MemoryProviderError(ProviderError):
    """Memory provider (Letta) error."""

    code = "E3002"

    def __init__(
        self, operation: str, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Memory provider error during '{operation}': {reason}",
            context={"operation": operation, "reason": reason, **(context or {})},
        )


class ExecutionProviderError(ProviderError):
    """Execution provider (sandbox) error."""

    code = "E3003"

    def __init__(
        self, provider: str, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Execution provider '{provider}' error: {reason}",
            context={"provider": provider, "reason": reason, **(context or {})},
        )


class ConnectivityProviderError(ProviderError):
    """Connectivity provider (remote environment) error."""

    code = "E3004"

    def __init__(
        self, target: str, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Connectivity error to '{target}': {reason}",
            context={"target": target, "reason": reason, **(context or {})},
        )


class SecretsProviderError(ProviderError):
    """Secrets provider error."""

    code = "E3005"

    def __init__(
        self, operation: str, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Secrets provider error during '{operation}': {reason}",
            context={"operation": operation, "reason": reason, **(context or {})},
        )


# === Security Errors ===


class SecurityError(BlackiceError):
    """Security policy violation."""

    code = "E4000"
    recoverable = False


class CommandBlockedError(SecurityError):
    """Command blocked by safety pipeline."""

    code = "E4001"

    def __init__(
        self,
        command: str,
        reason: str,
        *,
        policy: str = "default",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Command blocked by safety policy: {reason}",
            context={"command": command[:200], "reason": reason, "policy": policy, **(context or {})},
        )


class PolicyViolationError(SecurityError):
    """TaskSpec policy violation (Enterprise)."""

    code = "E4002"

    def __init__(
        self, violation: str, strictness: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"TaskSpec policy violation ({strictness}): {violation}",
            context={"violation": violation, "strictness": strictness, **(context or {})},
        )


class SecretExposureError(SecurityError):
    """Potential secret exposure detected."""

    code = "E4003"

    def __init__(self, location: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"Potential secret exposure detected in {location}",
            context={"location": location, **(context or {})},
        )


# === Recovery Errors ===


class RecoveryError(BlackiceError):
    """State recovery failure."""

    code = "E5000"


class CheckpointError(RecoveryError):
    """Checkpoint creation or loading failed."""

    code = "E5001"

    def __init__(
        self, operation: str, run_id: RunId, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Checkpoint {operation} failed for run {run_id}: {reason}",
            context={
                "operation": operation,
                "run_id": str(run_id),
                "reason": reason,
                **(context or {}),
            },
        )


class EventLogError(RecoveryError):
    """Event log operation failed."""

    code = "E5002"

    def __init__(
        self, operation: str, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Event log {operation} failed: {reason}",
            context={"operation": operation, "reason": reason, **(context or {})},
        )


class HashChainError(RecoveryError):
    """Hash chain integrity violation."""

    code = "E5003"

    def __init__(
        self, event_id: str, expected: str, actual: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Hash chain broken at event {event_id}",
            context={
                "event_id": event_id,
                "expected_hash": expected,
                "actual_hash": actual,
                **(context or {}),
            },
        )


class StateError(RecoveryError):
    """State machine transition error."""

    code = "E5010"

    def __init__(
        self, message: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, context=context)


# === Validation Errors ===


class ValidationError(BlackiceError):
    """Input validation error."""

    code = "E6000"

    def __init__(
        self, field: str, reason: str, *, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            f"Validation failed for '{field}': {reason}",
            context={"field": field, "reason": reason, **(context or {})},
        )


class VisionValidationError(ValidationError):
    """Vision description validation failed."""

    code = "E6001"

    def __init__(self, reason: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__("vision", reason, context=context)


class TaskSpecValidationError(ValidationError):
    """TaskSpec validation failed (Enterprise)."""

    code = "E6002"

    def __init__(self, reason: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__("taskspec", reason, context=context)
