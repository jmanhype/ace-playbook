"""Base interface for Secrets Providers in BLACKICE 3.0.

Secrets providers abstract credential management, ensuring
secrets are never exposed in logs, prompts, or outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class SecretType(str, Enum):
    """Types of secrets."""

    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    OAUTH_TOKEN = "oauth_token"
    DATABASE_URL = "database_url"
    GENERIC = "generic"


@dataclass
class Secret:
    """A secret credential.

    Note: The actual value is never logged or serialized.
    """

    name: str
    secret_type: SecretType
    _value: str = field(repr=False)  # Hidden from repr

    # Metadata
    description: str | None = None
    created_at: float | None = None
    expires_at: float | None = None
    rotation_due: float | None = None

    # Redaction pattern (for logs)
    redaction_pattern: str | None = None

    def get_value(self) -> str:
        """Get the secret value. Use sparingly."""
        return self._value

    def __str__(self) -> str:
        """Never reveal secret value in string representation."""
        return f"Secret({self.name}, type={self.secret_type.value})"


@dataclass
class SecretReference:
    """A reference to a secret (safe to log/serialize)."""

    name: str
    secret_type: SecretType
    source: str  # Provider that holds it


@dataclass
class InjectionResult:
    """Result of injecting secrets into environment."""

    injected_count: int
    env_vars: dict[str, str]  # Name -> masked value for logging
    errors: list[str] = field(default_factory=list)


@dataclass
class RedactionResult:
    """Result of redacting secrets from text."""

    redacted_text: str
    redaction_count: int
    patterns_matched: list[str] = field(default_factory=list)


@dataclass
class HealthStatus:
    """Health status of a secrets provider."""

    healthy: bool
    secret_count: int = 0
    expiring_soon: int = 0  # Within 7 days
    rotation_due: int = 0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SecretsProvider(Protocol):
    """Protocol for secrets providers.

    Implementations must provide methods for:
    - Secret retrieval (get)
    - Environment injection (inject_env)
    - Text redaction (redact)
    - Health checks (health)
    """

    @property
    def name(self) -> str:
        """Provider name for identification."""
        ...

    async def get(
        self,
        secret_name: str,
        *,
        required: bool = True,
    ) -> Secret | None:
        """Retrieve a secret by name.

        Args:
            secret_name: Name of the secret
            required: Raise error if not found

        Returns:
            Secret if found, None if not required and not found

        Raises:
            SecretsProviderError: If required and not found
        """
        ...

    async def list(self) -> list[SecretReference]:
        """List available secrets (without values).

        Returns:
            List of secret references
        """
        ...

    async def inject_env(
        self,
        secret_names: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> InjectionResult:
        """Inject secrets into environment variables.

        Args:
            secret_names: Names of secrets to inject
            env: Base environment to extend

        Returns:
            InjectionResult with the augmented environment
        """
        ...

    def redact(
        self,
        text: str,
        *,
        replacement: str = "[REDACTED]",
    ) -> RedactionResult:
        """Redact secrets from text.

        Args:
            text: Text that may contain secrets
            replacement: What to replace secrets with

        Returns:
            RedactionResult with cleaned text
        """
        ...

    async def rotate(
        self,
        secret_name: str,
    ) -> Secret:
        """Rotate a secret (generate new value).

        Args:
            secret_name: Name of secret to rotate

        Returns:
            New Secret with rotated value
        """
        ...

    async def health(self) -> HealthStatus:
        """Check provider health.

        Returns:
            HealthStatus indicating if provider is operational
        """
        ...


class BaseSecretsProvider(ABC):
    """Abstract base class for secrets providers.

    Provides common functionality and default implementations
    for the SecretsProvider protocol.
    """

    def __init__(self) -> None:
        self._known_patterns: list[tuple[str, str]] = []  # (pattern, name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @abstractmethod
    async def get(
        self,
        secret_name: str,
        *,
        required: bool = True,
    ) -> Secret | None:
        """Retrieve a secret by name."""
        ...

    @abstractmethod
    async def list(self) -> list[SecretReference]:
        """List available secrets (without values)."""
        ...

    async def inject_env(
        self,
        secret_names: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> InjectionResult:
        """Default injection: fetch and set env vars."""
        result_env = dict(env) if env else {}
        masked_env: dict[str, str] = {}
        errors: list[str] = []
        injected = 0

        for name in secret_names:
            try:
                secret = await self.get(name, required=True)
                if secret:
                    # Use uppercase name as env var
                    env_name = name.upper().replace("-", "_")
                    result_env[env_name] = secret.get_value()
                    masked_env[env_name] = f"***{name}***"
                    injected += 1

                    # Track for redaction
                    self._known_patterns.append((secret.get_value(), name))
            except Exception as e:
                errors.append(f"Failed to get {name}: {e}")

        return InjectionResult(
            injected_count=injected,
            env_vars=masked_env,
            errors=errors,
        )

    def redact(
        self,
        text: str,
        *,
        replacement: str = "[REDACTED]",
    ) -> RedactionResult:
        """Default redaction using known patterns."""
        redacted = text
        count = 0
        matched: list[str] = []

        for pattern, name in self._known_patterns:
            if pattern in redacted:
                redacted = redacted.replace(pattern, replacement)
                count += 1
                matched.append(name)

        # Also check for common secret patterns
        import re

        # API keys (various formats)
        api_key_patterns = [
            (r"sk-[a-zA-Z0-9]{32,}", "api_key"),
            (r"api[_-]?key[=:]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?", "api_key"),
            (r"token[=:]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?", "token"),
        ]

        for pattern, name in api_key_patterns:
            matches = re.findall(pattern, redacted, re.IGNORECASE)
            if matches:
                redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
                count += len(matches)
                matched.append(f"pattern:{name}")

        return RedactionResult(
            redacted_text=redacted,
            redaction_count=count,
            patterns_matched=matched,
        )

    async def rotate(
        self,
        secret_name: str,
    ) -> Secret:
        """Default rotation raises NotImplementedError."""
        raise NotImplementedError(f"{self.name} does not support rotation")

    async def health(self) -> HealthStatus:
        """Default health check."""
        try:
            secrets = await self.list()
            return HealthStatus(
                healthy=True,
                secret_count=len(secrets),
            )
        except Exception as e:
            return HealthStatus(healthy=False, error=str(e))
