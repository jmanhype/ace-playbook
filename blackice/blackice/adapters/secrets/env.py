"""Environment-based secrets provider for BLACKICE 3.0.

Reads secrets from environment variables with pattern-based discovery.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from blackice.adapters.secrets.base import (
    BaseSecretsProvider,
    HealthStatus,
    Secret,
    SecretReference,
    SecretType,
)
from blackice.primitives.errors import SecretsProviderError


@dataclass
class EnvPattern:
    """Pattern for discovering secrets in environment variables."""

    pattern: str  # Regex pattern for env var names
    secret_type: SecretType
    description: str | None = None


# Default patterns for common secrets
DEFAULT_ENV_PATTERNS: list[EnvPattern] = [
    EnvPattern(
        r"^ANTHROPIC_API_KEY$",
        SecretType.API_KEY,
        "Anthropic API key for Claude",
    ),
    EnvPattern(
        r"^OPENAI_API_KEY$",
        SecretType.API_KEY,
        "OpenAI API key",
    ),
    EnvPattern(
        r"^OLLAMA_API_KEY$",
        SecretType.API_KEY,
        "Ollama API key (optional)",
    ),
    EnvPattern(
        r"^GITHUB_TOKEN$",
        SecretType.TOKEN,
        "GitHub personal access token",
    ),
    EnvPattern(
        r"^DATABASE_URL$",
        SecretType.DATABASE_URL,
        "Database connection string",
    ),
    EnvPattern(
        r"^.*_API_KEY$",
        SecretType.API_KEY,
        "Generic API key",
    ),
    EnvPattern(
        r"^.*_TOKEN$",
        SecretType.TOKEN,
        "Generic token",
    ),
    EnvPattern(
        r"^.*_SECRET$",
        SecretType.GENERIC,
        "Generic secret",
    ),
    EnvPattern(
        r"^.*_PASSWORD$",
        SecretType.PASSWORD,
        "Password",
    ),
    EnvPattern(
        r"^.*_PRIVATE_KEY$",
        SecretType.SSH_KEY,
        "Private key",
    ),
]


class EnvSecretsProvider(BaseSecretsProvider):
    """Secrets provider that reads from environment variables.

    Supports:
    - Pattern-based discovery of secrets
    - Explicit secret mapping
    - Environment variable prefixes
    """

    def __init__(
        self,
        prefix: str | None = None,
        patterns: list[EnvPattern] | None = None,
        explicit_mappings: dict[str, str] | None = None,
    ) -> None:
        """Initialize the environment secrets provider.

        Args:
            prefix: Optional prefix for env vars (e.g., "BLACKICE_")
            patterns: Custom patterns for secret discovery
            explicit_mappings: Explicit secret_name -> env_var mappings
        """
        super().__init__()
        self.prefix = prefix or ""
        self.patterns = patterns or DEFAULT_ENV_PATTERNS
        self.explicit_mappings = explicit_mappings or {}
        self._cache: dict[str, Secret] = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = 60.0  # 1 minute

    @property
    def name(self) -> str:
        return "env"

    def _classify_secret(self, env_name: str) -> tuple[SecretType, str | None]:
        """Classify a secret based on its environment variable name."""
        for pattern in self.patterns:
            if re.match(pattern.pattern, env_name, re.IGNORECASE):
                return pattern.secret_type, pattern.description
        return SecretType.GENERIC, None

    def _get_env_name(self, secret_name: str) -> str:
        """Convert secret name to environment variable name."""
        # Check explicit mappings first
        if secret_name in self.explicit_mappings:
            return self.explicit_mappings[secret_name]

        # Apply prefix and normalize
        env_name = secret_name.upper().replace("-", "_").replace(".", "_")
        if self.prefix:
            return f"{self.prefix}{env_name}"
        return env_name

    def _refresh_cache(self) -> None:
        """Refresh the secrets cache if TTL expired."""
        now = time.monotonic()
        if now - self._cache_time < self._cache_ttl:
            return

        self._cache.clear()
        self._known_patterns.clear()

        # Discover secrets from environment
        for env_name, value in os.environ.items():
            # Skip empty values
            if not value:
                continue

            # Check if matches any pattern
            secret_type, description = self._classify_secret(env_name)

            # Only cache if it looks like a secret
            if secret_type != SecretType.GENERIC or env_name.endswith(
                ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")
            ):
                secret_name = env_name.lower().replace("_", "-")
                if self.prefix and secret_name.startswith(self.prefix.lower()):
                    secret_name = secret_name[len(self.prefix) :]

                self._cache[secret_name] = Secret(
                    name=secret_name,
                    secret_type=secret_type,
                    _value=value,
                    description=description,
                    created_at=time.time(),
                )
                # Track for redaction
                self._known_patterns.append((value, secret_name))

        self._cache_time = now

    async def get(
        self,
        secret_name: str,
        *,
        required: bool = True,
    ) -> Secret | None:
        """Retrieve a secret from environment variables.

        Args:
            secret_name: Name of the secret (case-insensitive)
            required: Raise error if not found

        Returns:
            Secret if found

        Raises:
            SecretsProviderError: If required and not found
        """
        # First check cache
        self._refresh_cache()

        # Normalize name
        normalized = secret_name.lower().replace("_", "-")

        if normalized in self._cache:
            return self._cache[normalized]

        # Try direct lookup
        env_name = self._get_env_name(secret_name)
        value = os.environ.get(env_name)

        if value:
            secret_type, description = self._classify_secret(env_name)
            secret = Secret(
                name=normalized,
                secret_type=secret_type,
                _value=value,
                description=description,
                created_at=time.time(),
            )
            self._cache[normalized] = secret
            self._known_patterns.append((value, normalized))
            return secret

        if required:
            raise SecretsProviderError(
                f"Secret '{secret_name}' not found in environment (tried {env_name})",
                context={"secret_name": secret_name, "env_var": env_name},
            )

        return None

    async def list(self) -> list[SecretReference]:
        """List all discovered secrets (without values)."""
        self._refresh_cache()

        return [
            SecretReference(
                name=secret.name,
                secret_type=secret.secret_type,
                source=self.name,
            )
            for secret in self._cache.values()
        ]

    async def health(self) -> HealthStatus:
        """Check provider health and report secret statistics."""
        try:
            secrets = await self.list()

            # Check for critical secrets
            critical_secrets = {"anthropic-api-key", "openai-api-key"}
            found_critical = sum(1 for s in secrets if s.name in critical_secrets)

            return HealthStatus(
                healthy=True,
                secret_count=len(secrets),
                details={
                    "provider": self.name,
                    "prefix": self.prefix or "(none)",
                    "critical_secrets_found": found_critical,
                    "pattern_count": len(self.patterns),
                },
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                error=str(e),
                details={"provider": self.name},
            )

    def clear_cache(self) -> None:
        """Clear the secrets cache (forces re-discovery)."""
        self._cache.clear()
        self._cache_time = 0.0
        self._known_patterns.clear()
