"""Secrets provider adapters for BLACKICE 3.0."""

from blackice.adapters.secrets.base import (
    BaseSecretsProvider,
    HealthStatus,
    InjectionResult,
    RedactionResult,
    Secret,
    SecretReference,
    SecretsProvider,
    SecretType,
)
from blackice.adapters.secrets.env import (
    EnvPattern,
    EnvSecretsProvider,
)
from blackice.adapters.secrets.redaction import (
    RedactionLevel,
    RedactionPattern,
    RedactionResult as AdvancedRedactionResult,
    SecretRedactor,
    create_log_filter,
    get_default_redactor,
    redact,
    register_secret,
)

__all__ = [
    # Protocol and base
    "SecretsProvider",
    "BaseSecretsProvider",
    # Data classes
    "SecretType",
    "Secret",
    "SecretReference",
    "InjectionResult",
    "RedactionResult",
    "HealthStatus",
    # Env provider
    "EnvSecretsProvider",
    "EnvPattern",
    # Advanced redaction
    "RedactionLevel",
    "RedactionPattern",
    "AdvancedRedactionResult",
    "SecretRedactor",
    "create_log_filter",
    "get_default_redactor",
    "redact",
    "register_secret",
]
