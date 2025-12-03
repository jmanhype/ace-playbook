"""
UMES configuration management with Pydantic settings.

Provides environment-based configuration for:
- KMS_PROVIDER selection (adapter swapping)
- IDP_TYPE selection (authentication backend)
- DATABASE_URL, REDIS_URL
- Environment-specific overrides

This is the KEY to cloud-agnostic deployment - changing providers
requires ONLY environment variable changes, zero code modifications.

Usage:
    from umes.config import get_settings

    settings = get_settings()
    print(settings.kms_provider)  # KMSProvider.ORACLE
    print(settings.idp_type)      # IDPType.KEYCLOAK
"""
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KMSProvider(str, Enum):
    """Supported KMS providers for cloud-agnostic key management.

    Values match environment variable format (lowercase).
    """
    GCP = "gcp"
    ORACLE = "oracle"
    AWS = "aws"
    AZURE = "azure"
    OPENBAO = "openbao"
    LOCAL = "local"


class IDPType(str, Enum):
    """Supported Identity Provider types.

    Values match environment variable format (lowercase).
    """
    KEYCLOAK = "keycloak"
    AUTH0 = "auth0"
    OKTA = "okta"
    WORKOS = "workos"
    LOCAL = "local"


class Settings(BaseSettings):
    """UMES configuration with environment variable support.

    All settings can be overridden via environment variables.
    Default values are safe for development (local providers).

    Attributes:
        kms_provider: KMS adapter to use (default: local for dev)
        idp_type: IdP adapter to use (default: local for dev)
        database_url: PostgreSQL connection string
        redis_url: Redis connection string (optional - enables L2 cache)
        environment: Deployment environment (development, staging, production)
        openbao_url: OpenBao server URL (required if KMS_PROVIDER=openbao)
    """

    # KMS Configuration
    kms_provider: KMSProvider = Field(
        default=KMSProvider.LOCAL,
        description="KMS provider (gcp/oracle/aws/azure/openbao/local)",
    )

    # IdP Configuration
    idp_type: IDPType = Field(
        default=IDPType.LOCAL,
        description="Identity provider type (keycloak/auth0/okta/workos/local)",
    )

    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://umes:umes@localhost:5432/umes",
        description="PostgreSQL connection URL (async driver)",
    )

    # Cache Configuration
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis URL for L2 cache (optional - L1-only if None)",
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Deployment environment (development/staging/production)",
    )

    # Provider-Specific Configuration
    openbao_url: Optional[str] = Field(
        default=None,
        description="OpenBao server URL (required if KMS_PROVIDER=openbao)",
    )

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow extra env vars (for provider-specific config)
        extra="ignore",
    )

    @field_validator("kms_provider", mode="before")
    @classmethod
    def validate_kms_provider(cls, v: str) -> str:
        """Validate KMS provider value.

        Converts to lowercase for case-insensitive matching.
        Raises ValidationError if invalid.
        """
        if isinstance(v, str):
            v = v.lower()
        return v

    @field_validator("idp_type", mode="before")
    @classmethod
    def validate_idp_type(cls, v: str) -> str:
        """Validate IdP type value.

        Converts to lowercase for case-insensitive matching.
        Raises ValidationError if invalid.
        """
        if isinstance(v, str):
            v = v.lower()
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern).

    Settings are loaded once and cached. Subsequent calls return
    the same instance. This prevents re-reading environment variables
    on every access.

    Returns:
        Settings instance with current configuration

    Example:
        >>> from umes.config import get_settings
        >>> settings = get_settings()
        >>> settings.kms_provider
        <KMSProvider.ORACLE: 'oracle'>
    """
    return Settings()


# Export public API
__all__ = [
    "Settings",
    "KMSProvider",
    "IDPType",
    "get_settings",
]
