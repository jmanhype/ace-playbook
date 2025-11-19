"""
Core configuration for RCST v2 backend.

Implements Constitution §III (Security-First), §IV (Compliance), §VI (Observability).
Uses Pydantic settings for type-safe environment variable loading.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Security:
    - Enforces minimum SECRET_KEY length (32 chars) per Constitution §III
    - Validates JWT configuration
    - Configures session timeouts per Spec §FR-015

    Compliance:
    - Audit log retention per Plan §Scale/Scope
    - ALCOA+ timestamp format (ISO 8601 UTC)

    Observability:
    - Structured logging configuration
    - OpenTelemetry tracing support
    - Prometheus metrics
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # =========================================================================
    # Application
    # =========================================================================
    ENVIRONMENT: Literal["development", "testing", "staging", "production"] = "development"
    API_VERSION: str = "v1"
    DEBUG: bool = Field(default=False, description="Enable debug mode")

    # =========================================================================
    # Security (Constitution §III)
    # =========================================================================
    SECRET_KEY: str = Field(
        ...,
        min_length=32,
        description="Secret key for cryptographic operations (min 32 chars)",
    )
    JWT_SECRET_KEY: str | None = Field(
        default=None,
        description="Separate JWT secret (defaults to SECRET_KEY if not set)",
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = Field(
        default=15, description="JWT access token expiration (Plan §Authentication)"
    )
    REFRESH_TOKEN_EXPIRATION_DAYS: int = Field(
        default=7, description="Refresh token expiration (Plan §Authentication)"
    )
    SESSION_TIMEOUT_MINUTES: int = Field(
        default=30, description="Session inactivity timeout (Spec §FR-015)"
    )

    # =========================================================================
    # Database (PostgreSQL with pgvector)
    # =========================================================================
    DATABASE_URL: str = Field(
        ..., description="PostgreSQL connection string (asyncpg format)"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(
        default=20, description="Max overflow connections beyond pool size"
    )

    # =========================================================================
    # Redis (caching & Celery broker)
    # =========================================================================
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_CACHE_TTL_SECONDS: int = Field(default=300, description="Default cache TTL (5 minutes)")

    # =========================================================================
    # CORS (Constitution §III Security)
    # =========================================================================
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins (comma-separated in env)",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # =========================================================================
    # OIDC Authentication (Plan §Authentication & Authorization)
    # =========================================================================
    OIDC_DISCOVERY_URL: str | None = Field(
        default=None, description="OIDC provider discovery URL"
    )
    OIDC_CLIENT_ID: str | None = Field(default=None, description="OIDC client ID")
    OIDC_CLIENT_SECRET: str | None = Field(default=None, description="OIDC client secret")

    # =========================================================================
    # LLM Providers (Constitution §VII Graceful Degradation)
    # =========================================================================
    OPENAI_API_KEY: str | None = Field(default=None, description="OpenAI API key (primary)")
    ANTHROPIC_API_KEY: str | None = Field(
        default=None, description="Anthropic API key (secondary)"
    )
    OLLAMA_BASE_URL: str | None = Field(
        default=None, description="Ollama base URL (local fallback)"
    )

    # =========================================================================
    # Storage (S3/Azure for WORM audit logs)
    # =========================================================================
    STORAGE_BACKEND: Literal["local", "s3", "azure"] = Field(
        default="local", description="Storage backend for files and audit logs"
    )
    STORAGE_LOCAL_PATH: str = Field(default="./storage", description="Local storage path")
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_S3_BUCKET: str | None = None
    AWS_S3_REGION: str = "us-east-1"
    AZURE_STORAGE_CONNECTION_STRING: str | None = None
    AZURE_STORAGE_CONTAINER: str | None = None

    # =========================================================================
    # Observability (Constitution §VI)
    # =========================================================================
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    ENABLE_TRACING: bool = Field(
        default=False, description="Enable OpenTelemetry distributed tracing"
    )
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = Field(
        default=None, description="OTLP endpoint for traces (e.g., Jaeger)"
    )
    PROMETHEUS_MULTIPROC_DIR: str | None = Field(
        default=None, description="Prometheus multiprocess directory for Celery workers"
    )

    # =========================================================================
    # Rate Limiting (Plan §Constraints)
    # =========================================================================
    RATE_LIMIT_UNAUTHENTICATED: int = Field(
        default=10, description="Rate limit for unauthenticated requests (req/min)"
    )
    RATE_LIMIT_AUTHENTICATED: int = Field(
        default=100, description="Rate limit for authenticated requests (req/min)"
    )
    RATE_LIMIT_ADMIN: int = Field(
        default=1000, description="Rate limit for admin requests (req/min)"
    )

    # =========================================================================
    # File Upload (Spec §FR-001)
    # =========================================================================
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50, description="Maximum file upload size in MB (Spec §FR-001)"
    )

    # =========================================================================
    # Compliance (Constitution §IV, Plan §Scale/Scope)
    # =========================================================================
    AUDIT_LOG_RETENTION_YEARS: int = Field(
        default=7, description="Audit log retention period (Plan §Scale/Scope)"
    )
    ALCOA_TIMESTAMP_FORMAT: str = Field(
        default="ISO8601_UTC",
        description="ALCOA+ timestamp format (Constitution §IV, Spec §FR-010)",
    )

    # =========================================================================
    # Embedding Model (Plan §Dependencies, research.md)
    # =========================================================================
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings (384D per research.md)",
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384, description="Embedding vector dimension (matches sentence-transformers)"
    )

    # =========================================================================
    # Celery (Background Tasks)
    # =========================================================================
    CELERY_BROKER_URL: str | None = Field(
        default=None, description="Celery broker URL (defaults to REDIS_URL)"
    )
    CELERY_RESULT_BACKEND: str | None = Field(
        default=None, description="Celery result backend (defaults to REDIS_URL)"
    )

    @field_validator("JWT_SECRET_KEY", mode="after")
    @classmethod
    def default_jwt_secret_to_secret_key(cls, v: str | None, info) -> str:
        """Default JWT_SECRET_KEY to SECRET_KEY if not explicitly set."""
        if v is None:
            return info.data["SECRET_KEY"]
        return v

    @field_validator("CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", mode="after")
    @classmethod
    def default_celery_to_redis(cls, v: str | None, info) -> str:
        """Default Celery URLs to REDIS_URL if not explicitly set."""
        if v is None:
            return info.data["REDIS_URL"]
        return v

    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are loaded once and reused.
    FastAPI Depends uses this function for dependency injection.
    """
    return Settings()
