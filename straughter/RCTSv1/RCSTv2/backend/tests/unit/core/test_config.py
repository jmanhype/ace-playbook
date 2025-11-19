"""
Unit tests for core configuration.

Following TDD - tests written before implementation.
Tests verify Constitution §I (TDD), §III (Security-First), §IV (Compliance).
"""

import os
from typing import Any

import pytest
from pydantic import ValidationError


def test_settings_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that settings correctly load from environment variables."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")
    monkeypatch.setenv("ENVIRONMENT", "testing")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.DATABASE_URL == "postgresql://test:test@localhost/testdb"
    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.SECRET_KEY == "test-secret-key-32-characters-long!"
    assert settings.ENVIRONMENT == "testing"


def test_settings_requires_database_url() -> None:
    """Test that DATABASE_URL is required."""
    # Arrange
    os.environ.pop("DATABASE_URL", None)

    # Act & Assert
    from src.core.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "DATABASE_URL" in str(exc_info.value)


def test_settings_requires_secret_key() -> None:
    """Test that SECRET_KEY is required and meets minimum length."""
    # Arrange
    os.environ.pop("SECRET_KEY", None)

    # Act & Assert
    from src.core.config import Settings

    with pytest.raises(ValidationError):
        Settings()


def test_settings_secret_key_minimum_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that SECRET_KEY must be at least 32 characters (Constitution §III Security)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "too-short")

    # Act & Assert
    from src.core.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "at least 32 characters" in str(exc_info.value).lower()


def test_settings_defaults_environment_to_development() -> None:
    """Test that ENVIRONMENT defaults to 'development'."""
    # Arrange
    os.environ.pop("ENVIRONMENT", None)

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.ENVIRONMENT == "development"


def test_settings_jwt_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JWT configuration defaults (Plan §Authentication & Authorization)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.JWT_ALGORITHM == "HS256"
    assert settings.JWT_EXPIRATION_MINUTES == 15
    assert settings.REFRESH_TOKEN_EXPIRATION_DAYS == 7


def test_settings_session_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test session timeout configuration (Spec §FR-015)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.SESSION_TIMEOUT_MINUTES == 30


def test_settings_database_pool_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test database connection pool settings."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")
    monkeypatch.setenv("DATABASE_POOL_SIZE", "20")
    monkeypatch.setenv("DATABASE_MAX_OVERFLOW", "40")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.DATABASE_POOL_SIZE == 20
    assert settings.DATABASE_MAX_OVERFLOW == 40


def test_settings_cors_origins_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CORS origins are correctly parsed from comma-separated string."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.CORS_ORIGINS == ["http://localhost:3000", "http://localhost:5173"]


def test_settings_log_level_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test LOG_LEVEL defaults to INFO (Constitution §VI Observability)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")
    os.environ.pop("LOG_LEVEL", None)

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.LOG_LEVEL == "INFO"


def test_settings_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test API version configuration (Constitution §V API Versioning)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.API_VERSION == "v1"


def test_settings_rate_limiting_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test rate limiting configuration (Plan §Constraints)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.RATE_LIMIT_UNAUTHENTICATED == 10  # req/min
    assert settings.RATE_LIMIT_AUTHENTICATED == 100  # req/min
    assert settings.RATE_LIMIT_ADMIN == 1000  # req/min


def test_settings_file_upload_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test file upload size limits (Spec §FR-001)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.MAX_UPLOAD_SIZE_MB == 50


def test_settings_audit_log_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test audit log retention configuration (Plan §Scale/Scope)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.AUDIT_LOG_RETENTION_YEARS == 7


def test_settings_validates_environment_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ENVIRONMENT only accepts valid values."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")
    monkeypatch.setenv("ENVIRONMENT", "invalid")

    # Act & Assert
    from src.core.config import Settings

    with pytest.raises(ValidationError):
        Settings()


def test_settings_tracing_enabled_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that tracing is disabled by default (Constitution §VI Observability)."""
    # Arrange
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32-characters-long!")
    os.environ.pop("ENABLE_TRACING", None)

    # Act
    from src.core.config import Settings

    settings = Settings()

    # Assert
    assert settings.ENABLE_TRACING is False
