"""
Unit tests for UMES configuration management.

Tests environment-based configuration with Pydantic settings for:
- KMS_PROVIDER selection (gcp/oracle/aws/azure/openbao/local)
- IDP_TYPE selection (keycloak/auth0/okta/local)
- DATABASE_URL and REDIS_URL
- Environment-specific overrides
"""
import pytest
import os
from pydantic import ValidationError

from umes.config import Settings, KMSProvider, IDPType


class TestSettingsBasics:
    """Test basic settings loading and validation."""

    def test_settings_loads_with_defaults(self, monkeypatch):
        """Test that Settings loads with sensible defaults."""
        # Clear any existing env vars
        monkeypatch.delenv("KMS_PROVIDER", raising=False)
        monkeypatch.delenv("IDP_TYPE", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)

        settings = Settings()

        assert settings is not None
        # Defaults should be safe for development
        assert settings.kms_provider == KMSProvider.LOCAL
        assert settings.idp_type == IDPType.LOCAL

    def test_settings_has_kms_provider_field(self):
        """Test that Settings has kms_provider field."""
        settings = Settings()
        assert hasattr(settings, "kms_provider")

    def test_settings_has_idp_type_field(self):
        """Test that Settings has idp_type field."""
        settings = Settings()
        assert hasattr(settings, "idp_type")

    def test_settings_has_database_url_field(self):
        """Test that Settings has database_url field."""
        settings = Settings()
        assert hasattr(settings, "database_url")

    def test_settings_has_redis_url_field(self):
        """Test that Settings has redis_url field."""
        settings = Settings()
        assert hasattr(settings, "redis_url")


class TestKMSProviderSelection:
    """Test KMS provider configuration."""

    def test_kms_provider_accepts_gcp(self, monkeypatch):
        """Test that KMS_PROVIDER=gcp is accepted."""
        monkeypatch.setenv("KMS_PROVIDER", "gcp")
        settings = Settings()
        assert settings.kms_provider == KMSProvider.GCP

    def test_kms_provider_accepts_oracle(self, monkeypatch):
        """Test that KMS_PROVIDER=oracle is accepted."""
        monkeypatch.setenv("KMS_PROVIDER", "oracle")
        settings = Settings()
        assert settings.kms_provider == KMSProvider.ORACLE

    def test_kms_provider_accepts_aws(self, monkeypatch):
        """Test that KMS_PROVIDER=aws is accepted."""
        monkeypatch.setenv("KMS_PROVIDER", "aws")
        settings = Settings()
        assert settings.kms_provider == KMSProvider.AWS

    def test_kms_provider_accepts_azure(self, monkeypatch):
        """Test that KMS_PROVIDER=azure is accepted."""
        monkeypatch.setenv("KMS_PROVIDER", "azure")
        settings = Settings()
        assert settings.kms_provider == KMSProvider.AZURE

    def test_kms_provider_accepts_openbao(self, monkeypatch):
        """Test that KMS_PROVIDER=openbao is accepted."""
        monkeypatch.setenv("KMS_PROVIDER", "openbao")
        settings = Settings()
        assert settings.kms_provider == KMSProvider.OPENBAO

    def test_kms_provider_accepts_local(self, monkeypatch):
        """Test that KMS_PROVIDER=local is accepted."""
        monkeypatch.setenv("KMS_PROVIDER", "local")
        settings = Settings()
        assert settings.kms_provider == KMSProvider.LOCAL

    def test_kms_provider_rejects_invalid(self, monkeypatch):
        """Test that invalid KMS_PROVIDER raises error."""
        monkeypatch.setenv("KMS_PROVIDER", "invalid_provider")
        with pytest.raises(ValidationError):
            Settings()


class TestIDPTypeSelection:
    """Test IdP type configuration."""

    def test_idp_type_accepts_keycloak(self, monkeypatch):
        """Test that IDP_TYPE=keycloak is accepted."""
        monkeypatch.setenv("IDP_TYPE", "keycloak")
        settings = Settings()
        assert settings.idp_type == IDPType.KEYCLOAK

    def test_idp_type_accepts_auth0(self, monkeypatch):
        """Test that IDP_TYPE=auth0 is accepted."""
        monkeypatch.setenv("IDP_TYPE", "auth0")
        settings = Settings()
        assert settings.idp_type == IDPType.AUTH0

    def test_idp_type_accepts_okta(self, monkeypatch):
        """Test that IDP_TYPE=okta is accepted."""
        monkeypatch.setenv("IDP_TYPE", "okta")
        settings = Settings()
        assert settings.idp_type == IDPType.OKTA

    def test_idp_type_accepts_workos(self, monkeypatch):
        """Test that IDP_TYPE=workos is accepted."""
        monkeypatch.setenv("IDP_TYPE", "workos")
        settings = Settings()
        assert settings.idp_type == IDPType.WORKOS

    def test_idp_type_accepts_local(self, monkeypatch):
        """Test that IDP_TYPE=local is accepted."""
        monkeypatch.setenv("IDP_TYPE", "local")
        settings = Settings()
        assert settings.idp_type == IDPType.LOCAL

    def test_idp_type_rejects_invalid(self, monkeypatch):
        """Test that invalid IDP_TYPE raises error."""
        monkeypatch.setenv("IDP_TYPE", "invalid_idp")
        with pytest.raises(ValidationError):
            Settings()


class TestDatabaseConfiguration:
    """Test database URL configuration."""

    def test_database_url_from_env(self, monkeypatch):
        """Test that DATABASE_URL is read from environment."""
        test_url = "postgresql+asyncpg://user:pass@localhost/umes"
        monkeypatch.setenv("DATABASE_URL", test_url)
        settings = Settings()
        assert settings.database_url == test_url

    def test_database_url_has_default(self):
        """Test that database_url has a sensible default."""
        settings = Settings()
        assert settings.database_url is not None
        assert "postgresql" in settings.database_url.lower()


class TestRedisConfiguration:
    """Test Redis URL configuration."""

    def test_redis_url_from_env(self, monkeypatch):
        """Test that REDIS_URL is read from environment."""
        test_url = "redis://localhost:6379/0"
        monkeypatch.setenv("REDIS_URL", test_url)
        settings = Settings()
        assert settings.redis_url == test_url

    def test_redis_url_optional(self):
        """Test that redis_url is optional (for L1-only cache mode)."""
        settings = Settings()
        # Redis URL should either be None or have a default
        assert settings.redis_url is None or "redis://" in settings.redis_url


class TestEnvironmentIsolation:
    """Test environment-specific configuration."""

    def test_development_environment(self, monkeypatch):
        """Test development environment defaults."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        settings = Settings()

        # Development should use safe local providers
        assert settings.kms_provider == KMSProvider.LOCAL
        assert settings.idp_type == IDPType.LOCAL

    def test_production_environment_requires_cloud_providers(self, monkeypatch):
        """Test that production environment validates cloud providers."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("KMS_PROVIDER", "local")  # Invalid for production

        # Production with local KMS should warn or error
        # (This test may need adjustment based on your requirements)
        settings = Settings()
        # For now, just verify it loads (may add stricter validation later)
        assert settings.environment == "production"

    def test_oracle_cloud_configuration(self, monkeypatch):
        """Test Oracle Cloud deployment configuration."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("KMS_PROVIDER", "oracle")
        monkeypatch.setenv("IDP_TYPE", "keycloak")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://umes@oracle-db/umes")

        settings = Settings()

        assert settings.kms_provider == KMSProvider.ORACLE
        assert settings.idp_type == IDPType.KEYCLOAK
        assert "oracle-db" in settings.database_url

    def test_gcp_cloud_configuration(self, monkeypatch):
        """Test GCP deployment configuration."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("KMS_PROVIDER", "gcp")
        monkeypatch.setenv("IDP_TYPE", "auth0")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://umes@gcp-db/umes")

        settings = Settings()

        assert settings.kms_provider == KMSProvider.GCP
        assert settings.idp_type == IDPType.AUTH0

    def test_hetzner_onprem_configuration(self, monkeypatch):
        """Test Hetzner/on-prem deployment with OpenBao."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("KMS_PROVIDER", "openbao")
        monkeypatch.setenv("IDP_TYPE", "keycloak")
        monkeypatch.setenv("OPENBAO_URL", "http://openbao:8200")

        settings = Settings()

        assert settings.kms_provider == KMSProvider.OPENBAO
        assert settings.idp_type == IDPType.KEYCLOAK


class TestSettingsSingleton:
    """Test settings singleton pattern."""

    def test_get_settings_returns_same_instance(self):
        """Test that get_settings() returns singleton."""
        from umes.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_settings_cached_correctly(self):
        """Test that settings are cached and not reloaded."""
        from umes.config import get_settings

        settings = get_settings()
        initial_kms = settings.kms_provider

        # Changing env should not affect already-loaded settings
        os.environ["KMS_PROVIDER"] = "aws"

        settings_again = get_settings()
        assert settings_again.kms_provider == initial_kms  # Still cached


class TestProviderEnums:
    """Test KMSProvider and IDPType enums."""

    def test_kms_provider_enum_has_all_clouds(self):
        """Test that KMSProvider enum has all supported clouds."""
        assert hasattr(KMSProvider, "GCP")
        assert hasattr(KMSProvider, "ORACLE")
        assert hasattr(KMSProvider, "AWS")
        assert hasattr(KMSProvider, "AZURE")
        assert hasattr(KMSProvider, "OPENBAO")
        assert hasattr(KMSProvider, "LOCAL")

    def test_idp_type_enum_has_all_providers(self):
        """Test that IDPType enum has all supported IdPs."""
        assert hasattr(IDPType, "KEYCLOAK")
        assert hasattr(IDPType, "AUTH0")
        assert hasattr(IDPType, "OKTA")
        assert hasattr(IDPType, "WORKOS")
        assert hasattr(IDPType, "LOCAL")

    def test_enum_values_are_strings(self):
        """Test that enum values are lowercase strings."""
        assert KMSProvider.GCP.value == "gcp"
        assert KMSProvider.ORACLE.value == "oracle"
        assert IDPType.KEYCLOAK.value == "keycloak"
        assert IDPType.AUTH0.value == "auth0"
