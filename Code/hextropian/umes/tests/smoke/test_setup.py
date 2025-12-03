"""
Smoke tests for UMES project setup.

Verifies that basic project infrastructure is working:
- All modules can be imported
- Database connection can be established
- Redis connection works (if configured)
- Configuration loads correctly

These are fast, high-level tests to catch setup issues early.
Run with: pytest tests/smoke/ -v
"""
import pytest
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class TestImports:
    """Test that all core modules can be imported."""

    def test_import_config(self):
        """Test that config module imports successfully."""
        from umes.config import Settings, KMSProvider, IDPType, get_settings

        assert Settings is not None
        assert KMSProvider is not None
        assert IDPType is not None
        assert get_settings is not None

    def test_import_database(self):
        """Test that database module imports successfully."""
        from umes.database import create_async_engine, get_engine, dispose_engine

        assert create_async_engine is not None
        assert get_engine is not None
        assert dispose_engine is not None

    def test_import_cache(self):
        """Test that cache module imports successfully."""
        from umes.utils.cache import TieredCache

        assert TieredCache is not None

    def test_import_circuit_breaker(self):
        """Test that circuit breaker module imports successfully."""
        from umes.utils.circuit_breaker import (
            AsyncCircuitBreakerFactory,
            CircuitBreakerWrapper,
        )

        assert AsyncCircuitBreakerFactory is not None
        assert CircuitBreakerWrapper is not None

    def test_import_models(self):
        """Test that model base classes import successfully."""
        from umes.models.base import Base, TimestampMixin, SoftDeleteMixin, TenantMixin

        assert Base is not None
        assert TimestampMixin is not None
        assert SoftDeleteMixin is not None
        assert TenantMixin is not None

    def test_import_tenant_context(self):
        """Test that tenant context middleware imports successfully."""
        from umes.middleware.tenant_context import (
            set_tenant_context,
            get_tenant_context,
            clear_tenant_context,
            setup_rls_listener,
        )

        assert set_tenant_context is not None
        assert get_tenant_context is not None
        assert clear_tenant_context is not None
        assert setup_rls_listener is not None


class TestConfigLoading:
    """Test that configuration loads successfully."""

    def test_config_loads_with_defaults(self, monkeypatch):
        """Test that configuration loads with default values."""
        # Clear environment to ensure defaults
        monkeypatch.delenv("KMS_PROVIDER", raising=False)
        monkeypatch.delenv("IDP_TYPE", raising=False)

        from umes.config import Settings, KMSProvider, IDPType

        settings = Settings()

        # Verify defaults (test environment from pytest.ini)
        assert settings is not None
        assert settings.kms_provider == KMSProvider.LOCAL
        assert settings.idp_type == IDPType.LOCAL
        # pytest.ini sets ENVIRONMENT=test
        assert settings.environment in ("test", "development")
        assert settings.database_url is not None
        assert "postgresql" in settings.database_url.lower()

    def test_get_settings_singleton(self):
        """Test that get_settings returns singleton instance."""
        from umes.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2


@pytest.mark.asyncio
class TestDatabaseConnection:
    """Test that database connection works."""

    async def test_database_engine_creation(self):
        """Test that async database engine can be created."""
        from umes.database import create_async_engine

        test_url = "postgresql+asyncpg://test:test@localhost/test"
        engine = create_async_engine(test_url)

        assert engine is not None
        assert isinstance(engine, AsyncEngine)

    async def test_database_connection_with_testcontainer(self, postgres_container):
        """Test actual database connection with testcontainer."""
        from umes.database import create_async_engine

        # Get connection URL from testcontainer and convert to asyncpg
        # Same approach as postgres_url fixture in conftest.py
        url = postgres_container.get_connection_url()
        async_url = url.replace("postgresql://", "postgresql+asyncpg://").replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://"
        )

        engine = create_async_engine(async_url)

        # Try to connect and execute simple query
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            assert row[0] == 1

        # Clean up
        await engine.dispose()


@pytest.mark.asyncio
class TestRedisConnection:
    """Test that Redis connection works (if configured)."""

    async def test_redis_connection_optional(self):
        """Test that Redis URL is optional (L1-only cache mode)."""
        from umes.utils.cache import TieredCache

        # Create cache without Redis
        cache = TieredCache(max_l1_size=100, redis_url=None)

        assert cache is not None
        assert cache.redis_url is None

        # Should be able to connect
        await cache.connect()

        # L1 cache should work
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"

        # Clean up
        await cache.close()

    async def test_redis_connection_with_testcontainer(self, redis_container):
        """Test actual Redis connection with testcontainer."""
        from umes.utils.cache import TieredCache

        # Get Redis URL from testcontainer fixture
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}/0"

        # Create cache with Redis
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)

        assert cache is not None
        await cache.connect()

        # Test L1 and L2 cache
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"

        # Clean up
        await cache.close()


@pytest.mark.asyncio
class TestCircuitBreakerSetup:
    """Test that circuit breakers initialize correctly."""

    async def test_circuit_breaker_factory_creation(self):
        """Test that circuit breaker factory can be created."""
        from umes.utils.circuit_breaker import AsyncCircuitBreakerFactory

        factory = AsyncCircuitBreakerFactory(
            failure_threshold=5, timeout_seconds=60
        )

        assert factory is not None
        assert factory.failure_threshold == 5
        assert factory.timeout_seconds == 60.0

    async def test_kms_circuit_breaker_creation(self):
        """Test that KMS circuit breaker can be created."""
        from umes.utils.circuit_breaker import AsyncCircuitBreakerFactory

        factory = AsyncCircuitBreakerFactory()
        kms_breaker = await factory.create_kms_breaker()

        assert kms_breaker is not None
        assert kms_breaker.is_closed

    async def test_idp_circuit_breaker_creation(self):
        """Test that IdP circuit breaker can be created."""
        from umes.utils.circuit_breaker import AsyncCircuitBreakerFactory

        factory = AsyncCircuitBreakerFactory()
        idp_breaker = await factory.create_idp_breaker()

        assert idp_breaker is not None
        assert idp_breaker.is_closed
