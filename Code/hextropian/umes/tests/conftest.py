"""
Pytest configuration and fixtures for UMES tests.

Provides session-scoped Testcontainers for PostgreSQL, Redis, and Keycloak.
Per research.md section 5: Testing Strategy.
"""
import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

# Set event loop policy for Windows compatibility
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ============================================================================
# Session-scoped container fixtures
# ============================================================================


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """
    Session-scoped PostgreSQL container with extensions.

    Provides PostgreSQL 17 with pgvecto.rs, Apache AGE, and TimescaleDB.
    Custom image from /PostgresWithExtensions.
    Container is started once and shared across all tests.
    """
    with PostgresContainer(
        image="postgres-extensions:17",
        username="umes",
        password="umes",
        dbname="umes_test",
    ) as postgres:
        # Wait for container to be ready
        postgres.get_connection_url()
        yield postgres


@pytest.fixture(scope="session")
def redis_container() -> Generator[RedisContainer, None, None]:
    """
    Session-scoped Redis container.

    Provides a clean Redis 7+ instance for caching and token revocation tests.
    """
    with RedisContainer(image="redis:7-alpine") as redis:
        yield redis


@pytest.fixture(scope="session")
def postgres_url(postgres_container: PostgresContainer) -> str:
    """
    Get PostgreSQL async connection URL.

    Returns asyncpg-compatible connection string.
    """
    # Convert psycopg2 URL to asyncpg URL
    url = postgres_container.get_connection_url()
    return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")


@pytest.fixture(scope="session")
def redis_url(redis_container: RedisContainer) -> str:
    """Get Redis connection URL."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}/0"


# ============================================================================
# Database engine and session fixtures
# ============================================================================


@pytest.fixture(scope="session")
def async_engine(postgres_url: str) -> AsyncEngine:
    """
    Create async SQLAlchemy engine for tests.

    Engine is created once per session and shared across tests.
    """
    engine = create_async_engine(
        postgres_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    return engine


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional database session for each test.

    Each test gets a fresh transaction that is rolled back after the test completes.
    This ensures test isolation without recreating the entire database.
    """
    # Create session factory
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Start transaction
    async with async_session_maker() as session:
        async with session.begin():
            yield session
            # Transaction is rolled back automatically


# ============================================================================
# Event loop fixture (function-scoped)
# ============================================================================


@pytest.fixture(scope="function")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create a new event loop for each test function.

    Required for pytest-asyncio to work correctly with function-scoped fixtures.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Environment configuration fixtures
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_env(postgres_url: str, redis_url: str) -> None:
    """
    Configure environment variables for test execution.

    Auto-used fixture that sets up test environment before any tests run.
    """
    os.environ.update({
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": postgres_url,
        "REDIS_URL": redis_url,
        "JWT_SECRET_KEY": "test-secret-key-32-characters-minimum-length",
        "JWT_ALGORITHM": "HS256",
        "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "30",
        "JWT_REFRESH_TOKEN_EXPIRE_DAYS": "7",
    })


# ============================================================================
# Utility fixtures
# ============================================================================


@pytest.fixture
def sample_user_data() -> dict:
    """Provide sample user data for tests."""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "SecureP@ssw0rd123",
        "first_name": "Test",
        "last_name": "User",
    }


@pytest.fixture
def sample_tenant_data() -> dict:
    """Provide sample tenant data for tests."""
    return {
        "name": "Test Tenant",
        "slug": "test-tenant",
        "is_active": True,
    }


# ============================================================================
# Markers configuration
# ============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (require containers)")
    config.addinivalue_line("markers", "smoke: Smoke tests (basic functionality checks)")
    config.addinivalue_line("markers", "contract: Contract/API tests (OpenAPI compliance)")
    config.addinivalue_line("markers", "security: Security tests (penetration, fuzzing)")
    config.addinivalue_line("markers", "performance: Performance/load tests (slow)")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "kms: Tests requiring KMS adapters")
    config.addinivalue_line("markers", "idp: Tests requiring IdP adapters")
    config.addinivalue_line("markers", "rls: Tests for Row-Level Security")
    config.addinivalue_line("markers", "audit: Tests for audit logging")
