"""
Pytest configuration and shared fixtures.

Implements Constitution §I (TDD) with comprehensive test fixtures.
Provides fixtures for database, authentication, and multi-tenant testing.
"""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment before importing application code
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only-32chars!")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")  # Use separate test DB


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Configure async backend for pytest-asyncio."""
    return "asyncio"


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Get test database URL (in-memory SQLite for tests)."""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def sync_test_database_url() -> str:
    """Get synchronous test database URL (for Alembic migrations)."""
    return "sqlite:///:memory:"


@pytest.fixture(scope="function")
async def async_db_engine(test_database_url: str) -> AsyncGenerator[Any, None]:
    """
    Create async database engine for tests.

    Uses in-memory SQLite with StaticPool to ensure database persists
    across connections within a single test.
    """
    engine = create_async_engine(
        test_database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture(scope="function")
def sync_db_engine(sync_test_database_url: str) -> Generator[Any, None, None]:
    """
    Create synchronous database engine for tests.

    Used for running Alembic migrations in tests.
    """
    engine = create_engine(
        sync_test_database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    # Enable foreign key constraints in SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):  # type: ignore
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="function")
async def async_db_session(async_db_engine: Any) -> AsyncGenerator[AsyncSession, None]:
    """
    Create async database session for tests.

    Automatically rolls back transactions after each test to ensure isolation.
    """
    # Import Base model here to avoid circular imports
    from src.models.database.base import Base

    # Create all tables
    async with async_db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session factory
    async_session_factory = async_sessionmaker(
        async_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        try:
            yield session
            await session.rollback()
        finally:
            await session.close()

    # Drop all tables after test
    async with async_db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="function")
def sync_db_session(sync_db_engine: Any) -> Generator[Session, None, None]:
    """
    Create synchronous database session for tests.

    Used for tests that require synchronous database access.
    """
    from src.models.database.base import Base

    # Create all tables
    Base.metadata.create_all(bind=sync_db_engine)

    # Create session factory
    Session = sessionmaker(bind=sync_db_engine)
    session = Session()

    try:
        yield session
        session.rollback()
    finally:
        session.close()

    # Drop all tables after test
    Base.metadata.drop_all(bind=sync_db_engine)


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    """
    Create FastAPI test client.

    Overrides database dependency to use test database.
    """
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def test_tenant_id() -> str:
    """Get test tenant ID for multi-tenant tests."""
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture(scope="function")
def test_user_id() -> str:
    """Get test user ID."""
    return "00000000-0000-0000-0000-000000000002"


@pytest.fixture(scope="function")
def auth_headers(test_user_id: str, test_tenant_id: str) -> dict[str, str]:
    """
    Create authentication headers for tests.

    Returns JWT token in Authorization header.
    """
    from datetime import datetime, timedelta

    from jose import jwt

    from src.core.config import get_settings

    settings = get_settings()

    # Create JWT token
    expires = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    payload = {
        "sub": test_user_id,
        "tenant_id": test_tenant_id,
        "exp": expires,
        "roles": ["Viewer", "Editor"],
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def admin_auth_headers(test_user_id: str, test_tenant_id: str) -> dict[str, str]:
    """
    Create admin authentication headers for tests.

    Returns JWT token with SystemAdmin role.
    """
    from datetime import datetime, timedelta

    from jose import jwt

    from src.core.config import get_settings

    settings = get_settings()

    expires = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    payload = {
        "sub": test_user_id,
        "tenant_id": test_tenant_id,
        "exp": expires,
        "roles": ["SystemAdmin"],
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Generator[None, None, None]:
    """
    Reset settings cache between tests.

    Ensures each test gets fresh settings from environment.
    """
    from src.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
