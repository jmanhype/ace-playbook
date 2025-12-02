"""
Unit tests for database engine configuration.

Tests the async SQLAlchemy engine setup with connection pooling.
"""
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from umes.database import create_async_engine, get_engine


class TestAsyncEngine:
    """Test async SQLAlchemy engine configuration."""

    @pytest.mark.asyncio
    async def test_create_async_engine_returns_engine(self, postgres_url: str):
        """Test that create_async_engine returns an AsyncEngine instance."""
        engine = create_async_engine(postgres_url)
        assert isinstance(engine, AsyncEngine)
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_engine_has_correct_pool_settings(self, postgres_url: str):
        """Test that engine has correct pool size and overflow settings."""
        engine = create_async_engine(postgres_url)

        # Verify pool configuration per research.md section 6
        assert engine.pool.size() == 20  # pool_size=20
        assert engine.pool._max_overflow == 10  # max_overflow=10

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_engine_can_execute_query(self, postgres_url: str):
        """Test that engine can execute a simple query."""
        engine = create_async_engine(postgres_url)

        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            row = result.fetchone()
            assert row[0] == 1

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_get_engine_returns_singleton(self, postgres_url: str):
        """Test that get_engine returns the same engine instance."""
        engine1 = get_engine()
        engine2 = get_engine()

        assert engine1 is engine2

        await engine1.dispose()

    @pytest.mark.asyncio
    async def test_engine_uses_asyncpg_driver(self, postgres_url: str):
        """Test that engine uses asyncpg driver as specified."""
        engine = create_async_engine(postgres_url)

        assert "asyncpg" in str(engine.url)

        await engine.dispose()
