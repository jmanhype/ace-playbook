"""
Unit tests for database engine configuration.

Tests the async SQLAlchemy engine setup with connection pooling.
"""
import os
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from umes.database import create_async_engine, get_engine, dispose_engine
import umes.database


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

    def test_get_engine_raises_when_no_database_url(self, monkeypatch):
        """Test that get_engine raises ValueError when DATABASE_URL not set."""
        # Reset singleton to force re-initialization
        umes.database._engine = None

        # Remove DATABASE_URL from environment
        monkeypatch.delenv("DATABASE_URL", raising=False)

        # Should raise ValueError
        with pytest.raises(ValueError, match="DATABASE_URL environment variable must be set"):
            get_engine()

    @pytest.mark.asyncio
    async def test_dispose_engine_closes_connections(self, postgres_url: str, monkeypatch):
        """Test that dispose_engine disposes of engine and resets singleton."""
        # Reset singleton
        umes.database._engine = None

        # Set DATABASE_URL in environment
        monkeypatch.setenv("DATABASE_URL", postgres_url)

        # Get engine (creates singleton)
        engine = get_engine()
        assert engine is not None
        assert umes.database._engine is engine

        # Dispose of engine
        await dispose_engine()

        # Verify engine was disposed and singleton reset
        assert umes.database._engine is None

    @pytest.mark.asyncio
    async def test_dispose_engine_when_no_engine_exists(self):
        """Test that dispose_engine handles case when no engine exists."""
        # Reset singleton
        umes.database._engine = None

        # Should not raise exception
        await dispose_engine()

        # Singleton should still be None
        assert umes.database._engine is None
