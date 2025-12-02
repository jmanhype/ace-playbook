"""
Unit tests for tenant context management (Row-Level Security).

Tests the ContextVar-based tenant context and SQLAlchemy RLS integration.
"""
import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine

from umes.middleware.tenant_context import (
    current_tenant_id,
    set_tenant_context,
    get_tenant_context,
    clear_tenant_context,
    setup_rls_listener,
)


class TestTenantContext:
    """Test tenant context ContextVar operations."""

    def test_set_and_get_tenant_context(self):
        """Test setting and retrieving tenant context."""
        test_tenant_id = "tenant-123"
        token = set_tenant_context(test_tenant_id)

        assert get_tenant_context() == test_tenant_id

        # Cleanup
        clear_tenant_context(token)

    def test_get_tenant_context_returns_none_when_not_set(self):
        """Test get returns None when no context set."""
        # Ensure clean state
        clear_tenant_context()

        assert get_tenant_context() is None

    def test_clear_tenant_context_resets_to_none(self):
        """Test clearing context resets to None."""
        set_tenant_context("tenant-456")
        clear_tenant_context()

        assert get_tenant_context() is None

    def test_context_isolation_between_calls(self):
        """Test that context changes don't leak between operations."""
        token1 = set_tenant_context("tenant-A")
        assert get_tenant_context() == "tenant-A"

        token2 = set_tenant_context("tenant-B")
        assert get_tenant_context() == "tenant-B"

        # Reset to previous context
        current_tenant_id.reset(token2)
        assert get_tenant_context() == "tenant-A"

        # Cleanup
        clear_tenant_context(token1)


class TestRLSListener:
    """Test SQLAlchemy RLS event listener."""

    @pytest.mark.asyncio
    async def test_setup_rls_listener_registers_event(self, postgres_url: str):
        """Test that RLS listener is registered on engine."""
        from umes.database import create_async_engine
        from umes.middleware.tenant_context import _set_rls_on_connect, _set_tenant_on_begin

        engine = create_async_engine(postgres_url)
        setup_rls_listener(engine)

        # Verify event listeners are registered
        assert event.contains(engine.sync_engine, "connect", _set_rls_on_connect)
        assert event.contains(engine.sync_engine, "begin", _set_tenant_on_begin)

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_rls_sets_current_tenant_on_begin(self, postgres_url: str):
        """Test that SET LOCAL is executed when transaction begins."""
        from umes.database import create_async_engine

        engine = create_async_engine(postgres_url)
        setup_rls_listener(engine)

        # Set tenant context
        token = set_tenant_context("tenant-789")

        # Begin transaction and verify SET LOCAL was called
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SHOW app.current_tenant_id")
            )
            tenant_value = result.scalar()
            assert tenant_value == "tenant-789"

        await engine.dispose()
        clear_tenant_context(token)

    @pytest.mark.asyncio
    async def test_rls_does_not_set_when_no_context(self, postgres_url: str):
        """Test that SET LOCAL is not called when no tenant context."""
        from umes.database import create_async_engine

        engine = create_async_engine(postgres_url)
        setup_rls_listener(engine)

        # Ensure no context
        clear_tenant_context()

        # Begin transaction - should not error without tenant context
        async with engine.begin() as conn:
            # This should work without tenant context
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_rls_context_isolation_between_transactions(self, postgres_url: str):
        """Test that different transactions can have different tenant contexts."""
        from umes.database import create_async_engine

        engine = create_async_engine(postgres_url)
        setup_rls_listener(engine)

        # Transaction 1 with tenant-A
        token1 = set_tenant_context("tenant-A")
        async with engine.begin() as conn1:
            result = await conn1.execute(text("SHOW app.current_tenant_id"))
            assert result.scalar() == "tenant-A"

        # Transaction 2 with tenant-B
        current_tenant_id.reset(token1)
        token2 = set_tenant_context("tenant-B")
        async with engine.begin() as conn2:
            result = await conn2.execute(text("SHOW app.current_tenant_id"))
            assert result.scalar() == "tenant-B"

        await engine.dispose()
        clear_tenant_context(token2)
