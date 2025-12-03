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

    @pytest.mark.asyncio
    async def test_rls_on_connect_executes_row_security_setup(self, postgres_url: str):
        """Test that _set_rls_on_connect executes SET row_security = on."""
        from umes.database import create_async_engine

        # Create engine with RLS listener
        engine = create_async_engine(postgres_url)
        setup_rls_listener(engine)

        # Get a connection - this should trigger the connect event
        async with engine.connect() as conn:
            # Verify row_security is enabled
            result = await conn.execute(text("SHOW row_security"))
            row_security_value = result.scalar()
            assert row_security_value == "on"

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_tenant_on_begin_cursor_cleanup(self, postgres_url: str):
        """Test that _set_tenant_on_begin properly closes cursor after setting tenant."""
        from umes.database import create_async_engine

        engine = create_async_engine(postgres_url)
        setup_rls_listener(engine)

        # Set tenant context and begin transaction
        token = set_tenant_context("tenant-cursor-test")

        # Begin transaction - should trigger _set_tenant_on_begin with cursor operations
        async with engine.begin() as conn:
            # Verify the tenant was set (cursor was used and closed)
            result = await conn.execute(text("SHOW app.current_tenant_id"))
            assert result.scalar() == "tenant-cursor-test"

            # Execute another query to ensure connection is still healthy after cursor.close()
            result2 = await conn.execute(text("SELECT 1"))
            assert result2.scalar() == 1

        await engine.dispose()
        clear_tenant_context(token)

    def test_set_rls_on_connect_function_directly(self):
        """Test _set_rls_on_connect function directly with mock connection."""
        from umes.middleware.tenant_context import _set_rls_on_connect
        from unittest.mock import Mock

        # Create mock DBAPI connection
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor

        # Call function directly
        _set_rls_on_connect(mock_conn, None)

        # Verify cursor operations were called
        mock_conn.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SET row_security = on")
        mock_cursor.close.assert_called_once()

    def test_set_tenant_on_begin_function_directly(self):
        """Test _set_tenant_on_begin function directly with tenant context set."""
        from umes.middleware.tenant_context import _set_tenant_on_begin, set_tenant_context, clear_tenant_context
        from unittest.mock import Mock

        # Set tenant context
        token = set_tenant_context("tenant-direct-test")

        # Create mock connection with nested structure
        mock_cursor = Mock()
        mock_dbapi_conn = Mock()
        mock_dbapi_conn.cursor.return_value = mock_cursor
        mock_connection = Mock()
        mock_connection.dbapi_connection = mock_dbapi_conn
        mock_conn = Mock()
        mock_conn.connection = mock_connection

        # Call function directly
        _set_tenant_on_begin(mock_conn)

        # Verify cursor operations were called
        mock_dbapi_conn.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once()
        mock_cursor.close.assert_called_once()

        # Cleanup
        clear_tenant_context(token)
