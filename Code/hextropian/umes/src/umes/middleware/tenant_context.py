"""
Tenant context management for Row-Level Security (RLS).

This module provides ContextVar-based tenant context management and SQLAlchemy
event listeners to enforce PostgreSQL Row-Level Security policies.

Usage:
    # Set tenant context before database operations
    token = set_tenant_context("tenant-abc123")

    # Query database - RLS automatically applied
    async with db_session() as session:
        users = await session.execute(select(User))  # Only tenant's users returned

    # Clear context when done
    clear_tenant_context(token)

Architecture:
    - ContextVar provides async-safe tenant context storage
    - SQLAlchemy "after_begin" event listener sets PostgreSQL session variable
    - PostgreSQL RLS policies use the session variable for filtering

References:
    - research.md section 3: Row-Level Security implementation
    - PostgreSQL RLS: https://www.postgresql.org/docs/current/ddl-rowsecurity.html
"""

from contextvars import ContextVar, Token
from typing import Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection


# Thread-safe, async-safe context variable for current tenant ID
current_tenant_id: ContextVar[Optional[str]] = ContextVar(
    "current_tenant_id", default=None
)


def set_tenant_context(tenant_id: str) -> Token[Optional[str]]:
    """Set the current tenant context.

    Args:
        tenant_id: The tenant ID to set as current context

    Returns:
        Token that can be used to reset the context

    Example:
        >>> token = set_tenant_context("tenant-123")
        >>> # Do work with tenant context
        >>> clear_tenant_context(token)
    """
    return current_tenant_id.set(tenant_id)


def get_tenant_context() -> Optional[str]:
    """Get the current tenant context.

    Returns:
        Current tenant ID, or None if not set

    Example:
        >>> tenant_id = get_tenant_context()
        >>> if tenant_id:
        ...     # Perform tenant-scoped operation
    """
    return current_tenant_id.get()


def clear_tenant_context(token: Optional[Token[Optional[str]]] = None) -> None:
    """Clear the current tenant context.

    Args:
        token: Optional token from set_tenant_context to reset to previous value.
               If None, clears to default (None).

    Example:
        >>> token = set_tenant_context("tenant-123")
        >>> clear_tenant_context(token)  # Reset to previous value
        >>> clear_tenant_context()  # Clear to None
    """
    if token is not None:
        current_tenant_id.reset(token)
    else:
        current_tenant_id.set(None)


def _set_rls_on_connect(dbapi_conn, connection_record):
    """Set up RLS configuration on new database connections.

    This runs once per connection to configure PostgreSQL parameters.

    Args:
        dbapi_conn: Raw DBAPI connection
        connection_record: SQLAlchemy connection record
    """
    # Enable row security for the connection
    cursor = dbapi_conn.cursor()
    cursor.execute("SET row_security = on")
    cursor.close()


def _set_tenant_on_begin(conn):
    """Set tenant context at the start of each transaction.

    This SQLAlchemy event listener is triggered on "begin" and sets the
    PostgreSQL session variable that RLS policies use for filtering.

    Args:
        conn: SQLAlchemy connection

    Note:
        Uses SET LOCAL so the variable is scoped to the transaction and
        automatically cleared when the transaction ends.
    """
    tenant_id = get_tenant_context()

    if tenant_id is not None:
        # SET LOCAL ensures the variable is transaction-scoped
        # It will be cleared when the transaction commits or rolls back
        # Use raw connection to execute
        dbapi_conn = conn.connection.dbapi_connection
        cursor = dbapi_conn.cursor()
        cursor.execute(f"SET LOCAL app.current_tenant_id = '{tenant_id}'")
        cursor.close()


def setup_rls_listener(engine: AsyncEngine) -> None:
    """Register RLS event listeners on the database engine.

    This should be called once during application startup after creating
    the engine.

    Args:
        engine: SQLAlchemy async engine

    Example:
        >>> from umes.database import create_async_engine
        >>> engine = create_async_engine(database_url)
        >>> setup_rls_listener(engine)

    Note:
        - "connect" event sets up the connection for RLS
        - "begin" event sets tenant context for each transaction
    """
    # Register connect event on the sync engine (connections happen synchronously)
    event.listen(engine.sync_engine, "connect", _set_rls_on_connect)

    # Register begin event on the sync engine pool
    # This fires when a transaction begins
    event.listen(engine.sync_engine, "begin", _set_tenant_on_begin)
