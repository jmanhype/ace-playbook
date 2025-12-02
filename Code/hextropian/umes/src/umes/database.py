"""Database engine configuration and management.

This module provides async SQLAlchemy engine creation and singleton management
for UMES database connections.
"""

import os
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine as sa_create_async_engine


# Global engine singleton
_engine: Optional[AsyncEngine] = None


def create_async_engine(url: str) -> AsyncEngine:
    """Create async SQLAlchemy engine with connection pooling.

    Args:
        url: Database connection URL (must use asyncpg driver for PostgreSQL)

    Returns:
        Configured AsyncEngine instance with optimized pool settings

    Example:
        >>> engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
        >>> async with engine.begin() as conn:
        ...     await conn.execute("SELECT 1")
    """
    return sa_create_async_engine(
        url,
        echo=False,  # Disable SQL logging in production
        pool_pre_ping=True,  # Verify connections before using
        pool_size=20,  # Base connection pool size
        max_overflow=10,  # Additional connections under load
        pool_recycle=3600,  # Recycle connections after 1 hour
    )


def get_engine() -> AsyncEngine:
    """Get singleton engine instance.

    Creates engine on first call using DATABASE_URL environment variable.
    Subsequent calls return the same engine instance.

    Returns:
        Singleton AsyncEngine instance

    Raises:
        ValueError: If DATABASE_URL environment variable is not set

    Example:
        >>> engine = get_engine()
        >>> # Same engine instance returned on subsequent calls
        >>> assert get_engine() is engine
    """
    global _engine

    if _engine is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable must be set")
        _engine = create_async_engine(database_url)

    return _engine


async def dispose_engine() -> None:
    """Dispose of the global engine instance.

    Closes all connections in the pool and resets the singleton.
    Should be called during application shutdown.

    Example:
        >>> await dispose_engine()
    """
    global _engine

    if _engine is not None:
        await _engine.dispose()
        _engine = None
