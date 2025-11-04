"""
Database Session Management

SQLAlchemy session factory with WAL mode for concurrent reads.
"""

import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from ace.models.base import Base
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="database")


def _enable_sqlite_wal_mode(dbapi_connection: any, connection_record: any) -> None:
    """Enable WAL mode for SQLite to support concurrent reads.

    Args:
        dbapi_connection: Database API connection object
        connection_record: SQLAlchemy connection record
    """
    if "sqlite" in str(connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def create_db_engine(database_url: str | None = None, echo: bool = False) -> Engine:
    """
    Create SQLAlchemy engine with optimized settings.

    Args:
        database_url: Database URL (defaults to DATABASE_URL env var or ace_playbook.db)
        echo: Enable SQL query logging (default: False)

    Returns:
        SQLAlchemy Engine instance
    """
    if database_url is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///ace_playbook.db")

    logger.info("creating_database_engine", database_url=database_url.split("@")[-1])  # Hide credentials

    engine = create_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,  # Verify connections before use
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
    )

    # Enable WAL mode for SQLite
    if "sqlite" in database_url:
        event.listen(engine, "connect", _enable_sqlite_wal_mode)

    return engine


def create_session_factory(engine: Engine) -> sessionmaker:
    """
    Create SQLAlchemy session factory.

    Args:
        engine: SQLAlchemy engine

    Returns:
        sessionmaker instance
    """
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Global engine and session factory
_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def get_engine(database_url: str | None = None) -> Engine:
    """Get or create global database engine (singleton)."""
    global _engine
    if _engine is None:
        _engine = create_db_engine(database_url=database_url)
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create global session factory (singleton)."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = create_session_factory(engine)
    return _session_factory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Get database session with automatic cleanup.

    Yields:
        SQLAlchemy Session

    Example:
        with get_session() as session:
            tasks = session.query(Task).filter_by(domain_id="customer-acme").all()
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("database_session_error", error=str(e))
        raise
    finally:
        session.close()


def init_database(database_url: str | None = None) -> None:
    """
    Initialize database schema (create all tables).

    Args:
        database_url: Database URL (optional)

    Example:
        init_database()  # Creates all tables if they don't exist
    """
    engine = get_engine(database_url=database_url)
    logger.info("initializing_database_schema")
    Base.metadata.create_all(bind=engine)
    logger.info("database_schema_initialized")
