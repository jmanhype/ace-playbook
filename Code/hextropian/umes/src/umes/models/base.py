"""
Base SQLAlchemy models and mixins for UMES.

Provides:
- Base: Declarative base for all models
- TimestampMixin: Automatic created_at/updated_at tracking
- SoftDeleteMixin: Soft delete support with deleted_at
- TenantMixin: Multi-tenancy via tenant_id (for RLS)

Usage:
    from umes.models.base import Base, TimestampMixin, SoftDeleteMixin, TenantMixin

    class User(Base, TimestampMixin, SoftDeleteMixin, TenantMixin):
        __tablename__ = "users"

        id = Column(Integer, primary_key=True)
        email = Column(String, nullable=False)
"""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, String, Index, event
from sqlalchemy.orm import declarative_base, DeclarativeBase


# Declarative base for all UMES models
Base = declarative_base()


class TimestampMixin:
    """Mixin to add automatic timestamp tracking.

    Adds:
        - created_at: Set once on insert (NOT NULL)
        - updated_at: Updated automatically on every modification (NOT NULL)

    The updated_at column is automatically updated via SQLAlchemy event listener.
    """

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SoftDeleteMixin:
    """Mixin to add soft delete support.

    Adds:
        - deleted_at: Timestamp when soft deleted (NULL = not deleted)
        - is_deleted: Property to check if object is soft deleted
        - soft_delete(): Method to mark object as deleted

    Soft deleted objects remain in database but are marked with deleted_at timestamp.
    Application logic should filter out soft deleted objects in queries.
    """

    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    @property
    def is_deleted(self) -> bool:
        """Check if this object is soft deleted.

        Returns:
            True if deleted_at is set, False otherwise
        """
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        """Mark this object as soft deleted.

        Sets deleted_at to current UTC timestamp.
        Does not commit - caller must commit the session.
        """
        self.deleted_at = datetime.now(timezone.utc)


class TenantMixin:
    """Mixin to add multi-tenancy support via tenant_id.

    Adds:
        - tenant_id: String column for tenant isolation (NOT NULL, indexed)

    This column is used by PostgreSQL Row-Level Security (RLS) policies
    to enforce tenant isolation at the database level.

    The tenant_id is set via middleware context (see middleware/tenant_context.py)
    and enforced via RLS policies on each table.
    """

    tenant_id = Column(
        String(255),
        nullable=False,
        index=True,  # Index for query performance
    )


# Register event listener to ensure updated_at is always current
@event.listens_for(Base, "before_update", propagate=True)
def receive_before_update(mapper, connection, target):
    """SQLAlchemy event listener to update updated_at on modification.

    This ensures updated_at is always set to current time on any update,
    even if the model doesn't explicitly set it.

    Args:
        mapper: SQLAlchemy mapper
        connection: Database connection
        target: The model instance being updated
    """
    if hasattr(target, "updated_at"):
        target.updated_at = datetime.now(timezone.utc)
