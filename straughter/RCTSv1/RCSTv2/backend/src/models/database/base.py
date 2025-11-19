"""
Base SQLAlchemy models and mixins.

Implements Constitution §II (SOLID) and §IV (Compliance - ALCOA+).
Provides base classes with common patterns:
- Timestamp tracking (ALCOA+ Contemporaneous)
- Tenant isolation (Constitution §III Security)
- Soft deletion
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Provides declarative base and common type annotations.
    """

    pass


class TimestampMixin:
    """
    Mixin for created_at and updated_at timestamps.

    Implements ALCOA+ Contemporaneous principle (Constitution §IV):
    - created_at: Auto-set on INSERT
    - updated_at: Auto-updated on UPDATE

    Both use UTC timezone and ISO 8601 format (Spec §FR-010).
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="ALCOA+ Contemporaneous: Record creation timestamp (UTC)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="ALCOA+ Contemporaneous: Record update timestamp (UTC)",
    )


class TenantMixin:
    """
    Mixin for tenant_id foreign key.

    Implements multi-tenant isolation (Constitution §III, Spec §FR-013).
    All tenant-scoped tables must include this mixin.

    Row-Level Security (RLS) policies enforce tenant isolation:
    - FORCE ROW LEVEL SECURITY enabled on all tenant-scoped tables
    - Policies filter by tenant_id automatically
    """

    tenant_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        comment="Multi-tenant isolation: Tenant UUID",
    )


class SoftDeleteMixin:
    """
    Mixin for soft deletion.

    Instead of hard deletes, sets deleted_at timestamp.
    Queries should filter WHERE deleted_at IS NULL by default.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="Soft deletion timestamp (NULL = active)",
    )

    @property
    def is_deleted(self) -> bool:
        """Check if record is soft-deleted."""
        return self.deleted_at is not None


class AuditableMixin(TimestampMixin):
    """
    Mixin for auditable entities.

    Extends TimestampMixin with user tracking for ALCOA+ Attributable.

    Tracks:
    - created_by: User ID who created the record
    - updated_by: User ID who last updated the record
    """

    created_by: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        comment="ALCOA+ Attributable: User who created record",
    )
    updated_by: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        comment="ALCOA+ Attributable: User who last updated record",
    )


class BaseModel(Base, TimestampMixin):
    """
    Base model with UUID primary key and timestamps.

    All domain models should inherit from this base.
    Provides:
    - UUID primary key (UUID v4)
    - created_at, updated_at timestamps (ALCOA+)
    """

    __abstract__ = True

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Primary key: UUID v4",
    )


class TenantScopedModel(BaseModel, TenantMixin, SoftDeleteMixin):
    """
    Base model for tenant-scoped entities.

    Includes:
    - UUID primary key
    - Timestamps (ALCOA+)
    - Tenant isolation
    - Soft deletion

    Use this for all entities that belong to a specific tenant.
    """

    __abstract__ = True


class ImmutableModel(Base, AuditableMixin):
    """
    Base model for immutable audit records.

    Implements ALCOA+ Enduring and Original principles:
    - No UPDATE or DELETE allowed (enforced by database triggers)
    - Append-only
    - Full audit metadata

    Use this for:
    - ProvenanceRecord
    - AuditLogEntry
    """

    __abstract__ = True

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Primary key: UUID v4 (immutable)",
    )
