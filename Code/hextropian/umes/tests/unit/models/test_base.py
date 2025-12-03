"""
Unit tests for base SQLAlchemy models.

Tests the Base declarative class, mixins (timestamp, soft delete, tenant),
and RLS support via tenant_id column.
"""
import pytest
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, select, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from umes.models.base import (
    Base,
    TimestampMixin,
    SoftDeleteMixin,
    TenantMixin,
    receive_before_update,
)


# Test model combining all mixins
class TestModel(Base, TimestampMixin, SoftDeleteMixin, TenantMixin):
    """Test model with all mixins for integration testing."""

    __tablename__ = "test_models"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)


# Test model WITHOUT TimestampMixin (to test event listener branch)
class TestModelNoTimestamp(Base, TenantMixin):
    """Test model without TimestampMixin for event listener branch testing."""

    __tablename__ = "test_models_no_timestamp"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)


class TestBaseDeclarative:
    """Test Base declarative class configuration."""

    def test_base_has_metadata(self):
        """Test that Base has metadata attribute."""
        assert hasattr(Base, "metadata")
        assert Base.metadata is not None

    def test_base_can_create_table_definition(self):
        """Test that models using Base can define tables."""
        assert TestModel.__tablename__ == "test_models"
        assert "id" in TestModel.__table__.columns
        assert "name" in TestModel.__table__.columns


class TestTimestampMixin:
    """Test timestamp mixin functionality."""

    def test_timestamp_mixin_has_created_at(self):
        """Test that TimestampMixin adds created_at column."""
        assert hasattr(TestModel, "created_at")
        assert "created_at" in TestModel.__table__.columns

    def test_timestamp_mixin_has_updated_at(self):
        """Test that TimestampMixin adds updated_at column."""
        assert hasattr(TestModel, "updated_at")
        assert "updated_at" in TestModel.__table__.columns

    def test_created_at_not_nullable(self):
        """Test that created_at is NOT NULL."""
        col = TestModel.__table__.columns["created_at"]
        assert col.nullable is False

    def test_updated_at_not_nullable(self):
        """Test that updated_at is NOT NULL."""
        col = TestModel.__table__.columns["updated_at"]
        assert col.nullable is False

    @pytest.mark.asyncio
    async def test_created_at_set_on_insert(self, postgres_url: str):
        """Test that created_at is automatically set on insert."""
        engine = create_async_engine(postgres_url)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async with async_session() as session:
            test_obj = TestModel(name="Test", tenant_id="tenant-1")
            session.add(test_obj)
            await session.commit()

            # created_at should be set automatically
            assert test_obj.created_at is not None
            assert isinstance(test_obj.created_at, datetime)

            # Should be recent (within last 5 seconds)
            now = datetime.now(timezone.utc)
            time_diff = (now - test_obj.created_at.replace(tzinfo=timezone.utc)).total_seconds()
            assert time_diff < 5

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_updated_at_changes_on_update(self, postgres_url: str):
        """Test that updated_at is automatically updated on modification."""
        engine = create_async_engine(postgres_url)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async with async_session() as session:
            # Create object
            test_obj = TestModel(name="Original", tenant_id="tenant-1")
            session.add(test_obj)
            await session.commit()

            original_updated_at = test_obj.updated_at
            obj_id = test_obj.id

        # Small delay to ensure timestamp difference
        import asyncio
        await asyncio.sleep(0.1)

        async with async_session() as session:
            # Update object
            result = await session.execute(
                select(TestModel).where(TestModel.id == obj_id)
            )
            test_obj = result.scalar_one()
            test_obj.name = "Updated"
            await session.commit()

            # updated_at should change
            assert test_obj.updated_at > original_updated_at

        await engine.dispose()


class TestSoftDeleteMixin:
    """Test soft delete mixin functionality."""

    def test_soft_delete_mixin_has_deleted_at(self):
        """Test that SoftDeleteMixin adds deleted_at column."""
        assert hasattr(TestModel, "deleted_at")
        assert "deleted_at" in TestModel.__table__.columns

    def test_deleted_at_is_nullable(self):
        """Test that deleted_at allows NULL (not deleted)."""
        col = TestModel.__table__.columns["deleted_at"]
        assert col.nullable is True

    def test_soft_delete_mixin_has_is_deleted_property(self):
        """Test that SoftDeleteMixin provides is_deleted property."""
        test_obj = TestModel(name="Test", tenant_id="tenant-1")
        assert hasattr(test_obj, "is_deleted")

    def test_is_deleted_false_when_deleted_at_none(self):
        """Test that is_deleted returns False when deleted_at is None."""
        test_obj = TestModel(name="Test", tenant_id="tenant-1")
        assert test_obj.is_deleted is False

    def test_is_deleted_true_when_deleted_at_set(self):
        """Test that is_deleted returns True when deleted_at is set."""
        test_obj = TestModel(name="Test", tenant_id="tenant-1")
        test_obj.deleted_at = datetime.now(timezone.utc)
        assert test_obj.is_deleted is True

    def test_soft_delete_mixin_has_soft_delete_method(self):
        """Test that SoftDeleteMixin provides soft_delete() method."""
        test_obj = TestModel(name="Test", tenant_id="tenant-1")
        assert hasattr(test_obj, "soft_delete")
        assert callable(test_obj.soft_delete)

    def test_soft_delete_method_sets_deleted_at(self):
        """Test that soft_delete() sets deleted_at timestamp."""
        test_obj = TestModel(name="Test", tenant_id="tenant-1")
        assert test_obj.deleted_at is None

        test_obj.soft_delete()

        assert test_obj.deleted_at is not None
        assert isinstance(test_obj.deleted_at, datetime)

        # Should be recent
        now = datetime.now(timezone.utc)
        time_diff = (now - test_obj.deleted_at.replace(tzinfo=timezone.utc)).total_seconds()
        assert time_diff < 5


class TestTenantMixin:
    """Test tenant mixin for RLS support."""

    def test_tenant_mixin_has_tenant_id(self):
        """Test that TenantMixin adds tenant_id column."""
        assert hasattr(TestModel, "tenant_id")
        assert "tenant_id" in TestModel.__table__.columns

    def test_tenant_id_not_nullable(self):
        """Test that tenant_id is NOT NULL (required for RLS)."""
        col = TestModel.__table__.columns["tenant_id"]
        assert col.nullable is False

    def test_tenant_id_indexed(self):
        """Test that tenant_id has index for query performance."""
        col = TestModel.__table__.columns["tenant_id"]
        # Check if column is part of any index
        indexes = [idx for idx in TestModel.__table__.indexes if "tenant_id" in [c.name for c in idx.columns]]
        assert len(indexes) > 0

    @pytest.mark.asyncio
    async def test_tenant_id_required_on_create(self, postgres_url: str):
        """Test that tenant_id must be provided when creating objects."""
        engine = create_async_engine(postgres_url)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async with async_session() as session:
            # Creating without tenant_id should fail
            test_obj = TestModel(name="No Tenant")
            session.add(test_obj)

            with pytest.raises(Exception):  # Will raise IntegrityError or similar
                await session.commit()

        await engine.dispose()


class TestModelIntegration:
    """Test integration of all mixins together."""

    @pytest.mark.asyncio
    async def test_full_model_lifecycle(self, postgres_url: str):
        """Test create, read, update, soft delete cycle."""
        engine = create_async_engine(postgres_url)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        # CREATE
        async with async_session() as session:
            test_obj = TestModel(name="Lifecycle Test", tenant_id="tenant-123")
            session.add(test_obj)
            await session.commit()

            obj_id = test_obj.id
            assert test_obj.created_at is not None
            assert test_obj.updated_at is not None
            assert test_obj.deleted_at is None
            assert test_obj.tenant_id == "tenant-123"

        # READ
        async with async_session() as session:
            result = await session.execute(
                select(TestModel).where(TestModel.id == obj_id)
            )
            test_obj = result.scalar_one()
            assert test_obj.name == "Lifecycle Test"
            assert test_obj.is_deleted is False

        # UPDATE
        async with async_session() as session:
            result = await session.execute(
                select(TestModel).where(TestModel.id == obj_id)
            )
            test_obj = result.scalar_one()
            original_updated = test_obj.updated_at

            test_obj.name = "Updated Name"
            await session.commit()

            assert test_obj.updated_at > original_updated

        # SOFT DELETE
        async with async_session() as session:
            result = await session.execute(
                select(TestModel).where(TestModel.id == obj_id)
            )
            test_obj = result.scalar_one()

            test_obj.soft_delete()
            await session.commit()

            assert test_obj.is_deleted is True
            assert test_obj.deleted_at is not None

        await engine.dispose()


class TestEventListener:
    """Test SQLAlchemy event listener for updated_at."""

    @pytest.mark.asyncio
    async def test_event_listener_skips_models_without_updated_at(self, postgres_url: str):
        """Test that event listener handles models without TimestampMixin gracefully."""
        engine = create_async_engine(postgres_url)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        # Create object without TimestampMixin
        async with async_session() as session:
            test_obj = TestModelNoTimestamp(name="No Timestamp", tenant_id="tenant-1")
            session.add(test_obj)
            await session.commit()
            obj_id = test_obj.id

        # Update object - event listener should skip it (no updated_at attribute)
        async with async_session() as session:
            result = await session.execute(
                select(TestModelNoTimestamp).where(TestModelNoTimestamp.id == obj_id)
            )
            test_obj = result.scalar_one()
            test_obj.name = "Updated Without Timestamp"
            await session.commit()

            # Should succeed without error
            assert test_obj.name == "Updated Without Timestamp"
            # Verify no updated_at attribute exists
            assert not hasattr(test_obj, "updated_at")

        await engine.dispose()
