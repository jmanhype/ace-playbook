"""
DiffJournalEntry and MergeOperation Models

Maps to DiffJournalEntry entity from data-model.md.
"""

from sqlalchemy import Column, String, Text, DateTime, Enum, JSON
from datetime import datetime
import enum
import uuid

from ace.models.base import Base


class MergeOperation(str, enum.Enum):
    """Types of playbook update operations."""

    ADD_NEW = "add"  # New bullet (similarity <0.8)
    INCREMENT_HELPFUL = "increment_helpful"  # Existing helpful strategy
    INCREMENT_HARMFUL = "increment_harmful"  # Existing harmful pattern
    MERGE_DUPLICATES = "merge"  # Consolidate similar bullets
    QUARANTINE = "quarantine"  # Harmful ≥ helpful, exclude from retrieval


class DiffJournalEntry(Base):
    """
    DiffJournalEntry entity - Immutable audit log of playbook changes.

    Schema defined in data-model.md lines 125-149.
    Append-only: enables rollback, debugging, and explainability.
    """

    __tablename__ = "diff_journal_entries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), nullable=False, index=True, comment="Task that triggered update")
    domain_id = Column(String(64), nullable=False, index=True, comment="Domain namespace for isolation")
    bullet_id = Column(String(36), nullable=False, index=True, comment="Affected bullet ID")
    operation = Column(
        Enum(MergeOperation), nullable=False, index=True, comment="Type of change"
    )
    before_hash = Column(String(64), nullable=True, comment="SHA-256 hash before update")
    after_hash = Column(String(64), nullable=True, comment="SHA-256 hash after update")
    before_state = Column(JSON, nullable=True, comment="Dict - bullet state before")
    after_state = Column(JSON, nullable=True, comment="Dict - bullet state after")
    similarity_score = Column(
        JSON,
        nullable=True,
        comment="Float - cosine similarity for duplicate detection"
    )
    metadata_dict = Column(
        "metadata",
        JSON,
        nullable=True,
        comment="Dict - additional metadata (merged_ids, etc.)",
    )
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    stage = Column(
        String(32), nullable=True, index=True, comment="Shadow/Staging/Prod at time of change"
    )
