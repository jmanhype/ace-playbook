"""
PlaybookBullet and PlaybookStage Models

Maps to PlaybookBullet entity from data-model.md.
"""

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, Enum, JSON, Index
from datetime import datetime
import enum
import uuid

from ace.models.base import Base


class PlaybookStage(str, enum.Enum):
    """Playbook deployment stages for canary rollout."""

    SHADOW = "shadow"  # Insights logged, not used in retrieval
    STAGING = "staging"  # Used by 5% of traffic
    PROD = "prod"  # Used by all production traffic
    QUARANTINED = "quarantined"  # Excluded from retrieval (harmful ≥ helpful)


class PlaybookBullet(Base):
    """
    PlaybookBullet entity - Strategy or observation with effectiveness counters.

    Schema defined in data-model.md lines 61-82.
    Append-only: content never rewritten, only counters incremented.
    """

    __tablename__ = "playbook_bullets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    domain_id = Column(
        String(64),
        nullable=False,
        index=True,
        comment="Multi-tenant namespace (CHK077-CHK078, pattern ^[a-z0-9-]+$)",
    )
    content = Column(Text, nullable=False, comment="Strategy text (never rewritten)")
    section = Column(
        String(32), nullable=False, index=True, comment="Helpful/Harmful/Neutral classification"
    )
    helpful_count = Column(Integer, nullable=False, default=0, comment="Success signal count")
    harmful_count = Column(Integer, nullable=False, default=0, comment="Failure signal count")
    tags = Column(JSON, nullable=False, comment="List[str] - domain/category tags")
    embedding = Column(JSON, nullable=False, comment="List[float] - 384-dim vector")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    stage = Column(
        Enum(PlaybookStage),
        nullable=False,
        default=PlaybookStage.SHADOW,
        index=True,
        comment="Shadow/Staging/Prod/Quarantined",
    )

    __table_args__ = (
        # CHK081: Enforce domain_id filtering for all queries
        Index("idx_domain_stage", "domain_id", "stage"),
        # Semantic search performance
        Index("idx_domain_section", "domain_id", "section"),
    )
