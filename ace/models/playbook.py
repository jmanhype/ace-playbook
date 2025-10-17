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

    # NEW: Tool-calling strategy fields (optional, for ReAct agent support)
    tool_sequence = Column(
        JSON,
        nullable=True,
        comment="Optional[List[str]] - Ordered sequence of tool names used in successful execution",
    )
    tool_success_rate = Column(
        Float,
        nullable=True,
        comment="Optional[float] - Success rate for this tool sequence (0.0-1.0)",
    )
    avg_iterations = Column(
        Integer,
        nullable=True,
        comment="Optional[int] - Average iterations when this tool sequence was used",
    )
    # T031: Tool reliability metrics
    avg_execution_time_ms = Column(
        Float,
        nullable=True,
        comment="Optional[float] - Average execution time in milliseconds for tool sequence",
    )

    # T058: Playbook archaeology - attribution metadata for traceability
    source_task_id = Column(
        String(64),
        nullable=True,
        index=True,
        comment="Task ID that generated this bullet (for traceability)",
    )
    source_reflection_id = Column(
        String(36),
        nullable=True,
        comment="Reflection ID that created this bullet (foreign key to reflections table)",
    )
    generated_by = Column(
        String(32),
        nullable=True,
        comment="Component that generated this: 'reflector', 'curator', 'manual', 'import'",
    )
    generation_context = Column(
        JSON,
        nullable=True,
        comment="Additional context about how this bullet was generated (metadata dict)",
    )

    __table_args__ = (
        # CHK081: Enforce domain_id filtering for all queries
        Index("idx_domain_stage", "domain_id", "stage"),
        # Semantic search performance
        Index("idx_domain_section", "domain_id", "section"),
        # Tool-calling strategy filtering
        Index("idx_domain_tool_sequence", "domain_id", "tool_sequence"),
    )
