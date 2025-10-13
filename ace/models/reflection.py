"""
Reflection and InsightCandidate Models

Maps to Reflection entity from data-model.md.
"""

from sqlalchemy import Column, String, Text, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ace.models.base import Base


class Reflection(Base):
    """
    Reflection entity - Reflector analysis output containing labeled insights.

    Schema defined in data-model.md lines 84-105.
    """

    __tablename__ = "reflections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False, index=True)
    analysis_summary = Column(Text, nullable=False, comment="High-level outcome summary")
    confidence_score = Column(
        Float, nullable=False, comment="Overall reflection quality (0.0-1.0)"
    )
    requires_human_review = Column(
        JSON, nullable=False, comment="bool - high impact but low confidence"
    )
    feedback_types_used = Column(
        JSON, nullable=False, comment="List[str] - feedback signals analyzed"
    )
    referenced_steps = Column(
        JSON, nullable=False, comment="List[int] - reasoning trace indices analyzed"
    )
    contradicts_existing = Column(
        JSON, nullable=False, comment="List[str] - conflicting bullet IDs"
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    insights = relationship(
        "InsightCandidate", back_populates="reflection", cascade="all, delete-orphan"
    )


class InsightCandidate(Base):
    """
    InsightCandidate entity - Single extracted strategy/observation with label.

    Schema defined in data-model.md lines 107-123.
    """

    __tablename__ = "insight_candidates"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reflection_id = Column(
        String(36), ForeignKey("reflections.id"), nullable=False, index=True
    )
    content = Column(Text, nullable=False, comment="Strategy or observation text")
    section = Column(
        String(32),
        nullable=False,
        index=True,
        comment="Helpful/Harmful/Neutral classification",
    )
    confidence = Column(
        Float, nullable=False, comment="Insight validity confidence (0.0-1.0)"
    )
    rationale = Column(
        Text, nullable=False, comment="Explanation linking insight to feedback"
    )
    tags = Column(JSON, nullable=False, comment="List[str] - domain/category tags")
    referenced_steps = Column(
        JSON, nullable=False, comment="List[int] - reasoning trace step indices"
    )
    embedding = Column(JSON, nullable=True, comment="List[float] - 384-dim vector (computed by Curator)")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    reflection = relationship("Reflection", back_populates="insights")
