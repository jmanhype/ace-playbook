"""
Review Queue Model for Low-Confidence Insights

Tracks insights that require human review before promotion to shadow stage.
Used when insight confidence < 0.6 threshold.

Based on tasks.md T062.
"""

import enum
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Text, DateTime, Enum, Index
)
from ace.utils.database import Base


class ReviewStatus(str, enum.Enum):
    """Status of insights in review queue."""
    PENDING = "pending"  # Awaiting human review
    APPROVED = "approved"  # Approved, promoted to shadow
    REJECTED = "rejected"  # Rejected, discarded


class ReviewQueueItem(Base):
    """
    Review queue entry for low-confidence insights.

    T062: Queue insights with confidence < 0.6 for manual review.
    """
    __tablename__ = "review_queue"

    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Insight content
    content = Column(Text, nullable=False)
    section = Column(String, nullable=False)  # "helpful" or "harmful"
    confidence = Column(Float, nullable=False)
    rationale = Column(Text, nullable=True)

    # Source context
    source_task_id = Column(String, nullable=False, index=True)
    domain_id = Column(String, nullable=False, index=True)

    # Review status
    status = Column(
        Enum(ReviewStatus),
        nullable=False,
        default=ReviewStatus.PENDING,
        index=True
    )

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_id = Column(String, nullable=True)  # Optional reviewer identifier

    # Review decision
    review_notes = Column(Text, nullable=True)
    promoted_bullet_id = Column(String, nullable=True)  # ID of created bullet if approved

    # Indexes
    __table_args__ = (
        Index("idx_review_status_created", "status", "created_at"),
        Index("idx_review_domain_status", "domain_id", "status"),
    )

    def __repr__(self):
        return (
            f"<ReviewQueueItem(id={self.id}, "
            f"content={self.content[:50]}..., "
            f"confidence={self.confidence:.2f}, "
            f"status={self.status})>"
        )

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "section": self.section,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source_task_id": self.source_task_id,
            "domain_id": self.domain_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "promoted_bullet_id": self.promoted_bullet_id
        }


__version__ = "v1.0.0"
