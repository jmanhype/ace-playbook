"""
Review Queue Repository

Database operations for managing insight review queue.
Handles queuing, approval, and rejection workflows.

Based on tasks.md T062.
"""

from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy.orm import Session

from ace.models.review_queue import ReviewQueueItem, ReviewStatus
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="review_queue_repo")


class ReviewQueueRepository:
    """
    Repository for review queue operations.

    T062: Queue low-confidence insights for manual review.
    """

    def __init__(self, session: Session):
        """
        Initialize ReviewQueueRepository.

        Args:
            session: SQLAlchemy session for database operations
        """
        self.session = session

    def add_to_queue(
        self,
        content: str,
        section: str,
        confidence: float,
        source_task_id: str,
        domain_id: str,
        rationale: Optional[str] = None
    ) -> ReviewQueueItem:
        """
        Add insight to review queue.

        T062: Queue insights with confidence < 0.6.

        Args:
            content: Insight content text
            section: "helpful" or "harmful"
            confidence: Confidence score
            source_task_id: Task that generated this insight
            domain_id: Domain namespace
            rationale: Optional rationale for the insight

        Returns:
            Created ReviewQueueItem
        """
        item = ReviewQueueItem(
            content=content,
            section=section,
            confidence=confidence,
            rationale=rationale,
            source_task_id=source_task_id,
            domain_id=domain_id,
            status=ReviewStatus.PENDING
        )

        self.session.add(item)
        self.session.flush()  # Get ID without committing

        logger.info(
            "insight_queued_for_review",
            item_id=item.id,
            confidence=confidence,
            section=section,
            domain_id=domain_id
        )

        return item

    def get_pending(
        self,
        domain_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ReviewQueueItem]:
        """
        Get pending review items.

        T062: CLI command `ace review list`.

        Args:
            domain_id: Optional filter by domain
            limit: Optional limit on number of items

        Returns:
            List of pending ReviewQueueItems
        """
        query = self.session.query(ReviewQueueItem).filter(
            ReviewQueueItem.status == ReviewStatus.PENDING
        )

        if domain_id:
            query = query.filter(ReviewQueueItem.domain_id == domain_id)

        query = query.order_by(ReviewQueueItem.created_at.asc())

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_by_id(self, item_id: str) -> Optional[ReviewQueueItem]:
        """
        Get review item by ID.

        Args:
            item_id: Review queue item ID

        Returns:
            ReviewQueueItem or None if not found
        """
        return self.session.query(ReviewQueueItem).filter(
            ReviewQueueItem.id == item_id
        ).first()

    def approve(
        self,
        item_id: str,
        reviewer_id: Optional[str] = None,
        review_notes: Optional[str] = None
    ) -> Optional[str]:
        """
        Approve review item and promote to shadow stage.

        T062: CLI command `ace review approve <insight_id>`.

        Args:
            item_id: Review queue item ID
            reviewer_id: Optional identifier of reviewer
            review_notes: Optional notes about the decision

        Returns:
            Promoted bullet ID if successful, None if item not found
        """
        item = self.get_by_id(item_id)

        if not item:
            logger.warning("review_item_not_found", item_id=item_id)
            return None

        if item.status != ReviewStatus.PENDING:
            logger.warning(
                "review_item_not_pending",
                item_id=item_id,
                status=item.status
            )
            return None

        # Create playbook bullet in shadow stage
        # Note: This requires embedding generation
        # For now, we mark as approved and return a placeholder
        # The actual bullet creation should be done by curator with embedding
        item.status = ReviewStatus.APPROVED
        item.reviewed_at = datetime.utcnow()
        item.reviewer_id = reviewer_id
        item.review_notes = review_notes

        self.session.flush()

        logger.info(
            "review_item_approved",
            item_id=item_id,
            reviewer_id=reviewer_id,
            domain_id=item.domain_id
        )

        return item.id

    def reject(
        self,
        item_id: str,
        reviewer_id: Optional[str] = None,
        review_notes: Optional[str] = None
    ) -> bool:
        """
        Reject review item and discard.

        T062: CLI command `ace review reject <insight_id>`.

        Args:
            item_id: Review queue item ID
            reviewer_id: Optional identifier of reviewer
            review_notes: Optional notes about the decision

        Returns:
            True if rejected, False if item not found
        """
        item = self.get_by_id(item_id)

        if not item:
            logger.warning("review_item_not_found", item_id=item_id)
            return False

        if item.status != ReviewStatus.PENDING:
            logger.warning(
                "review_item_not_pending",
                item_id=item_id,
                status=item.status
            )
            return False

        item.status = ReviewStatus.REJECTED
        item.reviewed_at = datetime.utcnow()
        item.reviewer_id = reviewer_id
        item.review_notes = review_notes

        self.session.flush()

        logger.info(
            "review_item_rejected",
            item_id=item_id,
            reviewer_id=reviewer_id,
            reason=review_notes
        )

        return True

    def get_statistics(self, domain_id: Optional[str] = None) -> Dict:
        """
        Get review queue statistics.

        Args:
            domain_id: Optional filter by domain

        Returns:
            Dict with counts by status
        """
        query = self.session.query(ReviewQueueItem)

        if domain_id:
            query = query.filter(ReviewQueueItem.domain_id == domain_id)

        total = query.count()
        pending = query.filter(ReviewQueueItem.status == ReviewStatus.PENDING).count()
        approved = query.filter(ReviewQueueItem.status == ReviewStatus.APPROVED).count()
        rejected = query.filter(ReviewQueueItem.status == ReviewStatus.REJECTED).count()

        return {
            "total": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected
        }


__version__ = "v1.0.0"
