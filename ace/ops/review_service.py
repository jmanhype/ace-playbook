"""
Review Service for Low-Confidence Insights

Service layer for managing insight review queue with curator integration.
Handles queuing, approval with bullet creation, and rejection workflows.

Based on tasks.md T062.
"""

from typing import List, Optional, Dict
from sqlalchemy.orm import Session

from ace.models.review_queue import ReviewQueueItem, ReviewStatus
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.repositories.review_queue_repository import ReviewQueueRepository
from ace.repositories.playbook_repository import PlaybookRepository
from ace.repositories.journal_repository import DiffJournalRepository
from ace.curator import SemanticCurator
from ace.reflector import InsightCandidate, InsightSection
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="review_service")

# Confidence threshold for automatic queuing
REVIEW_CONFIDENCE_THRESHOLD = 0.6


class ReviewService:
    """
    Service for managing insight review workflow.

    T062: Queue low-confidence insights (<0.6) for manual review.
    """

    def __init__(self, session: Session):
        """
        Initialize ReviewService.

        Args:
            session: SQLAlchemy session for database operations
        """
        self.session = session
        self.review_repo = ReviewQueueRepository(session)
        self.playbook_repo = PlaybookRepository(session)
        self.journal_repo = DiffJournalRepository(session)
        self.curator = SemanticCurator()

    def should_queue_for_review(self, confidence: float) -> bool:
        """
        Check if insight should be queued for review.

        T062: Queue if confidence < 0.6.

        Args:
            confidence: Insight confidence score

        Returns:
            True if should be queued, False if can be processed directly
        """
        return confidence < REVIEW_CONFIDENCE_THRESHOLD

    def queue_insight(
        self,
        insight: InsightCandidate,
        source_task_id: str,
        domain_id: str
    ) -> ReviewQueueItem:
        """
        Queue insight for review.

        T062: Add low-confidence insight to review queue.

        Args:
            insight: InsightCandidate to queue
            source_task_id: Task that generated this insight
            domain_id: Domain namespace

        Returns:
            Created ReviewQueueItem
        """
        return self.review_repo.add_to_queue(
            content=insight.content,
            section=insight.section.value,
            confidence=insight.confidence,
            source_task_id=source_task_id,
            domain_id=domain_id,
            rationale=insight.rationale
        )

    def list_pending(
        self,
        domain_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ReviewQueueItem]:
        """
        List pending review items.

        T062: CLI command `ace review list`.

        Args:
            domain_id: Optional filter by domain
            limit: Optional limit on number of items

        Returns:
            List of pending ReviewQueueItems
        """
        return self.review_repo.get_pending(domain_id=domain_id, limit=limit)

    def approve_and_promote(
        self,
        item_id: str,
        reviewer_id: Optional[str] = None,
        review_notes: Optional[str] = None
    ) -> Optional[str]:
        """
        Approve review item and create playbook bullet in shadow stage.

        T062: CLI command `ace review approve <insight_id>`.

        Args:
            item_id: Review queue item ID
            reviewer_id: Optional identifier of reviewer
            review_notes: Optional notes about the decision

        Returns:
            Created bullet ID if successful, None if failed
        """
        item = self.review_repo.get_by_id(item_id)

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

        try:
            # Convert to InsightCandidate format for curator
            insight_dict = {
                "content": item.content,
                "section": item.section,
                "confidence": item.confidence,
                "source_task_id": item.source_task_id
            }

            # Get current playbook for domain
            current_playbook = self.playbook_repo.get_by_domain(
                domain_id=item.domain_id
            )

            # Merge with curator to create bullet in shadow stage
            merge_result = self.curator.batch_merge(
                task_insights=[{
                    "task_id": item.source_task_id,
                    "domain_id": item.domain_id,
                    "insights": [insight_dict]
                }],
                current_playbook=current_playbook,
                target_stage=PlaybookStage.SHADOW,
                similarity_threshold=0.8
            )

            # Extract created bullet ID
            bullet_id = None
            if merge_result["total_new_bullets"] > 0 and merge_result["updated_playbook"]:
                bullet_id = merge_result["updated_playbook"][0].id

            # Mark as approved
            item.status = ReviewStatus.APPROVED
            item.reviewed_at = self.session.query(ReviewQueueItem).filter(
                ReviewQueueItem.id == item_id
            ).first().created_at  # Use current timestamp
            item.reviewer_id = reviewer_id
            item.review_notes = review_notes
            item.promoted_bullet_id = bullet_id

            self.session.commit()

            logger.info(
                "review_item_approved_and_promoted",
                item_id=item_id,
                bullet_id=bullet_id,
                reviewer_id=reviewer_id
            )

            return bullet_id

        except Exception as e:
            self.session.rollback()
            logger.error(
                "review_approval_failed",
                item_id=item_id,
                error=str(e),
                exc_info=True
            )
            return None

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
            True if rejected, False if failed
        """
        success = self.review_repo.reject(
            item_id=item_id,
            reviewer_id=reviewer_id,
            review_notes=review_notes
        )

        if success:
            self.session.commit()

        return success

    def get_statistics(self, domain_id: Optional[str] = None) -> Dict:
        """
        Get review queue statistics.

        Args:
            domain_id: Optional filter by domain

        Returns:
            Dict with counts by status
        """
        return self.review_repo.get_statistics(domain_id=domain_id)


def create_review_service(session: Session) -> ReviewService:
    """
    Factory function to create ReviewService.

    Args:
        session: SQLAlchemy session

    Returns:
        ReviewService instance
    """
    return ReviewService(session)


__version__ = "v1.0.0"
