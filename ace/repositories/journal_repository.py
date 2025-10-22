"""
DiffJournalRepository - Audit trail for playbook changes

Implements append-only immutable journal for rollback and debugging.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from ace.models.journal import DiffJournalEntry, MergeOperation
from ace.curator.curator_models import CuratorOperationType, DeltaUpdate
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="journal_repository")


class DiffJournalRepository:
    """
    Repository for DiffJournalEntry audit log.

    Provides append-only journal for playbook changes with rollback support.
    """

    def __init__(self, session: Session):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy session
        """
        self.session = session

    def add_entry(
        self,
        task_id: str,
        domain_id: str,
        delta_update: DeltaUpdate,
    ) -> DiffJournalEntry:
        """
        Add single journal entry for a delta update.

        Args:
            task_id: Task that triggered this update
            domain_id: Domain namespace
            delta_update: DeltaUpdate from SemanticCurator

        Returns:
            Persisted DiffJournalEntry
        """
        # Map operation string to enum
        operation_map = {
            CuratorOperationType.ADD: MergeOperation.ADD_NEW,
            CuratorOperationType.INCREMENT_HELPFUL: MergeOperation.INCREMENT_HELPFUL,
            CuratorOperationType.INCREMENT_HARMFUL: MergeOperation.INCREMENT_HARMFUL,
            CuratorOperationType.INCREMENT_NEUTRAL: MergeOperation.INCREMENT_HELPFUL,
            CuratorOperationType.QUARANTINE: MergeOperation.QUARANTINE,
        }

        operation = operation_map.get(
            delta_update.operation,
            MergeOperation.ADD_NEW,
        )

        # Build before/after state dictionaries
        before_state = None
        after_state = None

        if delta_update.new_bullet:
            # New bullet - only after state
            after_state = {
                "id": delta_update.new_bullet.id,
                "content": delta_update.new_bullet.content,
                "section": delta_update.new_bullet.section,
                "helpful_count": delta_update.new_bullet.helpful_count,
                "harmful_count": delta_update.new_bullet.harmful_count,
                "stage": delta_update.new_bullet.stage.value,
            }

        entry = DiffJournalEntry(
            task_id=task_id,
            domain_id=domain_id,
            bullet_id=delta_update.bullet_id,
            operation=operation,
            before_hash=delta_update.before_hash,
            after_hash=delta_update.after_hash,
            before_state=before_state,
            after_state=after_state,
            similarity_score=delta_update.similarity_score,
            metadata_dict=delta_update.metadata,
            timestamp=delta_update.timestamp,
        )

        self.session.add(entry)
        self.session.flush()

        logger.debug(
            "journal_entry_added",
            task_id=task_id,
            domain_id=domain_id,
            bullet_id=delta_update.bullet_id,
            operation=operation.value,
        )

        return entry

    def add_entries_batch(
        self,
        task_id: str,
        domain_id: str,
        delta_updates: List[DeltaUpdate],
    ) -> List[DiffJournalEntry]:
        """
        Add multiple journal entries efficiently (batch insert).

        Args:
            task_id: Task that triggered these updates
            domain_id: Domain namespace
            delta_updates: List of DeltaUpdate objects

        Returns:
            List of persisted DiffJournalEntry objects
        """
        entries = []

        for delta_update in delta_updates:
            effective_task_id = delta_update.metadata.get("task_id") if delta_update.metadata else None
            entry = self.add_entry(effective_task_id or task_id, domain_id, delta_update)
            entries.append(entry)

        self.session.flush()

        logger.info(
            "journal_entries_batch_added",
            task_id=task_id,
            domain_id=domain_id,
            count=len(entries),
        )

        return entries

    def get_by_task(self, task_id: str) -> List[DiffJournalEntry]:
        """
        Get all journal entries for a task.

        Args:
            task_id: Task UUID

        Returns:
            List of journal entries ordered by timestamp
        """
        entries = self.session.query(DiffJournalEntry).filter(
            DiffJournalEntry.task_id == task_id
        ).order_by(
            DiffJournalEntry.timestamp
        ).all()

        logger.debug("journal_entries_retrieved_by_task", task_id=task_id, count=len(entries))

        return entries

    def get_by_domain(
        self,
        domain_id: str,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[DiffJournalEntry]:
        """
        Get journal entries for a domain with optional time filter.

        Args:
            domain_id: Domain namespace
            since: Optional timestamp to filter entries after this time
            limit: Maximum number of entries to return

        Returns:
            List of journal entries ordered by timestamp (newest first)
        """
        query = self.session.query(DiffJournalEntry).filter(
            DiffJournalEntry.domain_id == domain_id
        )

        if since:
            query = query.filter(DiffJournalEntry.timestamp >= since)

        entries = query.order_by(
            DiffJournalEntry.timestamp.desc()
        ).limit(limit).all()

        logger.info(
            "journal_entries_retrieved_by_domain",
            domain_id=domain_id,
            since=since,
            count=len(entries),
        )

        return entries

    def get_by_bullet(
        self,
        bullet_id: str,
        domain_id: str
    ) -> List[DiffJournalEntry]:
        """
        Get history of changes for a specific bullet.

        Args:
            bullet_id: Bullet UUID
            domain_id: Domain namespace (for isolation)

        Returns:
            List of journal entries ordered by timestamp
        """
        entries = self.session.query(DiffJournalEntry).filter(
            DiffJournalEntry.bullet_id == bullet_id,
            DiffJournalEntry.domain_id == domain_id,
        ).order_by(
            DiffJournalEntry.timestamp
        ).all()

        logger.debug(
            "journal_entries_retrieved_by_bullet",
            bullet_id=bullet_id,
            domain_id=domain_id,
            count=len(entries),
        )

        return entries

    def get_recent_changes(
        self,
        domain_id: str,
        window_seconds: int = 300
    ) -> List[DiffJournalEntry]:
        """
        Get recent changes within time window (for rollback monitoring).

        Args:
            domain_id: Domain namespace
            window_seconds: Time window in seconds (default: 300 = 5 minutes)

        Returns:
            List of recent journal entries
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)

        entries = self.session.query(DiffJournalEntry).filter(
            DiffJournalEntry.domain_id == domain_id,
            DiffJournalEntry.timestamp >= cutoff_time,
        ).order_by(
            DiffJournalEntry.timestamp.desc()
        ).all()

        logger.info(
            "recent_changes_retrieved",
            domain_id=domain_id,
            window_seconds=window_seconds,
            count=len(entries),
        )

        return entries

    def count_operations_by_type(
        self,
        domain_id: str,
        since: Optional[datetime] = None
    ) -> dict:
        """
        Get operation counts by type for monitoring.

        Args:
            domain_id: Domain namespace
            since: Optional timestamp to filter after

        Returns:
            Dict mapping operation type to count
        """
        from sqlalchemy import func

        query = self.session.query(
            DiffJournalEntry.operation,
            func.count(DiffJournalEntry.id)
        ).filter(
            DiffJournalEntry.domain_id == domain_id
        )

        if since:
            query = query.filter(DiffJournalEntry.timestamp >= since)

        results = query.group_by(DiffJournalEntry.operation).all()

        counts = {operation.value: count for operation, count in results}

        logger.debug(
            "operation_counts_retrieved",
            domain_id=domain_id,
            since=since,
            counts=counts,
        )

        return counts
