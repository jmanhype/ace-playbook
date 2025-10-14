"""
CuratorService - Integrated service for playbook management

Combines SemanticCurator with database persistence (repositories).
"""

from __future__ import annotations

from typing import List, Dict
from datetime import datetime

from typing import TYPE_CHECKING

from ace.curator.semantic_curator import (
    SemanticCurator,
    CuratorInput,
    CuratorOutput,
    SIMILARITY_THRESHOLD_DEFAULT,
)
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.utils.database import get_session
from ace.utils.logging_config import get_logger

if TYPE_CHECKING:
    from ace.repositories.journal_repository import DiffJournalRepository

logger = get_logger(__name__, component="curator_service")


class CuratorService:
    """
    High-level Curator service with database persistence.

    Orchestrates:
    1. SemanticCurator for deduplication logic
    2. PlaybookRepository for bullet CRUD
    3. DiffJournalRepository for audit trail
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    ):
        """
        Initialize CuratorService.

        Args:
            embedding_model: sentence-transformers model name
            similarity_threshold: Cosine similarity threshold (0.8)
        """
        self.semantic_curator = SemanticCurator(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
        )

        logger.info(
            "curator_service_initialized",
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
        )

    def merge_insights(
        self,
        task_id: str,
        domain_id: str,
        insights: List[Dict],
        target_stage: PlaybookStage = PlaybookStage.SHADOW,
        similarity_threshold: float | None = None,
    ) -> CuratorOutput:
        """
        Merge insights into domain's playbook with semantic deduplication.

        Args:
            task_id: Task that generated these insights
            domain_id: Domain namespace (multi-tenant isolation)
            insights: List of insight dicts with content, section, tags
            target_stage: Stage for new bullets (shadow/staging/prod)
            similarity_threshold: Override default threshold if provided

        Returns:
            CuratorOutput with delta updates and statistics

        Raises:
            ValueError: If domain isolation is violated
        """
        logger.info(
            "merge_insights_start",
            task_id=task_id,
            domain_id=domain_id,
            num_insights=len(insights),
            target_stage=target_stage,
        )

        # Lazy import to avoid circular dependency
        from ace.repositories.journal_repository import DiffJournalRepository

        with get_session() as session:
            playbook_repo = PlaybookRepository(session)
            journal_repo = DiffJournalRepository(session)

            # Load current playbook from database
            current_playbook = playbook_repo.get_active_playbook(
                domain_id=domain_id,
                exclude_quarantined=False,  # Include all for deduplication
            )

            logger.debug(
                "current_playbook_loaded",
                domain_id=domain_id,
                bullet_count=len(current_playbook),
            )

            # Build CuratorInput
            curator_input = CuratorInput(
                task_id=task_id,
                domain_id=domain_id,
                insights=insights,
                current_playbook=current_playbook,
                target_stage=target_stage,
                similarity_threshold=(
                    similarity_threshold
                    if similarity_threshold is not None
                    else self.semantic_curator.similarity_threshold
                ),
            )

            # Apply semantic deduplication
            curator_output = self.semantic_curator.apply_delta(curator_input)

            # Persist changes to database
            self._persist_curator_output(
                session=session,
                curator_output=curator_output,
                playbook_repo=playbook_repo,
                journal_repo=journal_repo,
            )

            # Commit transaction
            session.commit()

            logger.info(
                "merge_insights_complete",
                task_id=task_id,
                domain_id=domain_id,
                new_bullets=curator_output.new_bullets_added,
                incremented=curator_output.existing_bullets_incremented,
                quarantined=curator_output.bullets_quarantined,
            )

            return curator_output

    def _persist_curator_output(
        self,
        session,
        curator_output: CuratorOutput,
        playbook_repo: PlaybookRepository,
        journal_repo: DiffJournalRepository,
    ) -> None:
        """
        Persist CuratorOutput to database (bullets + journal).

        Args:
            session: Database session (for transaction context)
            curator_output: Output from SemanticCurator
            playbook_repo: Playbook repository instance
            journal_repo: Journal repository instance
        """
        # Update playbook bullets
        bullets_to_update = []
        bullets_to_add = []

        for delta_update in curator_output.delta_updates:
            if delta_update.operation == "add" and delta_update.new_bullet:
                bullets_to_add.append(delta_update.new_bullet)
            else:
                # Find updated bullet in updated_playbook
                bullet = next(
                    (b for b in curator_output.updated_playbook if b.id == delta_update.bullet_id),
                    None,
                )
                if bullet:
                    bullets_to_update.append(bullet)

        # Bulk operations
        for bullet in bullets_to_add:
            playbook_repo.add(bullet)

        if bullets_to_update:
            playbook_repo.bulk_update(bullets_to_update)

        logger.debug(
            "bullets_persisted",
            added=len(bullets_to_add),
            updated=len(bullets_to_update),
        )

        # Persist journal entries
        journal_repo.add_entries_batch(
            task_id=curator_output.task_id,
            domain_id=curator_output.domain_id,
            delta_updates=curator_output.delta_updates,
        )

        logger.debug("journal_entries_persisted", count=len(curator_output.delta_updates))

    def get_playbook(
        self,
        domain_id: str,
        stage: PlaybookStage | None = None,
        section: str | None = None,
    ) -> List[PlaybookBullet]:
        """
        Retrieve playbook bullets for a domain.

        Args:
            domain_id: Domain namespace
            stage: Optional stage filter (shadow/staging/prod)
            section: Optional section filter (Helpful/Harmful/Neutral)

        Returns:
            List of PlaybookBullet entities
        """
        with get_session() as session:
            playbook_repo = PlaybookRepository(session)
            bullets = playbook_repo.get_by_domain(
                domain_id=domain_id,
                stage=stage,
                section=section,
            )

            # Eagerly load all attributes before session closes to prevent DetachedInstanceError
            for bullet in bullets:
                # Access all attributes to force SQLAlchemy to load them
                _ = bullet.id
                _ = bullet.domain_id
                _ = bullet.content
                _ = bullet.section
                _ = bullet.helpful_count
                _ = bullet.harmful_count
                _ = bullet.tags
                _ = bullet.embedding
                _ = bullet.created_at
                _ = bullet.last_used_at
                _ = bullet.stage

            # Detach objects from session so they can be used outside the context
            session.expunge_all()

        logger.info(
            "playbook_retrieved",
            domain_id=domain_id,
            stage=stage,
            section=section,
            count=len(bullets),
        )

        return bullets

    def get_stage_counts(self, domain_id: str) -> dict:
        """
        Get bullet counts by stage for monitoring.

        Args:
            domain_id: Domain namespace

        Returns:
            Dict mapping stage to count
        """
        with get_session() as session:
            playbook_repo = PlaybookRepository(session)
            counts = playbook_repo.count_by_stage(domain_id)

        logger.debug("stage_counts_retrieved", domain_id=domain_id, counts=counts)

        return counts

    def get_recent_changes(
        self,
        domain_id: str,
        window_seconds: int = 300,
    ) -> List:
        """
        Get recent changes for rollback monitoring.

        Args:
            domain_id: Domain namespace
            window_seconds: Time window in seconds (default: 300 = 5 minutes)

        Returns:
            List of recent DiffJournalEntry objects
        """
        # Lazy import to avoid circular dependency
        from ace.repositories.journal_repository import DiffJournalRepository

        with get_session() as session:
            journal_repo = DiffJournalRepository(session)
            entries = journal_repo.get_recent_changes(
                domain_id=domain_id,
                window_seconds=window_seconds,
            )

        logger.info(
            "recent_changes_retrieved",
            domain_id=domain_id,
            window_seconds=window_seconds,
            count=len(entries),
        )

        return entries
