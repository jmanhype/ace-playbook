"""
CuratorService - Integrated service for playbook management

Combines SemanticCurator with database persistence (repositories).
"""

from __future__ import annotations

from typing import List, Dict
from datetime import datetime

from typing import TYPE_CHECKING

import hashlib

from ace.curator.semantic_curator import SemanticCurator
from ace.curator.curator_models import (
    CuratorInput,
    CuratorOutput,
    CuratorOperation,
    CuratorOperationType,
    DeltaUpdate,
    InsightSection,
    SIMILARITY_THRESHOLD_DEFAULT,
)
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.utils.database import get_session
from ace.utils.logging_config import get_logger
from ace.curator.curator_utils import compute_bullet_hash

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

    def merge_batch(
        self,
        domain_id: str,
        task_insights: List[Dict],
        target_stage: PlaybookStage = PlaybookStage.SHADOW,
    ) -> CuratorOutput:
        """Batch merge multiple task insights atomically."""

        if not task_insights:
            raise ValueError("task_insights cannot be empty")

        logger.info(
            "merge_batch_start",
            domain_id=domain_id,
            num_tasks=len(task_insights),
            target_stage=target_stage,
        )

        from ace.repositories.journal_repository import DiffJournalRepository

        with get_session() as session:
            playbook_repo = PlaybookRepository(session)
            journal_repo = DiffJournalRepository(session)

            current_playbook = playbook_repo.get_active_playbook(
                domain_id=domain_id,
                exclude_quarantined=False,
            )

            curator_dict = self.semantic_curator.batch_merge(
                task_insights=task_insights,
                current_playbook=current_playbook,
                target_stage=target_stage,
            )

            delta_updates = curator_dict["batch_results"]
            updated_playbook = curator_dict["updated_playbook"]

            output = CuratorOutput(
                task_id="batch",
                domain_id=domain_id,
                delta_updates=delta_updates,
                updated_playbook=updated_playbook,
            )
            for op in curator_dict.get("operations", []):
                output.delta.append(op)
            output.new_bullets_added = curator_dict.get("total_new_bullets", 0)
            output.existing_bullets_incremented = curator_dict.get("total_increments", 0)
            output.duplicates_detected = curator_dict.get("total_increments", 0)

            self._persist_curator_output(
                session=session,
                curator_output=output,
                playbook_repo=playbook_repo,
                journal_repo=journal_repo,
            )

            session.commit()

            logger.info(
                "merge_batch_complete",
                domain_id=domain_id,
                new_bullets=output.new_bullets_added,
                increments=output.existing_bullets_incremented,
            )

            return output

    def prune_redundant(
        self,
        domain_id: str,
        reason: str = "prune_duplicate",
    ) -> int:
        """Quarantine duplicate bullets within a domain."""

        from ace.repositories.journal_repository import DiffJournalRepository

        with get_session() as session:
            playbook_repo = PlaybookRepository(session)
            journal_repo = DiffJournalRepository(session)

            bullets = playbook_repo.get_active_playbook(
                domain_id=domain_id,
                exclude_quarantined=False,
            )

            groups: Dict[str, List[PlaybookBullet]] = {}
            for bullet in bullets:
                key = bullet.content.strip().lower()
                groups.setdefault(key, []).append(bullet)

            delta_updates: List[DeltaUpdate] = []
            updated_playbook = list(bullets)

            for duplicates in groups.values():
                if len(duplicates) <= 1:
                    continue
                keep = max(
                    duplicates,
                    key=lambda b: (b.helpful_count - b.harmful_count, -b.harmful_count, b.created_at),
                )
                for bullet in duplicates:
                    if bullet.id == keep.id or bullet.stage == PlaybookStage.QUARANTINED:
                        continue
                    before_hash = compute_bullet_hash(bullet)
                    bullet.stage = PlaybookStage.QUARANTINED
                    after_hash = compute_bullet_hash(bullet)
                    delta_updates.append(
                        DeltaUpdate(
                            operation=CuratorOperationType.QUARANTINE,
                            bullet_id=bullet.id,
                            before_hash=before_hash,
                            after_hash=after_hash,
                            metadata={"reason": reason},
                        )
                    )

            if not delta_updates:
                return 0

            output = CuratorOutput(
                task_id="prune",
                domain_id=domain_id,
                delta_updates=delta_updates,
                updated_playbook=updated_playbook,
            )
            for delta_update in delta_updates:
                try:
                    section_enum = InsightSection(
                        next(
                            (
                                b.section
                                for b in updated_playbook
                                if b.id == delta_update.bullet_id
                            ),
                            InsightSection.NEUTRAL.value,
                        )
                    )
                except ValueError:
                    section_enum = InsightSection.NEUTRAL
                output.delta.append(
                    CuratorOperation(
                        type=delta_update.operation,
                        section=section_enum,
                        content=next(
                            (
                                b.content
                                for b in updated_playbook
                                if b.id == delta_update.bullet_id
                            ),
                            "",
                        ),
                        bullet_id=delta_update.bullet_id,
                        metadata=delta_update.metadata,
                    )
                )
            output.bullets_quarantined = len(delta_updates)

            self._persist_curator_output(
                session=session,
                curator_output=output,
                playbook_repo=playbook_repo,
                journal_repo=journal_repo,
            )

            session.commit()

        logger.info(
            "prune_redundant_complete",
            domain_id=domain_id,
            quarantined=len(delta_updates),
        )

        return len(delta_updates)

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
            if delta_update.operation == CuratorOperationType.ADD and delta_update.new_bullet:
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
            immutable_fields = ("content", "tags", "embedding")
            for bullet in bullets_to_update:
                current = playbook_repo.get_by_id(bullet.id, bullet.domain_id)
                if current is None:
                    continue
                for field in immutable_fields:
                    if getattr(current, field) != getattr(bullet, field):
                        raise ValueError(
                            f"Immutable field '{field}' modified for bullet {bullet.id}"
                        )
            playbook_repo.bulk_update(bullets_to_update)

        logger.debug(
            "bullets_persisted",
            added=len(bullets_to_add),
            updated=len(bullets_to_update),
        )

        # Persist journal entries
        snapshot_source = "|".join(
            sorted(
                f"{bullet.id}:{bullet.stage}:{bullet.helpful_count}:{bullet.harmful_count}:{bullet.content.strip()}"
                for bullet in curator_output.updated_playbook
            )
        )
        snapshot_hash = hashlib.sha256(snapshot_source.encode("utf-8")).hexdigest() if snapshot_source else ""

        for delta_update in curator_output.delta_updates:
            metadata = delta_update.metadata or {}
            metadata.setdefault("context_snapshot_hash", snapshot_hash)
            metadata.setdefault("summary", f"{delta_update.operation}:{delta_update.bullet_id}")
            delta_update.metadata = metadata

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
