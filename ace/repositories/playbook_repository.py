"""
PlaybookRepository - Database operations for PlaybookBullet entities

Implements CHK081 domain filtering and provides CRUD operations.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="playbook_repository")


class PlaybookRepository:
    """
    Repository for PlaybookBullet database operations.

    Enforces domain isolation (CHK081) in all queries.
    """

    def __init__(self, session: Session):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy session
        """
        self.session = session

    def get_by_domain(
        self,
        domain_id: str,
        stage: Optional[PlaybookStage] = None,
        section: Optional[str] = None
    ) -> List[PlaybookBullet]:
        """
        Get all bullets for a domain with optional filtering.

        Args:
            domain_id: Domain namespace (CHK081 enforced)
            stage: Optional stage filter (shadow/staging/prod)
            section: Optional section filter (Helpful/Harmful/Neutral)

        Returns:
            List of PlaybookBullet entities
        """
        query = self.session.query(PlaybookBullet).filter(
            PlaybookBullet.domain_id == domain_id
        )

        if stage is not None:
            query = query.filter(PlaybookBullet.stage == stage)

        if section is not None:
            query = query.filter(PlaybookBullet.section == section)

        bullets = query.all()

        logger.info(
            "playbook_bullets_retrieved",
            domain_id=domain_id,
            stage=stage,
            section=section,
            count=len(bullets),
        )

        return bullets

    def get_by_id(self, bullet_id: str, domain_id: str) -> Optional[PlaybookBullet]:
        """
        Get single bullet by ID with domain isolation check.

        Args:
            bullet_id: Bullet UUID
            domain_id: Domain namespace (CHK081 enforced)

        Returns:
            PlaybookBullet or None if not found or wrong domain
        """
        bullet = self.session.query(PlaybookBullet).filter(
            PlaybookBullet.id == bullet_id,
            PlaybookBullet.domain_id == domain_id,  # CHK081: Domain isolation
        ).first()

        if bullet:
            logger.debug("bullet_retrieved", bullet_id=bullet_id, domain_id=domain_id)
        else:
            logger.warning(
                "bullet_not_found_or_wrong_domain",
                bullet_id=bullet_id,
                domain_id=domain_id,
            )

        return bullet

    def add(self, bullet: PlaybookBullet) -> PlaybookBullet:
        """
        Add new bullet to database.

        Args:
            bullet: PlaybookBullet entity

        Returns:
            Persisted bullet with generated ID
        """
        self.session.add(bullet)
        self.session.flush()  # Get ID without committing

        logger.info(
            "bullet_added",
            bullet_id=bullet.id,
            domain_id=bullet.domain_id,
            section=bullet.section,
            stage=bullet.stage,
        )

        return bullet

    def update(self, bullet: PlaybookBullet) -> PlaybookBullet:
        """
        Update existing bullet (merge changes).

        Args:
            bullet: PlaybookBullet entity with updated fields

        Returns:
            Updated bullet
        """
        bullet.last_used_at = datetime.utcnow()
        self.session.merge(bullet)
        self.session.flush()

        logger.info(
            "bullet_updated",
            bullet_id=bullet.id,
            domain_id=bullet.domain_id,
            helpful_count=bullet.helpful_count,
            harmful_count=bullet.harmful_count,
        )

        return bullet

    def bulk_update(self, bullets: List[PlaybookBullet]) -> None:
        """
        Update multiple bullets efficiently.

        Args:
            bullets: List of PlaybookBullet entities to update
        """
        for bullet in bullets:
            bullet.last_used_at = datetime.utcnow()
            self.session.merge(bullet)

        self.session.flush()

        logger.info("bullets_bulk_updated", count=len(bullets))

    def get_active_playbook(
        self,
        domain_id: str,
        exclude_quarantined: bool = True
    ) -> List[PlaybookBullet]:
        """
        Get active playbook bullets for retrieval (excludes quarantined by default).

        Args:
            domain_id: Domain namespace
            exclude_quarantined: If True, exclude quarantined bullets

        Returns:
            List of active bullets for this domain
        """
        query = self.session.query(PlaybookBullet).filter(
            PlaybookBullet.domain_id == domain_id
        )

        if exclude_quarantined:
            query = query.filter(
                PlaybookBullet.stage != PlaybookStage.QUARANTINED
            )

        bullets = query.all()

        logger.info(
            "active_playbook_retrieved",
            domain_id=domain_id,
            exclude_quarantined=exclude_quarantined,
            count=len(bullets),
        )

        return bullets

    def count_by_stage(self, domain_id: str) -> dict:
        """
        Get bullet counts by stage for a domain.

        Args:
            domain_id: Domain namespace

        Returns:
            Dict mapping stage to count: {"shadow": 10, "staging": 5, ...}
        """
        from sqlalchemy import func

        results = self.session.query(
            PlaybookBullet.stage,
            func.count(PlaybookBullet.id)
        ).filter(
            PlaybookBullet.domain_id == domain_id
        ).group_by(
            PlaybookBullet.stage
        ).all()

        counts = {stage.value: count for stage, count in results}

        logger.debug("stage_counts_retrieved", domain_id=domain_id, counts=counts)

        return counts
