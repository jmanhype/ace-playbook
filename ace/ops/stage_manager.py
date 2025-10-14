"""
Stage Management for Shadow Learning

Implements stage transitions and promotion gates for playbook bullets.
Supports Shadow → Staging → Production promotion based on effectiveness signals.

Based on tasks.md T058-T060.
"""

from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from datetime import datetime

from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.repositories.journal_repository import DiffJournalRepository
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="stage_manager")


class StageManager:
    """
    Manages playbook bullet stage transitions and promotion gates.

    T058-T060: Shadow learning, promotion gates, and quarantine logic.
    """

    def __init__(self, session: Session):
        """
        Initialize StageManager.

        Args:
            session: SQLAlchemy session for database operations
        """
        self.session = session
        self.playbook_repo = PlaybookRepository(session)
        self.journal_repo = DiffJournalRepository(session)

    def set_stage(
        self,
        bullet_id: str,
        domain_id: str,
        target_stage: PlaybookStage,
        reason: Optional[str] = None
    ) -> bool:
        """
        Manually set bullet stage.

        T058: CLI command for stage management.

        Args:
            bullet_id: Bullet UUID
            domain_id: Domain namespace
            target_stage: New stage to set
            reason: Optional reason for stage change

        Returns:
            True if stage changed, False if bullet not found
        """
        bullet = self.playbook_repo.get_by_id(bullet_id, domain_id)

        if not bullet:
            logger.warning(
                "bullet_not_found_for_stage_change",
                bullet_id=bullet_id,
                domain_id=domain_id
            )
            return False

        old_stage = bullet.stage
        bullet.stage = target_stage

        self.playbook_repo.update(bullet)
        self.session.commit()

        logger.info(
            "stage_changed",
            bullet_id=bullet_id,
            old_stage=old_stage,
            new_stage=target_stage,
            reason=reason
        )

        return True

    def check_promotion_eligibility(
        self,
        bullet: PlaybookBullet
    ) -> Optional[PlaybookStage]:
        """
        Check if bullet is eligible for automatic promotion.

        T059: Promotion gate logic.

        Rules:
        - Shadow → Staging: helpful_count ≥3 AND helpful:harmful ≥3:1
        - Staging → Prod: helpful_count ≥5 AND helpful:harmful ≥5:1

        Args:
            bullet: PlaybookBullet to check

        Returns:
            Target stage if eligible for promotion, None otherwise
        """
        helpful = bullet.helpful_count
        harmful = bullet.harmful_count

        # Avoid division by zero
        ratio = helpful / harmful if harmful > 0 else float('inf')

        if bullet.stage == PlaybookStage.SHADOW:
            # Shadow → Staging promotion
            if helpful >= 3 and ratio >= 3.0:
                return PlaybookStage.STAGING

        elif bullet.stage == PlaybookStage.STAGING:
            # Staging → Prod promotion
            if helpful >= 5 and ratio >= 5.0:
                return PlaybookStage.PROD

        return None

    def check_quarantine_eligibility(self, bullet: PlaybookBullet) -> bool:
        """
        Check if bullet should be quarantined.

        T060: Quarantine logic.

        Rule: harmful_count ≥ helpful_count AND helpful_count > 0

        Args:
            bullet: PlaybookBullet to check

        Returns:
            True if should be quarantined, False otherwise
        """
        return (
            bullet.helpful_count > 0 and
            bullet.harmful_count >= bullet.helpful_count
        )

    def promote_bullet(
        self,
        bullet_id: str,
        domain_id: str,
        force: bool = False
    ) -> Optional[PlaybookStage]:
        """
        Attempt to promote bullet to next stage.

        T059: Automated promotion based on counters.

        Args:
            bullet_id: Bullet UUID
            domain_id: Domain namespace
            force: If True, bypass promotion checks

        Returns:
            New stage if promoted, None if not eligible or not found
        """
        bullet = self.playbook_repo.get_by_id(bullet_id, domain_id)

        if not bullet:
            logger.warning(
                "bullet_not_found_for_promotion",
                bullet_id=bullet_id,
                domain_id=domain_id
            )
            return None

        # Check if already in production
        if bullet.stage == PlaybookStage.PROD:
            logger.info("bullet_already_in_prod", bullet_id=bullet_id)
            return bullet.stage

        # Check quarantine eligibility first
        if self.check_quarantine_eligibility(bullet):
            logger.warning(
                "bullet_eligible_for_quarantine_not_promotion",
                bullet_id=bullet_id,
                helpful=bullet.helpful_count,
                harmful=bullet.harmful_count
            )
            return None

        # Check promotion eligibility
        target_stage = self.check_promotion_eligibility(bullet)

        if target_stage is None and not force:
            logger.info(
                "bullet_not_eligible_for_promotion",
                bullet_id=bullet_id,
                current_stage=bullet.stage,
                helpful=bullet.helpful_count,
                harmful=bullet.harmful_count
            )
            return None

        # Determine target stage for forced promotion
        if force and target_stage is None:
            if bullet.stage == PlaybookStage.SHADOW:
                target_stage = PlaybookStage.STAGING
            elif bullet.stage == PlaybookStage.STAGING:
                target_stage = PlaybookStage.PROD

        # Perform promotion
        old_stage = bullet.stage
        bullet.stage = target_stage

        self.playbook_repo.update(bullet)
        self.session.commit()

        logger.info(
            "bullet_promoted",
            bullet_id=bullet_id,
            old_stage=old_stage,
            new_stage=target_stage,
            helpful=bullet.helpful_count,
            harmful=bullet.harmful_count,
            forced=force
        )

        return target_stage

    def quarantine_bullet(
        self,
        bullet_id: str,
        domain_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Quarantine a bullet (remove from active retrieval).

        T060: Quarantine harmful strategies.

        Args:
            bullet_id: Bullet UUID
            domain_id: Domain namespace
            reason: Optional reason for quarantine

        Returns:
            True if quarantined, False if bullet not found
        """
        bullet = self.playbook_repo.get_by_id(bullet_id, domain_id)

        if not bullet:
            logger.warning(
                "bullet_not_found_for_quarantine",
                bullet_id=bullet_id,
                domain_id=domain_id
            )
            return False

        old_stage = bullet.stage
        bullet.stage = PlaybookStage.QUARANTINED

        self.playbook_repo.update(bullet)
        self.session.commit()

        logger.info(
            "bullet_quarantined",
            bullet_id=bullet_id,
            old_stage=old_stage,
            helpful=bullet.helpful_count,
            harmful=bullet.harmful_count,
            reason=reason
        )

        return True

    def check_all_promotions(
        self,
        domain_id: str
    ) -> Dict[str, List[str]]:
        """
        Check all bullets in domain for promotion/quarantine eligibility.

        T059-T060: Periodic promotion check for online learning loop.

        Args:
            domain_id: Domain namespace

        Returns:
            Dict with "promoted", "quarantined", and "no_action" bullet ID lists
        """
        promoted_ids = []
        quarantined_ids = []
        no_action_ids = []

        # Get all non-quarantined, non-prod bullets
        bullets = self.playbook_repo.get_by_domain(domain_id)

        for bullet in bullets:
            if bullet.stage == PlaybookStage.QUARANTINED:
                continue

            if bullet.stage == PlaybookStage.PROD:
                # Check if should be quarantined even in prod
                if self.check_quarantine_eligibility(bullet):
                    if self.quarantine_bullet(bullet.id, domain_id, "harmful_in_production"):
                        quarantined_ids.append(bullet.id)
                continue

            # Check quarantine first
            if self.check_quarantine_eligibility(bullet):
                if self.quarantine_bullet(bullet.id, domain_id, "automatic_check"):
                    quarantined_ids.append(bullet.id)
                continue

            # Check promotion
            target_stage = self.check_promotion_eligibility(bullet)
            if target_stage:
                result = self.promote_bullet(bullet.id, domain_id)
                if result:
                    promoted_ids.append(bullet.id)
            else:
                no_action_ids.append(bullet.id)

        logger.info(
            "promotion_check_complete",
            domain_id=domain_id,
            promoted=len(promoted_ids),
            quarantined=len(quarantined_ids),
            no_action=len(no_action_ids)
        )

        return {
            "promoted": promoted_ids,
            "quarantined": quarantined_ids,
            "no_action": no_action_ids
        }

    def get_shadow_bullets(self, domain_id: str) -> List[PlaybookBullet]:
        """
        Get all shadow bullets for a domain.

        T058: Shadow bullets are logged but not used in retrieval.

        Args:
            domain_id: Domain namespace

        Returns:
            List of shadow-stage bullets
        """
        return self.playbook_repo.get_by_domain(
            domain_id,
            stage=PlaybookStage.SHADOW
        )

    def get_production_bullets(self, domain_id: str) -> List[PlaybookBullet]:
        """
        Get production-ready bullets for active retrieval.

        T058: Only prod bullets used in generator context.

        Args:
            domain_id: Domain namespace

        Returns:
            List of production-stage bullets
        """
        return self.playbook_repo.get_by_domain(
            domain_id,
            stage=PlaybookStage.PROD
        )


def create_stage_manager(session: Session) -> StageManager:
    """
    Factory function to create StageManager.

    Args:
        session: SQLAlchemy session

    Returns:
        StageManager instance
    """
    return StageManager(session)


__version__ = "v1.0.0"
