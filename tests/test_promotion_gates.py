"""
Integration Tests for Promotion Gates

Tests automated promotion logic based on helpful/harmful effectiveness ratios.
Verifies promotion thresholds, quarantine logic, and batch processing.

Coverage:
- T064: Integration test for promotion gates
- Shadow → Staging promotion (helpful≥3, ratio≥3:1)
- Staging → Production promotion (helpful≥5, ratio≥5:1)
- Quarantine logic (harmful≥helpful)
- Batch promotion checks
"""

import pytest
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from ace.ops import StageManager, create_stage_manager
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.repositories.journal_repository import DiffJournalRepository
from ace.reflector import InsightSection
from ace.utils.database import get_session, get_engine, init_database


@pytest.fixture
def db_session():
    """Create clean database session for testing."""
    # Reset global database engine to ensure test isolation
    import ace.utils.database as db_module
    db_module._engine = None
    db_module._session_factory = None

    # Initialize fresh database with in-memory SQLite for each test
    init_database("sqlite:///:memory:")

    # Get session
    with get_session() as session:
        yield session
        # Rollback any uncommitted changes
        session.rollback()


@pytest.fixture
def stage_manager(db_session):
    """Create StageManager with test session."""
    return StageManager(db_session)


class TestPromotionEligibility:
    """Test promotion eligibility checks."""

    def test_shadow_to_staging_eligible_exact_threshold(self, db_session, stage_manager):
        """
        T064: Test shadow → staging promotion at exact threshold.
        Threshold: helpful≥3 AND ratio≥3:1
        """
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
            helpful_count=3,
            harmful_count=1,
        tags=[]
        )

        target_stage = stage_manager.check_promotion_eligibility(bullet)
        assert target_stage == PlaybookStage.STAGING

    def test_shadow_to_staging_not_eligible_low_count(self, db_session, stage_manager):
        """Test shadow → staging not eligible due to low helpful count."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
            helpful_count=2,  # Below threshold of 3,
            harmful_count=0,
        tags=[]
        )

        target_stage = stage_manager.check_promotion_eligibility(bullet)
        assert target_stage is None

    def test_shadow_to_staging_not_eligible_low_ratio(self, db_session, stage_manager):
        """Test shadow → staging not eligible due to low ratio."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
            helpful_count=5,
            harmful_count=2,
        tags=[]
        )

        target_stage = stage_manager.check_promotion_eligibility(bullet)
        assert target_stage is None

    def test_staging_to_prod_eligible_exact_threshold(self, db_session, stage_manager):
        """
        T064: Test staging → production promotion at exact threshold.
        Threshold: helpful≥5 AND ratio≥5:1
        """
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.STAGING,
            embedding=[0.1] * 384,
            helpful_count=5,
            harmful_count=1,
        tags=[]
        )

        target_stage = stage_manager.check_promotion_eligibility(bullet)
        assert target_stage == PlaybookStage.PROD

    def test_staging_to_prod_not_eligible_low_count(self, db_session, stage_manager):
        """Test staging → production not eligible due to low helpful count."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.STAGING,
            embedding=[0.1] * 384,
            helpful_count=4,  # Below threshold of 5,
            harmful_count=0,
        tags=[]
        )

        target_stage = stage_manager.check_promotion_eligibility(bullet)
        assert target_stage is None

    def test_staging_to_prod_not_eligible_low_ratio(self, db_session, stage_manager):
        """Test staging → production not eligible due to low ratio."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.STAGING,
            embedding=[0.1] * 384,
            helpful_count=8,
            harmful_count=2,
        tags=[]
        )

        target_stage = stage_manager.check_promotion_eligibility(bullet)
        assert target_stage is None

    def test_production_bullets_not_promotable(self, db_session, stage_manager):
        """Test production bullets return None for promotion."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.PROD,
            embedding=[0.1] * 384,
            helpful_count=20,
            harmful_count=1,
        tags=[]
        )

        target_stage = stage_manager.check_promotion_eligibility(bullet)
        assert target_stage is None


class TestAutomatedPromotion:
    """Test automated promotion execution."""

    def test_promote_shadow_to_staging(self, db_session, stage_manager):
        """T064: Test automated promotion from shadow to staging."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
            helpful_count=3,
            harmful_count=1,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        # Promote
        result = stage_manager.promote_bullet("B001", "test")

        assert result == PlaybookStage.STAGING

        # Verify stage changed
        updated_bullet = db_session.query(PlaybookBullet).filter(
            PlaybookBullet.id == "B001"
        ).first()
        assert updated_bullet.stage == PlaybookStage.STAGING

    def test_promote_staging_to_production(self, db_session, stage_manager):
        """T064: Test automated promotion from staging to production."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.STAGING,
            embedding=[0.1] * 384,
            helpful_count=5,
            harmful_count=1,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        # Promote
        result = stage_manager.promote_bullet("B001", "test")

        assert result == PlaybookStage.PROD

        # Verify stage changed
        updated_bullet = db_session.query(PlaybookBullet).filter(
            PlaybookBullet.id == "B001"
        ).first()
        assert updated_bullet.stage == PlaybookStage.PROD

    def test_promotion_already_in_prod_returns_prod(self, db_session, stage_manager):
        """Test promoting production bullet returns prod stage."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.PROD,
            embedding=[0.1] * 384,
            helpful_count=20,
            harmful_count=1,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        result = stage_manager.promote_bullet("B001", "test")
        assert result == PlaybookStage.PROD

    def test_promotion_nonexistent_bullet_returns_none(self, db_session, stage_manager):
        """Test promoting non-existent bullet returns None."""
        result = stage_manager.promote_bullet("NONEXISTENT", "test")
        assert result is None


class TestForcePromotion:
    """Test force promotion override."""

    def test_force_promotion_bypasses_checks(self, db_session, stage_manager):
        """Test force promotion bypasses eligibility checks."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
            helpful_count=1,  # Below threshold,
            harmful_count=0,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        # Promote with force
        result = stage_manager.promote_bullet("B001", "test", force=True)

        assert result == PlaybookStage.STAGING



class TestQuarantineLogic:
    """Test quarantine eligibility and execution."""

    def test_quarantine_eligible_harmful_equals_helpful(self, db_session, stage_manager):
        """
        T064: Test quarantine when harmful≥helpful.
        Rule: harmful_count ≥ helpful_count AND helpful_count > 0
        """
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.STAGING,
            embedding=[0.1] * 384,
            helpful_count=3,
            harmful_count=3,
        tags=[]
        )

        is_eligible = stage_manager.check_quarantine_eligibility(bullet)
        assert is_eligible is True

    def test_quarantine_eligible_harmful_exceeds_helpful(self, db_session, stage_manager):
        """Test quarantine when harmful exceeds helpful."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.PROD,
            embedding=[0.1] * 384,
            helpful_count=5,
            harmful_count=8,
        tags=[]
        )

        is_eligible = stage_manager.check_quarantine_eligibility(bullet)
        assert is_eligible is True

    def test_quarantine_not_eligible_no_feedback(self, db_session, stage_manager):
        """Test quarantine not eligible when no helpful feedback."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
            helpful_count=0,
            harmful_count=2,
        tags=[]
        )

        is_eligible = stage_manager.check_quarantine_eligibility(bullet)
        assert is_eligible is False

    def test_quarantine_execution(self, db_session, stage_manager):
        """Test executing quarantine on eligible bullet."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.STAGING,
            embedding=[0.1] * 384,
            helpful_count=3,
            harmful_count=5,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        # Quarantine
        success = stage_manager.quarantine_bullet("B001", "test", reason="harmful_detected")

        assert success is True

        # Verify quarantined
        updated_bullet = db_session.query(PlaybookBullet).filter(
            PlaybookBullet.id == "B001"
        ).first()
        assert updated_bullet.stage == PlaybookStage.QUARANTINED

    def test_quarantine_blocks_promotion(self, db_session, stage_manager):
        """Test that eligible quarantine blocks promotion."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
            helpful_count=3,  # Meets promotion threshold,
            harmful_count=3,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        # Attempt promotion (should be blocked)
        result = stage_manager.promote_bullet("B001", "test")

        assert result is None  # Promotion blocked


class TestBatchPromotionChecks:
    """Test batch promotion checking for multiple bullets."""

    def test_check_all_promotions_promotes_eligible(self, db_session, stage_manager):
        """T064: Test batch check promotes eligible bullets."""
        bullets = [
            # Eligible for shadow → staging
            PlaybookBullet(
id="B001",
                content="Strategy 1",
                domain_id="test",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.SHADOW,
                embedding=[0.1] * 384,
                helpful_count=3,
                harmful_count=1,
            tags=[]
        ),
            # Eligible for staging → prod
            PlaybookBullet(
id="B002",
                content="Strategy 2",
                domain_id="test",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.STAGING,
                embedding=[0.2] * 384,
                helpful_count=5,
                harmful_count=1,
            tags=[]
        ),
            # Not eligible (low count)
            PlaybookBullet(
id="B003",
                content="Strategy 3",
                domain_id="test",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.SHADOW,
                embedding=[0.3] * 384,
                helpful_count=1,
                harmful_count=0,
            tags=[]
        )
        ]

        db_session.add_all(bullets)
        db_session.commit()

        # Run batch check
        result = stage_manager.check_all_promotions("test")

        assert len(result["promoted"]) == 2
        assert "B001" in result["promoted"]
        assert "B002" in result["promoted"]
        assert "B003" in result["no_action"]

    def test_check_all_promotions_quarantines_harmful(self, db_session, stage_manager):
        """Test batch check quarantines harmful bullets."""
        bullets = [
            # Eligible for quarantine
            PlaybookBullet(
id="B001",
                content="Harmful strategy",
                domain_id="test",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.STAGING,
                embedding=[0.1] * 384,
                helpful_count=2,
                harmful_count=5,
            tags=[]
        ),
            # Good bullet
            PlaybookBullet(
id="B002",
                content="Good strategy",
                domain_id="test",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.STAGING,
                embedding=[0.2] * 384,
                helpful_count=8,
                harmful_count=1,
            tags=[]
        )
        ]

        db_session.add_all(bullets)
        db_session.commit()

        # Run batch check
        result = stage_manager.check_all_promotions("test")

        assert len(result["quarantined"]) == 1
        assert "B001" in result["quarantined"]

    def test_check_all_promotions_respects_domain_isolation(self, db_session, stage_manager):
        """Test batch check only affects specified domain."""
        bullets = [
            PlaybookBullet(
id="B001",
                content="Strategy 1",
                domain_id="domain_a",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.SHADOW,
                embedding=[0.1] * 384,
                helpful_count=3,
                harmful_count=1,
            tags=[]
        ),
            PlaybookBullet(
id="B002",
                content="Strategy 2",
                domain_id="domain_b",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.SHADOW,
                embedding=[0.2] * 384,
                helpful_count=3,
                harmful_count=1,
            tags=[]
        )
        ]

        db_session.add_all(bullets)
        db_session.commit()

        # Check only domain_a
        result = stage_manager.check_all_promotions("domain_a")

        assert len(result["promoted"]) == 1
        assert "B001" in result["promoted"]
        assert "B002" not in result["promoted"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
