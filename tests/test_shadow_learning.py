"""
Integration Tests for Shadow Learning Mode

Tests shadow learning workflow: insights created in shadow stage,
not used in retrieval, can be promoted through stages.

Coverage:
- T063: Integration test for shadow learning
- Shadow bullets not retrieved for generator context
- Staging bullets used by subset of traffic
- Production bullets used by all traffic
- Stage transitions and filtering
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from ace.ops import StageManager, create_stage_manager
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.repositories.journal_repository import DiffJournalRepository
from ace.reflector import InsightCandidate, InsightSection
from ace.curator import SemanticCurator
from ace.utils.database import get_session, get_engine, init_database


@pytest.fixture
def db_session():
    """Create clean database session for testing."""
    # Initialize database with in-memory SQLite
    init_database("sqlite:///:memory:")

    # Get session
    with get_session() as session:
        yield session


@pytest.fixture
def stage_manager(db_session):
    """Create StageManager with test session."""
    return StageManager(db_session)


@pytest.fixture
def playbook_repo(db_session):
    """Create PlaybookRepository with test session."""
    return PlaybookRepository(db_session)


class TestShadowStageCreation:
    """Test creating bullets in shadow stage."""

    def test_curator_creates_bullets_in_shadow_stage(self, db_session, playbook_repo):
        """
        T063: Test that curator creates new bullets in shadow stage.
        """
        curator = SemanticCurator()

        # Create insight for merging
        insight = {
            "content": "Break complex problems into smaller steps",
            "section": "helpful",
            "confidence": 0.85,
            "source_task_id": "task-001"
        }

        # Merge with empty playbook, target shadow stage
        result = curator.batch_merge(
            task_insights=[{
                "task_id": "task-001",
                "domain_id": "arithmetic",
                "insights": [insight]
            }],
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
            similarity_threshold=0.8
        )

        assert result["total_new_bullets"] == 1
        assert len(result["updated_playbook"]) == 1

        bullet = result["updated_playbook"][0]
        assert bullet.stage == PlaybookStage.SHADOW
        assert bullet.domain_id == "arithmetic"
        assert bullet.content == insight["content"]

    def test_shadow_bullets_have_correct_metadata(self, db_session, playbook_repo):
        """Test shadow bullets have proper stage and metadata."""
        curator = SemanticCurator()

        insights = [
            {
                "content": f"Strategy {i}",
                "section": "helpful",
                "confidence": 0.8,
                "source_task_id": f"task-{i:03d}"
            }
            for i in range(3)
        ]

        result = curator.batch_merge(
            task_insights=[{
                "task_id": "batch-001",
                "domain_id": "test_domain",
                "insights": insights
            }],
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
            similarity_threshold=0.8
        )

        for bullet in result["updated_playbook"]:
            assert bullet.stage == PlaybookStage.SHADOW
            assert bullet.helpful_count == 0
            assert bullet.harmful_count == 0
            assert bullet.domain_id == "test_domain"


class TestShadowRetrievalExclusion:
    """Test that shadow bullets are excluded from retrieval."""

    def test_shadow_bullets_not_in_production_retrieval(self, db_session, stage_manager):
        """
        T063: Verify shadow bullets are not retrieved for generator context.
        """
        # Create bullets in different stages
        shadow_bullet = PlaybookBullet(
id="B001",
            content="Shadow strategy",
            domain_id="arithmetic",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
        tags=[]
        )

        prod_bullet = PlaybookBullet(
id="B002",
            content="Production strategy",
            domain_id="arithmetic",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.PROD,
            embedding=[0.2] * 384,
        tags=[]
        )

        db_session.add_all([shadow_bullet, prod_bullet])
        db_session.commit()

        # Get production bullets only
        prod_bullets = stage_manager.get_production_bullets("arithmetic")

        assert len(prod_bullets) == 1
        assert prod_bullets[0].id == "B002"
        assert prod_bullets[0].stage == PlaybookStage.PROD

    def test_get_shadow_bullets_returns_only_shadow(self, db_session, stage_manager):
        """Test get_shadow_bullets returns only shadow-stage bullets."""
        # Create bullets in all stages
        bullets = [
            PlaybookBullet(
id=f"B{i:03d}",
                content=f"Strategy {i}",
                domain_id="test",
                section=InsightSection.HELPFUL,
                stage=stage,
                embedding=[float(i,
            tags=[]
        )] * 384,
            tags=[]
            )
            for i, stage in enumerate([
                PlaybookStage.SHADOW,
                PlaybookStage.STAGING,
                PlaybookStage.PROD,
                PlaybookStage.QUARANTINED
            ])
        ]

        db_session.add_all(bullets)
        db_session.commit()

        # Get shadow bullets
        shadow_bullets = stage_manager.get_shadow_bullets("test")

        assert len(shadow_bullets) == 1
        assert shadow_bullets[0].stage == PlaybookStage.SHADOW
        assert shadow_bullets[0].id == "B000"

    def test_staging_bullets_separate_from_production(self, db_session, playbook_repo):
        """Test staging bullets can be retrieved separately."""
        # Create staging and production bullets
        bullets = [
            PlaybookBullet(
id="B001",
                content="Staging strategy",
                domain_id="test",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.STAGING,
                embedding=[0.1] * 384,
            tags=[]
        ),
            PlaybookBullet(
id="B002",
                content="Production strategy",
                domain_id="test",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.PROD,
                embedding=[0.2] * 384,
            tags=[]
        )
        ]

        db_session.add_all(bullets)
        db_session.commit()

        # Get staging bullets
        staging_bullets = playbook_repo.get_by_domain("test", stage=PlaybookStage.STAGING)
        assert len(staging_bullets) == 1
        assert staging_bullets[0].id == "B001"

        # Get production bullets
        prod_bullets = playbook_repo.get_by_domain("test", stage=PlaybookStage.PROD)
        assert len(prod_bullets) == 1
        assert prod_bullets[0].id == "B002"


class TestStageTransitions:
    """Test stage transitions and filtering."""

    def test_manual_stage_change_shadow_to_staging(self, db_session, stage_manager):
        """Test manually changing bullet from shadow to staging."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        # Change to staging
        success = stage_manager.set_stage(
            id="B001",
            domain_id="test",
            target_stage=PlaybookStage.STAGING,
            reason="manual_promotion_for_testing"
        )

        assert success is True

        # Verify stage changed
        updated_bullet = db_session.query(PlaybookBullet).filter(
            PlaybookBullet.id == "B001"
        ).first()

        assert updated_bullet.stage == PlaybookStage.STAGING

    def test_stage_change_logs_to_journal(self, db_session, stage_manager):
        """Test that stage changes are logged to diff_journal."""
        bullet = PlaybookBullet(
id="B001",
            content="Test strategy",
            domain_id="test",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        # Change stage
        stage_manager.set_stage(
            id="B001",
            domain_id="test",
            target_stage=PlaybookStage.PROD,
            reason="test_promotion"
        )

        # Check journal entry exists
        journal_repo = DiffJournalRepository(db_session)
        entries = journal_repo.get_by_bullet("B001", "test")

        assert len(entries) > 0

        stage_change_entry = next(
            (e for e in entries if e.operation_type == "stage_change"),
            None
        )

        assert stage_change_entry is not None
        assert stage_change_entry.before_value["stage"] == "shadow"
        assert stage_change_entry.after_value["stage"] == "prod"

    def test_cannot_change_stage_of_nonexistent_bullet(self, db_session, stage_manager):
        """Test that changing stage of non-existent bullet returns False."""
        success = stage_manager.set_stage(
            id="NONEXISTENT",
            domain_id="test",
            target_stage=PlaybookStage.PROD,
            reason="test"
        )

        assert success is False


class TestDomainIsolation:
    """Test that shadow bullets respect domain isolation (CHK081)."""

    def test_shadow_bullets_isolated_by_domain(self, db_session, stage_manager):
        """Test shadow bullets from different domains are isolated."""
        bullets = [
            PlaybookBullet(
id=f"B{i:03d}",
                content=f"Strategy {i}",
                domain_id=domain,
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.SHADOW,
                embedding=[float(i,
            tags=[]
        )] * 384,
            tags=[]
            )
            for i, domain in enumerate(["arithmetic", "geometry", "algebra"])
        ]

        db_session.add_all(bullets)
        db_session.commit()

        # Get shadow bullets for specific domain
        arithmetic_shadow = stage_manager.get_shadow_bullets("arithmetic")
        geometry_shadow = stage_manager.get_shadow_bullets("geometry")

        assert len(arithmetic_shadow) == 1
        assert arithmetic_shadow[0].domain_id == "arithmetic"

        assert len(geometry_shadow) == 1
        assert geometry_shadow[0].domain_id == "geometry"

    def test_production_bullets_isolated_by_domain(self, db_session, stage_manager):
        """Test production bullets from different domains are isolated."""
        bullets = [
            PlaybookBullet(
id=f"B{i:03d}",
                content=f"Strategy {i}",
                domain_id=domain,
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.PROD,
                embedding=[float(i,
            tags=[]
        )] * 384,
            tags=[]
            )
            for i, domain in enumerate(["domain_a", "domain_b"])
        ]

        db_session.add_all(bullets)
        db_session.commit()

        # Get production bullets per domain
        domain_a_prod = stage_manager.get_production_bullets("domain_a")
        domain_b_prod = stage_manager.get_production_bullets("domain_b")

        assert len(domain_a_prod) == 1
        assert domain_a_prod[0].domain_id == "domain_a"

        assert len(domain_b_prod) == 1
        assert domain_b_prod[0].domain_id == "domain_b"


class TestShadowWorkflow:
    """Test complete shadow learning workflow."""

    def test_complete_shadow_to_production_workflow(self, db_session, stage_manager):
        """
        T063: Test complete workflow from shadow creation to production.
        """
        # 1. Create bullet in shadow stage
        bullet = PlaybookBullet(
id="B001",
            content="Decompose complex problems",
            domain_id="arithmetic",
            section=InsightSection.HELPFUL,
            stage=PlaybookStage.SHADOW,
            embedding=[0.1] * 384,
            helpful_count=0,
            harmful_count=0,
        tags=[]
        )

        db_session.add(bullet)
        db_session.commit()

        # 2. Verify not in production retrieval
        prod_bullets = stage_manager.get_production_bullets("arithmetic")
        assert len(prod_bullets) == 0

        # 3. Simulate helpful feedback
        bullet.helpful_count = 3
        bullet.harmful_count = 0
        db_session.commit()

        # 4. Promote to staging
        result = stage_manager.promote_bullet("B001", "arithmetic")
        assert result == PlaybookStage.STAGING

        # 5. Verify in staging
        updated_bullet = db_session.query(PlaybookBullet).filter(
            PlaybookBullet.id == "B001"
        ).first()
        assert updated_bullet.stage == PlaybookStage.STAGING

        # 6. More helpful feedback
        updated_bullet.helpful_count = 5
        db_session.commit()

        # 7. Promote to production
        result = stage_manager.promote_bullet("B001", "arithmetic")
        assert result == PlaybookStage.PROD

        # 8. Verify in production retrieval
        prod_bullets = stage_manager.get_production_bullets("arithmetic")
        assert len(prod_bullets) == 1
        assert prod_bullets[0].id == "B001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
