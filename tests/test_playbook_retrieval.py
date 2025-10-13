"""
Test playbook retrieval with domain filtering (CHK081-CHK082).

Verifies that CuratorService.get_playbook() correctly filters by:
- Domain ID (multi-tenant isolation)
- Stage (shadow/staging/prod/quarantined)
- Section (Helpful/Harmful/Neutral)
"""

import pytest

from ace.curator import CuratorService
from ace.models.playbook import PlaybookStage
from ace.utils.database import init_database


@pytest.fixture(scope="module")
def setup_database():
    """Initialize test database."""
    import os
    import ace.utils.database as db_module

    # Reset global database engine/factory to avoid conflicts between test modules
    db_module._engine = None
    db_module._session_factory = None

    test_db_path = "test_playbook_retrieval.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    init_database(database_url="sqlite:///test_playbook_retrieval.db")
    yield

    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Reset globals again for next test module
    db_module._engine = None
    db_module._session_factory = None


# Mock fixture now provided by tests/conftest.py


@pytest.fixture
def curator_service():
    """Create CuratorService instance."""
    return CuratorService(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.8,
    )


@pytest.fixture
def populated_playbook(setup_database, curator_service):
    """Create populated playbook with diverse bullets."""
    domain_id = "retrieval-test-domain"

    # Add bullets with different stages and sections
    # Use extremely distinct content to avoid mock embedding collisions
    # Mock uses hash(prefix + str(i)) where prefix = first 10 chars only
    insights = [
        # Shadow - Helpful
        {"content": "Q7W!e2R9t# write comprehensive unit tests", "section": "Helpful", "tags": []},
        {"content": "M4k@L3p$J8 dependency injection pattern", "section": "Helpful", "tags": []},
        # Shadow - Harmful
        {"content": "Z1x^C6v&N5 skip code reviews harmful", "section": "Harmful", "tags": []},
        # Staging - Helpful
        {"content": "A0s%D9f*H3 circuit breakers resilience", "section": "Helpful", "tags": []},
    ]

    # Add shadow bullets
    curator_service.merge_insights(
        task_id="setup-task-1",
        domain_id=domain_id,
        insights=insights[:3],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.40,  # Lower threshold for mock embeddings
    )

    # Add staging bullet
    curator_service.merge_insights(
        task_id="setup-task-2",
        domain_id=domain_id,
        insights=insights[3:],
        target_stage=PlaybookStage.STAGING,
        similarity_threshold=0.40,  # Lower threshold for mock embeddings
    )

    return domain_id


def test_filter_by_domain_only(populated_playbook, curator_service):
    """Test retrieving all bullets for a domain (no stage/section filter)."""
    bullets = curator_service.get_playbook(domain_id=populated_playbook)

    assert len(bullets) == 4
    domains = {b.domain_id for b in bullets}
    assert domains == {populated_playbook}


def test_filter_by_stage(populated_playbook, curator_service):
    """Test filtering by stage (shadow vs staging)."""
    # Get shadow bullets
    shadow_bullets = curator_service.get_playbook(
        domain_id=populated_playbook,
        stage=PlaybookStage.SHADOW,
    )
    assert len(shadow_bullets) == 3
    assert all(b.stage == PlaybookStage.SHADOW for b in shadow_bullets)

    # Get staging bullets
    staging_bullets = curator_service.get_playbook(
        domain_id=populated_playbook,
        stage=PlaybookStage.STAGING,
    )
    assert len(staging_bullets) == 1
    assert all(b.stage == PlaybookStage.STAGING for b in staging_bullets)


def test_filter_by_section(populated_playbook, curator_service):
    """Test filtering by section (Helpful vs Harmful)."""
    # Get Helpful bullets
    helpful_bullets = curator_service.get_playbook(
        domain_id=populated_playbook,
        section="Helpful",
    )
    assert len(helpful_bullets) == 3
    assert all(b.section == "Helpful" for b in helpful_bullets)

    # Get Harmful bullets
    harmful_bullets = curator_service.get_playbook(
        domain_id=populated_playbook,
        section="Harmful",
    )
    assert len(harmful_bullets) == 1
    assert all(b.section == "Harmful" for b in harmful_bullets)


def test_filter_by_stage_and_section(populated_playbook, curator_service):
    """Test combined stage + section filtering."""
    # Get Shadow + Helpful bullets
    shadow_helpful = curator_service.get_playbook(
        domain_id=populated_playbook,
        stage=PlaybookStage.SHADOW,
        section="Helpful",
    )
    assert len(shadow_helpful) == 2
    assert all(b.stage == PlaybookStage.SHADOW for b in shadow_helpful)
    assert all(b.section == "Helpful" for b in shadow_helpful)

    # Get Shadow + Harmful bullets
    shadow_harmful = curator_service.get_playbook(
        domain_id=populated_playbook,
        stage=PlaybookStage.SHADOW,
        section="Harmful",
    )
    assert len(shadow_harmful) == 1
    assert shadow_harmful[0].stage == PlaybookStage.SHADOW
    assert shadow_harmful[0].section == "Harmful"


def test_domain_isolation_enforced(populated_playbook, curator_service):
    """Test that domain isolation (CHK081) is enforced."""
    # Add bullets to different domain
    other_domain = "other-domain"
    insights = [
        {"content": "Strategy for other domain", "section": "Helpful", "tags": []},
    ]

    curator_service.merge_insights(
        task_id="other-task",
        domain_id=other_domain,
        insights=insights,
        target_stage=PlaybookStage.SHADOW,
    )

    # Query original domain - should NOT see other domain's bullets
    original_bullets = curator_service.get_playbook(domain_id=populated_playbook)
    assert len(original_bullets) == 4  # Only original domain bullets

    # Query other domain - should only see its own bullet
    other_bullets = curator_service.get_playbook(domain_id=other_domain)
    assert len(other_bullets) == 1
    assert other_bullets[0].domain_id == other_domain

    # Verify no cross-contamination
    all_domains = {b.domain_id for b in original_bullets}
    assert other_domain not in all_domains


def test_empty_result_for_nonexistent_domain(curator_service):
    """Test that querying non-existent domain returns empty list."""
    bullets = curator_service.get_playbook(domain_id="nonexistent-domain-xyz")
    assert bullets == []


def test_empty_result_for_no_matches(populated_playbook, curator_service):
    """Test that filtering with no matches returns empty list."""
    # Query prod stage (none exist)
    prod_bullets = curator_service.get_playbook(
        domain_id=populated_playbook,
        stage=PlaybookStage.PROD,
    )
    assert prod_bullets == []

    # Query Neutral section (none exist)
    neutral_bullets = curator_service.get_playbook(
        domain_id=populated_playbook,
        section="Neutral",
    )
    assert neutral_bullets == []
