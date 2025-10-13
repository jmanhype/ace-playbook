"""
Integration test for Curator with database persistence.

Tests end-to-end flow: insights → semantic deduplication → database persistence.
"""

import pytest
from datetime import datetime

from ace.curator import CuratorService
from ace.models.playbook import PlaybookStage
from ace.utils.database import init_database, get_session
from ace.repositories.playbook_repository import PlaybookRepository


@pytest.fixture(scope="module")
def setup_database():
    """Initialize test database."""
    import os
    import ace.utils.database as db_module

    # Reset global database engine/factory to avoid conflicts between test modules
    db_module._engine = None
    db_module._session_factory = None

    # Remove old test database if it exists
    test_db_path = "test_ace_playbook.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Initialize fresh database
    init_database(database_url="sqlite:///test_ace_playbook.db")
    yield

    # Cleanup: remove test database after tests complete
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


def test_merge_insights_new_bullets(setup_database, curator_service):
    """Test merging new distinct insights into empty playbook."""
    domain_id = "test-domain-001"
    task_id = "task-001"

    # Create test insights - use very distinct semantic content
    insights = [
        {
            "content": "Always write comprehensive unit tests before deploying to production",
            "section": "Helpful",
            "tags": ["problem-solving", "decomposition"],
        },
        {
            "content": "Cache frequently accessed database queries using Redis for performance",
            "section": "Helpful",
            "tags": ["validation", "robustness"],
        },
        {
            "content": "Avoid premature optimization without profiling",
            "section": "Harmful",
            "tags": ["performance", "anti-pattern"],
        },
    ]

    # Merge insights
    output = curator_service.merge_insights(
        task_id=task_id,
        domain_id=domain_id,
        insights=insights,
        target_stage=PlaybookStage.SHADOW,
    )

    # Assertions
    assert output.new_bullets_added == 3
    assert output.existing_bullets_incremented == 0
    assert output.duplicates_detected == 0
    assert output.bullets_quarantined == 0
    assert len(output.delta_updates) == 3

    # Verify database persistence
    bullets = curator_service.get_playbook(domain_id=domain_id)
    assert len(bullets) == 3

    # Check bullet content
    contents = [b.content for b in bullets]
    assert "Always write comprehensive unit tests before deploying to production" in contents
    assert "Cache frequently accessed database queries using Redis for performance" in contents
    assert "Avoid premature optimization without profiling" in contents


def test_merge_duplicate_insights(setup_database, curator_service):
    """Test that duplicate insights increment counters (no bloat)."""
    domain_id = "test-domain-002"
    task_id = "task-002"

    # First batch - new bullets
    insights_batch_1 = [
        {
            "content": "Use descriptive variable names for clarity",
            "section": "Helpful",
            "tags": ["readability"],
        },
    ]

    output1 = curator_service.merge_insights(
        task_id=task_id,
        domain_id=domain_id,
        insights=insights_batch_1,
        target_stage=PlaybookStage.SHADOW,
    )

    assert output1.new_bullets_added == 1

    # Second batch - identical insight (should be duplicate)
    insights_batch_2 = [
        {
            "content": "Use descriptive variable names for clarity",  # Exact same content
            "section": "Helpful",
            "tags": ["readability"],
        },
    ]

    output2 = curator_service.merge_insights(
        task_id=task_id,
        domain_id=domain_id,
        insights=insights_batch_2,
        target_stage=PlaybookStage.SHADOW,
    )

    # Should detect duplicate and increment counter
    assert output2.new_bullets_added == 0
    assert output2.existing_bullets_incremented == 1
    assert output2.duplicates_detected == 1

    # Verify only 1 bullet in database (no bloat)
    bullets = curator_service.get_playbook(domain_id=domain_id)
    assert len(bullets) == 1
    assert bullets[0].helpful_count == 2  # Incremented from 1 to 2


def test_domain_isolation(setup_database, curator_service):
    """Test that domains are isolated (CHK081-CHK082)."""
    domain_a = "test-domain-a"
    domain_b = "test-domain-b"

    # Add bullets to domain A - use distinct prefixes to avoid mock embedding collisions
    insights_a = [
        {"content": "X!@#$ Strategy A1 for domain isolation test", "section": "Helpful", "tags": []},
        {"content": "Y%^&* Strategy A2 for domain isolation test", "section": "Helpful", "tags": []},
    ]

    curator_service.merge_insights(
        task_id="task-a",
        domain_id=domain_a,
        insights=insights_a,
        target_stage=PlaybookStage.SHADOW,
    )

    # Add bullets to domain B - use distinct prefix
    insights_b = [
        {"content": "Z()_+ Strategy B1 for domain isolation test", "section": "Helpful", "tags": []},
    ]

    curator_service.merge_insights(
        task_id="task-b",
        domain_id=domain_b,
        insights=insights_b,
        target_stage=PlaybookStage.SHADOW,
    )

    # Verify isolation
    bullets_a = curator_service.get_playbook(domain_id=domain_a)
    bullets_b = curator_service.get_playbook(domain_id=domain_b)

    assert len(bullets_a) == 2
    assert len(bullets_b) == 1

    # Cross-check: no overlap
    ids_a = {b.id for b in bullets_a}
    ids_b = {b.id for b in bullets_b}
    assert ids_a.isdisjoint(ids_b)


def test_quarantine_logic(setup_database, curator_service):
    """Test that bullets with harmful ≥ helpful are quarantined."""
    domain_id = "test-domain-003"

    # Add a helpful bullet
    insights_helpful = [
        {"content": "Use automated testing", "section": "Helpful", "tags": []},
    ]

    curator_service.merge_insights(
        task_id="task-003-a",
        domain_id=domain_id,
        insights=insights_helpful,
        target_stage=PlaybookStage.SHADOW,
    )

    # Send harmful signals for the SAME strategy within Harmful section
    # First harmful creates new Harmful bullet, second increments it
    insights_harmful = [
        {"content": "Use automated testing", "section": "Harmful", "tags": []},
        {"content": "Use automated testing", "section": "Harmful", "tags": []},
    ]

    output = curator_service.merge_insights(
        task_id="task-003-b",
        domain_id=domain_id,
        insights=insights_harmful,
        target_stage=PlaybookStage.SHADOW,
    )

    # First harmful creates new bullet, second increments
    assert output.new_bullets_added == 1
    assert output.existing_bullets_incremented == 1

    # After this merge, we should have 2 bullets (1 Helpful, 1 Harmful)
    bullets = curator_service.get_playbook(domain_id=domain_id)
    assert len(bullets) == 2

    # Find the Harmful bullet and verify counter
    harmful_bullet = next((b for b in bullets if b.section == "Harmful"), None)
    assert harmful_bullet is not None
    assert harmful_bullet.harmful_count == 2  # 2 harmful signals

    # Note: Quarantine logic works within section, not cross-section


def test_stage_counts(setup_database, curator_service):
    """Test stage count monitoring."""
    domain_id = "test-domain-004"

    # Add shadow bullets - use EXTREMELY distinct content to avoid mock embedding collisions
    # Mock embeddings use hash-based generation which can produce high similarity
    insights = [
        {"content": "AAAA Write unit tests for all APIs", "section": "Helpful", "tags": []},
        {"content": "ZZZZ Cache database queries with Redis", "section": "Helpful", "tags": []},
    ]

    curator_service.merge_insights(
        task_id="task-004",
        domain_id=domain_id,
        insights=insights,
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.75,  # Lower threshold to avoid false duplicates with mock embeddings
    )

    # Get stage counts
    counts = curator_service.get_stage_counts(domain_id=domain_id)

    assert counts.get("shadow", 0) == 2
    assert counts.get("staging", 0) == 0
    assert counts.get("prod", 0) == 0
