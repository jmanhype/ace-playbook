"""
Unit tests for SemanticCurator core logic.

Tests deduplication algorithms, similarity computation, and quarantine logic
in isolation (without database dependencies).
"""

import pytest
from datetime import datetime

from ace.curator.semantic_curator import SemanticCurator
from ace.curator.curator_models import (
    CuratorInput,
    CuratorOutput,
    CuratorOperationType,
    DeltaUpdate,
    SIMILARITY_THRESHOLD_DEFAULT,
)
from ace.models.playbook import PlaybookBullet, PlaybookStage


# Mock fixture now provided by tests/conftest.py


@pytest.fixture
def curator():
    """Create SemanticCurator instance."""
    return SemanticCurator(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.8,
    )


def test_curator_initialization(curator):
    """Test SemanticCurator initializes correctly."""
    assert curator.similarity_threshold == 0.8
    assert curator.embedding_service is not None


def test_empty_playbook_adds_all_insights(curator):
    """Test that all insights are added when playbook is empty."""
    curator_input = CuratorInput(
        task_id="task-001",
        domain_id="test-domain",
        insights=[
            {"content": "Use version control for all code changes", "section": "Helpful", "tags": ["vcs"]},
            {"content": "Cache database queries for performance", "section": "Helpful", "tags": ["performance"]},
        ],
        current_playbook=[],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.75,  # Lower threshold to avoid false duplicates with mock embeddings
    )

    output = curator.apply_delta(curator_input)

    assert output.new_bullets_added == 2
    assert output.existing_bullets_incremented == 0
    assert output.duplicates_detected == 0
    assert len(output.updated_playbook) == 2


def test_exact_duplicate_increments_counter(curator):
    """Test that exact duplicate insight increments counter."""
    existing_bullet = PlaybookBullet(
        domain_id="test-domain",
        content="Use descriptive variable names",
        section="Helpful",
        helpful_count=1,
        harmful_count=0,
        tags=["readability"],
        embedding=curator.embedding_service.encode_single("Use descriptive variable names"),
        stage=PlaybookStage.SHADOW,
        created_at=datetime.utcnow(),
        last_used_at=datetime.utcnow(),
    )

    curator_input = CuratorInput(
        task_id="task-002",
        domain_id="test-domain",
        insights=[
            {"content": "Use descriptive variable names", "section": "Helpful", "tags": ["readability"]},
        ],
        current_playbook=[existing_bullet],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    output = curator.apply_delta(curator_input)

    assert output.new_bullets_added == 0
    assert output.existing_bullets_incremented == 1
    assert output.duplicates_detected == 1
    assert len(output.updated_playbook) == 1
    assert output.updated_playbook[0].helpful_count == 2


def test_section_isolation(curator):
    """Test that Helpful and Harmful sections are isolated."""
    helpful_bullet = PlaybookBullet(
        domain_id="test-domain",
        content="Use automated testing",
        section="Helpful",
        helpful_count=1,
        harmful_count=0,
        tags=[],
        embedding=curator.embedding_service.encode_single("Use automated testing"),
        stage=PlaybookStage.SHADOW,
        created_at=datetime.utcnow(),
        last_used_at=datetime.utcnow(),
    )

    # Send same content but in Harmful section
    curator_input = CuratorInput(
        task_id="task-003",
        domain_id="test-domain",
        insights=[
            {"content": "Use automated testing", "section": "Harmful", "tags": []},
        ],
        current_playbook=[helpful_bullet],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    output = curator.apply_delta(curator_input)

    # Should create NEW Harmful bullet, not increment Helpful bullet
    assert output.new_bullets_added == 1
    assert output.existing_bullets_incremented == 0
    assert len(output.updated_playbook) == 2

    # Verify both sections exist
    sections = {b.section for b in output.updated_playbook}
    assert "Helpful" in sections
    assert "Harmful" in sections


def test_domain_isolation_enforced(curator):
    """Test that cross-domain access is blocked (CHK081)."""
    other_domain_bullet = PlaybookBullet(
        domain_id="other-domain",
        content="Write unit tests",
        section="Helpful",
        helpful_count=1,
        harmful_count=0,
        tags=[],
        embedding=curator.embedding_service.encode_single("Write unit tests"),
        stage=PlaybookStage.SHADOW,
        created_at=datetime.utcnow(),
        last_used_at=datetime.utcnow(),
    )

    # Try to merge insights for test-domain with playbook containing other-domain bullet
    # This should raise ValueError due to domain isolation violation
    curator_input = CuratorInput(
        task_id="task-004",
        domain_id="test-domain",
        insights=[
            {"content": "Write unit tests", "section": "Helpful", "tags": []},
        ],
        current_playbook=[other_domain_bullet],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    # Domain isolation should be enforced - expect ValueError
    with pytest.raises(ValueError, match="Cross-domain access violation"):
        curator.apply_delta(curator_input)


def test_similarity_threshold_respected(curator):
    """Test that similarity threshold controls duplicate detection."""
    existing_bullet = PlaybookBullet(
        domain_id="test-domain",
        content="Always validate user input before processing",
        section="Helpful",
        helpful_count=1,
        harmful_count=0,
        tags=[],
        embedding=curator.embedding_service.encode_single("Always validate user input before processing"),
        stage=PlaybookStage.SHADOW,
        created_at=datetime.utcnow(),
        last_used_at=datetime.utcnow(),
    )

    # Very different content
    curator_input = CuratorInput(
        task_id="task-005",
        domain_id="test-domain",
        insights=[
            {"content": "Cache expensive database queries using Redis", "section": "Helpful", "tags": []},
        ],
        current_playbook=[existing_bullet],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    output = curator.apply_delta(curator_input)

    # Different content should NOT be duplicate
    assert output.new_bullets_added == 1
    assert output.existing_bullets_incremented == 0
    assert len(output.updated_playbook) == 2


def test_multiple_insights_batch_processing(curator):
    """Test processing multiple insights in one batch."""
    curator_input = CuratorInput(
        task_id="task-006",
        domain_id="test-domain",
        insights=[
            {"content": "Use dependency injection for testability", "section": "Helpful", "tags": []},
            {"content": "Implement circuit breakers for resilience", "section": "Helpful", "tags": []},
            {"content": "Skip code reviews to save time", "section": "Harmful", "tags": []},
        ],
        current_playbook=[],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    output = curator.apply_delta(curator_input)

    assert output.new_bullets_added == 3
    assert output.existing_bullets_incremented == 0
    assert len(output.delta_updates) == 3


def test_delta_updates_structure(curator):
    """Test that delta updates contain correct metadata."""
    curator_input = CuratorInput(
        task_id="task-007",
        domain_id="test-domain",
        insights=[
            {"content": "Log all security events", "section": "Helpful", "tags": ["security"]},
        ],
        current_playbook=[],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    output = curator.apply_delta(curator_input)

    assert len(output.delta_updates) == 1
    delta = output.delta_updates[0]

    assert delta.operation == CuratorOperationType.ADD
    assert delta.bullet_id is not None
    assert delta.new_bullet is not None
    assert delta.new_bullet.content == "Log all security events"
    assert delta.new_bullet.section == "Helpful"
    assert output.delta.operations
    assert output.delta.operations[0].type == CuratorOperationType.ADD


def test_harmful_signal_creates_harmful_bullet(curator):
    """Test that harmful signals create Harmful bullets."""
    curator_input = CuratorInput(
        task_id="task-008",
        domain_id="test-domain",
        insights=[
            {"content": "Hardcode credentials in source code", "section": "Harmful", "tags": ["security"]},
        ],
        current_playbook=[],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    output = curator.apply_delta(curator_input)

    assert output.new_bullets_added == 1
    assert len(output.updated_playbook) == 1

    bullet = output.updated_playbook[0]
    assert bullet.section == "Harmful"
    assert bullet.harmful_count == 1
    assert bullet.helpful_count == 0


def test_stage_assignment(curator):
    """Test that bullets are assigned to correct stage."""
    # Test SHADOW stage
    curator_input = CuratorInput(
        task_id="task-009",
        domain_id="test-domain",
        insights=[
            {"content": "Monitor application metrics", "section": "Helpful", "tags": []},
        ],
        current_playbook=[],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    output = curator.apply_delta(curator_input)
    assert output.updated_playbook[0].stage == PlaybookStage.SHADOW

    # Test STAGING stage
    curator_input.target_stage = PlaybookStage.STAGING
    output = curator.apply_delta(curator_input)
    assert output.updated_playbook[0].stage == PlaybookStage.STAGING


def test_tags_preserved(curator):
    """Test that insight tags are preserved in bullets."""
    curator_input = CuratorInput(
        task_id="task-010",
        domain_id="test-domain",
        insights=[
            {"content": "Implement rate limiting", "section": "Helpful", "tags": ["security", "performance"]},
        ],
        current_playbook=[],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.8,
    )

    output = curator.apply_delta(curator_input)

    bullet = output.updated_playbook[0]
    assert set(bullet.tags) == {"security", "performance"}


def test_output_statistics_accurate(curator):
    """Test that CuratorOutput statistics are accurate."""
    existing_bullet = PlaybookBullet(
        domain_id="test-domain",
        content="Always validate user input before processing",
        section="Helpful",
        helpful_count=1,
        harmful_count=0,
        tags=[],
        embedding=curator.embedding_service.encode_single("Always validate user input before processing"),
        stage=PlaybookStage.SHADOW,
        created_at=datetime.utcnow(),
        last_used_at=datetime.utcnow(),
    )

    curator_input = CuratorInput(
        task_id="task-011",
        domain_id="test-domain",
        insights=[
            {"content": "Always validate user input before processing", "section": "Helpful", "tags": []},  # Duplicate
            {"content": "Cache database queries using Redis for better performance", "section": "Helpful", "tags": []},  # New
        ],
        current_playbook=[existing_bullet],
        target_stage=PlaybookStage.SHADOW,
        similarity_threshold=0.75,  # Lower threshold to avoid false duplicates with mock embeddings
    )

    output = curator.apply_delta(curator_input)

    assert output.new_bullets_added == 1
    assert output.existing_bullets_incremented == 1
    assert output.duplicates_detected == 1
    assert output.bullets_quarantined == 0
