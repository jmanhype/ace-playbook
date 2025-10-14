"""
Unit tests for batch_merge() input validation (T070).

Tests verify that batch_merge() properly validates:
- Empty task_insights list
- Mixed domain_ids in batch
- Missing required keys (task_id, domain_id, insights)
- Invalid domain_id format
- Reserved domain_ids
"""

import pytest
from ace.curator.semantic_curator import SemanticCurator
from ace.models.playbook import PlaybookStage


@pytest.fixture
def curator():
    """Create SemanticCurator instance for testing."""
    return SemanticCurator(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.8,
    )


def test_batch_merge_empty_input(curator):
    """T070: batch_merge() should reject empty task_insights list."""
    with pytest.raises(ValueError, match="task_insights cannot be empty"):
        curator.batch_merge(
            task_insights=[],
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
        )


def test_batch_merge_mixed_domains(curator):
    """T070: batch_merge() should reject mixed domain_ids."""
    task_insights = [
        {
            "task_id": "task-1",
            "domain_id": "customer-acme",
            "insights": [{"content": "Test 1", "section": "Helpful"}],
        },
        {
            "task_id": "task-2",
            "domain_id": "customer-globex",  # Different domain!
            "insights": [{"content": "Test 2", "section": "Helpful"}],
        },
    ]

    with pytest.raises(ValueError, match="Multiple domain_ids in batch"):
        curator.batch_merge(
            task_insights=task_insights,
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
        )


def test_batch_merge_missing_domain_id(curator):
    """T070: batch_merge() should reject tasks missing domain_id."""
    task_insights = [
        {
            "task_id": "task-1",
            # Missing "domain_id" key
            "insights": [{"content": "Test", "section": "Helpful"}],
        }
    ]

    with pytest.raises(ValueError, match="missing 'domain_id' key"):
        curator.batch_merge(
            task_insights=task_insights,
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
        )


def test_batch_merge_missing_task_id(curator):
    """T070: batch_merge() should reject tasks missing task_id."""
    task_insights = [
        {
            # Missing "task_id" key
            "domain_id": "customer-acme",
            "insights": [{"content": "Test", "section": "Helpful"}],
        }
    ]

    with pytest.raises(ValueError, match="missing required 'task_id' key"):
        curator.batch_merge(
            task_insights=task_insights,
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
        )


def test_batch_merge_missing_insights(curator):
    """T070: batch_merge() should reject tasks missing insights."""
    task_insights = [
        {
            "task_id": "task-1",
            "domain_id": "customer-acme",
            # Missing "insights" key
        }
    ]

    with pytest.raises(ValueError, match="missing required 'insights' key"):
        curator.batch_merge(
            task_insights=task_insights,
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
        )


def test_batch_merge_insights_not_list(curator):
    """T070: batch_merge() should reject non-list insights."""
    task_insights = [
        {
            "task_id": "task-1",
            "domain_id": "customer-acme",
            "insights": "not a list",  # Should be list!
        }
    ]

    with pytest.raises(ValueError, match="'insights' must be a list"):
        curator.batch_merge(
            task_insights=task_insights,
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
        )


def test_batch_merge_invalid_domain_format(curator):
    """T070: batch_merge() should reject invalid domain_id format."""
    task_insights = [
        {
            "task_id": "task-1",
            "domain_id": "Customer_ACME",  # Invalid: uppercase + underscore
            "insights": [{"content": "Test", "section": "Helpful"}],
        }
    ]

    with pytest.raises(ValueError, match="Invalid domain_id.*Must match pattern"):
        curator.batch_merge(
            task_insights=task_insights,
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
        )


def test_batch_merge_reserved_domain(curator):
    """T070: batch_merge() should reject reserved domain_ids."""
    task_insights = [
        {
            "task_id": "task-1",
            "domain_id": "system",  # Reserved domain!
            "insights": [{"content": "Test", "section": "Helpful"}],
        }
    ]

    with pytest.raises(ValueError, match="Reserved domain_id 'system'"):
        curator.batch_merge(
            task_insights=task_insights,
            current_playbook=[],
            target_stage=PlaybookStage.SHADOW,
        )


def test_batch_merge_valid_input(curator):
    """T070: batch_merge() should accept valid input."""
    task_insights = [
        {
            "task_id": "task-1",
            "domain_id": "customer-acme",
            "insights": [
                {"content": "Use strategy A", "section": "Helpful", "tags": []},
                {"content": "Avoid strategy B", "section": "Harmful", "tags": []},
            ],
        },
        {
            "task_id": "task-2",
            "domain_id": "customer-acme",  # Same domain
            "insights": [{"content": "Use strategy C", "section": "Helpful", "tags": []}],
        },
    ]

    # Should not raise
    result = curator.batch_merge(
        task_insights=task_insights,
        current_playbook=[],
        target_stage=PlaybookStage.SHADOW,
    )

    assert result is not None
    assert "updated_playbook" in result
    assert "total_new_bullets" in result
    assert result["total_processed"] == 3  # 2 + 1 insights
