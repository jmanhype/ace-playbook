"""
Integration Tests for Tool Strategy Learning (T033-T034)

Tests for User Story 3: Tool Usage Learning and Optimization
Validates that strategies are organized by domain and reused effectively.
"""

import pytest
from typing import List, Dict
from unittest.mock import Mock, MagicMock, patch
import dspy

from ace.generator.react_generator import ReActGenerator
from ace.generator.signatures import TaskInput, TaskOutput
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.curator.semantic_curator import SemanticCurator
from ace.reflector.grounded_reflector import GroundedReflector


def create_mock_bullet(
    domain_id: str,
    content: str,
    tool_sequence: List[str] = None,
    tool_success_rate: float = None,
    helpful_count: int = 5,
    harmful_count: int = 1,
) -> PlaybookBullet:
    """Create a mock PlaybookBullet for testing."""
    return PlaybookBullet(
        id=f"bullet-{domain_id}-{len(content)}",
        domain_id=domain_id,
        content=content,
        section="Helpful",
        helpful_count=helpful_count,
        harmful_count=harmful_count,
        tags=[domain_id],
        embedding=[0.1] * 384,  # Mock embedding
        stage=PlaybookStage.PROD,
        tool_sequence=tool_sequence,
        tool_success_rate=tool_success_rate,
        avg_iterations=3 if tool_sequence else None,
    )


@pytest.mark.integration
class TestPlaybookStrategyRetrieval:
    """T033: Verify strategies are organized by domain with success metrics."""

    def test_curator_calculates_success_metrics(self):
        """Verify curator correctly calculates tool success rates."""
        curator = SemanticCurator()

        # Test success rate calculation
        assert curator.calculate_tool_success_rate(8, 2) == 0.8  # 8/(8+2)
        assert curator.calculate_tool_success_rate(10, 0) == 1.0  # 10/10
        assert curator.calculate_tool_success_rate(0, 5) == 0.0   # 0/5
        assert curator.calculate_tool_success_rate(0, 0) == 0.0   # 0/0 edge case

    def test_high_success_strategies_retrieval(self):
        """Verify high-success strategies are correctly filtered and sorted."""
        curator = SemanticCurator()

        # Create playbook with mixed success rates
        playbook = [
            create_mock_bullet(
                "rag-db",
                "Use vector search for semantic queries",
                tool_sequence=["search_vector_db", "rank_results"],
                tool_success_rate=0.9,
                helpful_count=9,
                harmful_count=1,
            ),
            create_mock_bullet(
                "rag-db",
                "Try SQL search with exact filters",
                tool_sequence=["search_sql_db", "filter_results"],
                tool_success_rate=0.5,
                helpful_count=5,
                harmful_count=5,
            ),
            create_mock_bullet(
                "rag-db",
                "Combine both search methods",
                tool_sequence=["search_vector_db", "search_sql_db", "merge_results"],
                tool_success_rate=0.95,
                helpful_count=19,
                harmful_count=1,
            ),
            create_mock_bullet(
                "other-domain",
                "Different domain strategy",
                tool_sequence=["some_tool"],
                tool_success_rate=0.99,
                helpful_count=99,
                harmful_count=1,
            ),
        ]

        # Retrieve high-success strategies for rag-db domain
        strategies = curator.get_high_success_strategies(
            playbook=playbook,
            domain_id="rag-db",
            min_success_rate=0.7,
            max_results=10,
        )

        # Should return 2 strategies (0.9 and 0.95), sorted by success rate
        assert len(strategies) == 2
        assert strategies[0].tool_success_rate == 0.95  # Highest first
        assert strategies[1].tool_success_rate == 0.9

        # Should not include low-success (0.5) or other-domain strategies
        success_rates = [s.tool_success_rate for s in strategies]
        assert 0.5 not in success_rates
        assert 0.99 not in success_rates

    def test_domain_isolation_in_strategy_retrieval(self):
        """Verify strategies are properly isolated by domain."""
        curator = SemanticCurator()

        playbook = [
            create_mock_bullet("domain-a", "Strategy A", ["tool1"], 0.9),
            create_mock_bullet("domain-b", "Strategy B", ["tool2"], 0.95),
            create_mock_bullet("domain-a", "Strategy A2", ["tool3"], 0.85),
        ]

        # Retrieve for domain-a only
        strategies_a = curator.get_high_success_strategies(
            playbook=playbook,
            domain_id="domain-a",
            min_success_rate=0.7,
        )

        # Should only get domain-a strategies
        assert len(strategies_a) == 2
        assert all(s.domain_id == "domain-a" for s in strategies_a)

        # Retrieve for domain-b only
        strategies_b = curator.get_high_success_strategies(
            playbook=playbook,
            domain_id="domain-b",
            min_success_rate=0.7,
        )

        assert len(strategies_b) == 1
        assert strategies_b[0].domain_id == "domain-b"

    def test_success_rate_threshold_filtering(self):
        """Verify min_success_rate threshold is enforced."""
        curator = SemanticCurator()

        playbook = [
            create_mock_bullet("test", "High", ["t1"], 0.9),
            create_mock_bullet("test", "Medium", ["t2"], 0.75),
            create_mock_bullet("test", "Low", ["t3"], 0.6),
        ]

        # With threshold 0.8, only get high-success strategies
        strategies = curator.get_high_success_strategies(
            playbook=playbook,
            domain_id="test",
            min_success_rate=0.8,
        )

        assert len(strategies) == 1
        assert strategies[0].tool_success_rate == 0.9

    def test_strategies_sorted_by_success_rate(self):
        """Verify strategies are returned in descending success rate order."""
        curator = SemanticCurator()

        playbook = [
            create_mock_bullet("test", "Third", ["t3"], 0.75),
            create_mock_bullet("test", "First", ["t1"], 0.95),
            create_mock_bullet("test", "Second", ["t2"], 0.85),
        ]

        strategies = curator.get_high_success_strategies(
            playbook=playbook,
            domain_id="test",
            min_success_rate=0.7,
        )

        # Should be sorted: 0.95, 0.85, 0.75
        assert len(strategies) == 3
        assert strategies[0].tool_success_rate == 0.95
        assert strategies[1].tool_success_rate == 0.85
        assert strategies[2].tool_success_rate == 0.75


@pytest.mark.integration
class TestStrategyReuse:
    """T034: Verify new tasks reuse proven patterns from playbook."""

    def test_playbook_strategies_formatted_for_agent(self):
        """Verify playbook strategies can be formatted for agent use."""
        curator = SemanticCurator()

        # Create playbook with proven strategies
        playbook = [
            create_mock_bullet(
                domain_id="test-domain",
                content="For search queries, use tool_a with specific filters",
                tool_sequence=["tool_a", "tool_b"],
                tool_success_rate=0.9,
                helpful_count=18,
                harmful_count=2,
            ),
            create_mock_bullet(
                domain_id="test-domain",
                content="Use tool_c for ranking results",
                tool_sequence=["tool_c"],
                tool_success_rate=0.85,
                helpful_count=17,
                harmful_count=3,
            ),
        ]

        # Get high-success strategies for this domain
        strategies = curator.get_high_success_strategies(
            playbook=playbook,
            domain_id="test-domain",
            min_success_rate=0.7,
        )

        # Verify strategies are available for agent use
        assert len(strategies) == 2
        assert strategies[0].tool_success_rate == 0.9  # Highest first
        assert strategies[0].tool_sequence == ["tool_a", "tool_b"]
        assert strategies[1].tool_success_rate == 0.85
        assert strategies[1].tool_sequence == ["tool_c"]

        # Verify strategies contain actionable content
        assert "search queries" in strategies[0].content
        assert "ranking results" in strategies[1].content

    def test_cross_domain_pattern_transfer(self):
        """Verify strategies can be transferred across related domains."""
        curator = SemanticCurator()

        # Create playbook with strategies from related domains
        playbook = [
            # Similar search strategies in different domains
            create_mock_bullet(
                "rag-research",
                "Use vector search for academic papers",
                ["search_vector_db", "rank_by_citations"],
                0.9,
            ),
            create_mock_bullet(
                "rag-docs",
                "Use vector search for documentation",
                ["search_vector_db", "rank_by_relevance"],
                0.85,
            ),
            # Unrelated domain
            create_mock_bullet(
                "calculator",
                "Use arithmetic tools",
                ["add", "multiply"],
                0.95,
            ),
        ]

        # Query embedding for a new search task
        query_embedding = [0.1] * 384

        # Find patterns from related domains
        patterns = curator.find_cross_domain_patterns(
            playbook=playbook,
            source_domain="rag-research",
            related_domains=["rag-docs", "rag-database"],
            query_embedding=query_embedding,
            similarity_threshold=0.0,  # Low threshold for test
            max_results=5,
        )

        # Should find rag-docs strategy (related domain)
        # Should NOT find rag-research (source domain) or calculator (unrelated)
        domain_ids = [p.domain_id for p in patterns]
        assert "rag-docs" in domain_ids
        assert "rag-research" not in domain_ids  # Source domain excluded
        assert "calculator" not in domain_ids    # Not in related_domains

    @patch('ace.generator.react_generator.dspy.configure')
    def test_strategy_selection_prioritizes_high_success(self, mock_configure):
        """Verify agent prioritizes high-success strategies when multiple are available."""
        curator = SemanticCurator()

        # Create strategies with different success rates
        playbook = [
            create_mock_bullet(
                "test",
                "Low success approach",
                ["tool_a"],
                tool_success_rate=0.6,
                helpful_count=6,
                harmful_count=4,
            ),
            create_mock_bullet(
                "test",
                "High success approach",
                ["tool_b"],
                tool_success_rate=0.95,
                helpful_count=19,
                harmful_count=1,
            ),
            create_mock_bullet(
                "test",
                "Medium success approach",
                ["tool_c"],
                tool_success_rate=0.75,
                helpful_count=15,
                harmful_count=5,
            ),
        ]

        # Get high-success strategies
        strategies = curator.get_high_success_strategies(
            playbook=playbook,
            domain_id="test",
            min_success_rate=0.7,
        )

        # First strategy should be highest success rate
        assert strategies[0].tool_success_rate == 0.95
        assert strategies[0].tool_sequence == ["tool_b"]

        # Low-success strategy should be excluded
        success_rates = [s.tool_success_rate for s in strategies]
        assert 0.6 not in success_rates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
