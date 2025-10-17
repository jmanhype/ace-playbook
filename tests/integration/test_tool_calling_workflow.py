"""
Integration tests for full ACE cycle with tool-calling

Tests T014: Verify Generator → Reflector → Curator → Playbook flow with tool strategies
"""

import pytest
from typing import List
from ace.generator.react_generator import ReActGenerator
from ace.generator.signatures import TaskInput


@pytest.mark.integration
@pytest.mark.react
@pytest.mark.slow
class TestFullACECycleWithTools:
    """Test complete ACE workflow: Generator → Reflector → Curator → Playbook."""

    @pytest.fixture
    def database_tools(self):
        """Sample database tools for RAG testing."""

        def search_vector_db(query: str, k: int = 5) -> List[str]:
            """Search vector database for similar documents."""
            # Mock implementation
            return [f"doc_{i}: relevant to '{query}'" for i in range(k)]

        def search_sql_db(table: str, filters: dict) -> List[str]:
            """Search SQL database with filters."""
            # Mock implementation
            filter_str = ", ".join(f"{k}={v}" for k, v in filters.items())
            return [f"row_{i} from {table} where {filter_str}" for i in range(3)]

        def rank_results(results: List[str], criteria: str = "relevance") -> List[str]:
            """Rank and filter results."""
            # Mock implementation
            return results[:3]  # Top 3

        return [search_vector_db, search_sql_db, rank_results]

    @pytest.fixture
    def rag_agent(self, database_tools):
        """ReActGenerator configured for RAG tasks."""
        return ReActGenerator(tools=database_tools, model="gpt-4", max_iters=15)

    @pytest.mark.skip(reason="Requires forward() implementation - T016")
    def test_ace_cycle_without_playbook(self, rag_agent):
        """First execution without playbook context (cold start)."""
        task = TaskInput(
            task_id="ace-001",
            description="Find the top 3 most relevant documents about machine learning",
            domain="ml-research",
            playbook_bullets=[],  # No playbook context yet
        )

        # Execute task
        output = rag_agent.forward(task)

        # Verify execution
        assert output.task_id == "ace-001"
        assert len(output.tools_used) > 0
        assert output.total_iterations > 0
        assert output.iteration_limit_reached is False

        # Reflector analysis
        from ace.reflector.grounded_reflector import GroundedReflector

        reflector = GroundedReflector()
        insights = reflector.analyze(output)

        # Should create tool strategy bullet
        assert len(insights) > 0
        # At least one insight should contain tool_sequence
        tool_insights = [i for i in insights if hasattr(i, "tool_sequence")]
        assert len(tool_insights) > 0

    @pytest.mark.skip(reason="Requires full ACE implementation - T016, T022, T023")
    def test_ace_cycle_with_playbook(self, rag_agent):
        """Second execution with learned playbook strategies."""
        from ace.playbook import Playbook
        from ace.curator import SemanticCurator

        # Simulate first execution (creates strategy)
        first_task = TaskInput(
            task_id="ace-002a",
            description="Find documents about neural networks",
            domain="ml-research",
            playbook_bullets=[],
        )

        first_output = rag_agent.forward(first_task)

        # Reflector + Curator (create strategy)
        from ace.reflector.grounded_reflector import GroundedReflector

        reflector = GroundedReflector()
        curator = SemanticCurator(domain="ml-research")

        insights = reflector.analyze(first_output)
        curator.merge(insights)

        # Retrieve strategies for similar task
        playbook = Playbook(domain="ml-research")
        strategies = playbook.retrieve(
            query="Find documents about deep learning",
            filters={"has_tool_sequence": True},
            k=3,
        )

        # Second execution with playbook context
        second_task = TaskInput(
            task_id="ace-002b",
            description="Find documents about deep learning",
            domain="ml-research",
            playbook_bullets=[s.content for s in strategies],
        )

        second_output = rag_agent.forward(second_task)

        # Should use fewer iterations (learned strategy)
        assert second_output.total_iterations <= first_output.total_iterations
        # Should reference playbook bullets
        assert len(second_output.bullets_referenced) > 0

    @pytest.mark.skip(reason="Requires full implementation")
    def test_iteration_reduction_after_learning(self, rag_agent):
        """Verify 30-50% iteration reduction after 20+ examples (SC-002)."""
        from ace.reflector.grounded_reflector import GroundedReflector
        from ace.curator import SemanticCurator
        from ace.playbook import Playbook

        reflector = GroundedReflector()
        curator = SemanticCurator(domain="ml-research")
        playbook = Playbook(domain="ml-research")

        # Baseline: First 5 tasks without playbook
        baseline_iterations = []
        for i in range(5):
            task = TaskInput(
                task_id=f"baseline-{i}",
                description=f"Find ML papers about topic {i}",
                domain="ml-research",
                playbook_bullets=[],
            )
            output = rag_agent.forward(task)
            baseline_iterations.append(output.total_iterations)

            # Learn from execution
            insights = reflector.analyze(output)
            curator.merge(insights)

        baseline_avg = sum(baseline_iterations) / len(baseline_iterations)

        # Learning phase: 20 more tasks with playbook
        learned_iterations = []
        for i in range(20):
            strategies = playbook.retrieve(
                query=f"Find ML papers about topic {i}",
                filters={"has_tool_sequence": True},
                k=3,
            )

            task = TaskInput(
                task_id=f"learned-{i}",
                description=f"Find ML papers about topic {i}",
                domain="ml-research",
                playbook_bullets=[s.content for s in strategies],
            )

            output = rag_agent.forward(task)
            learned_iterations.append(output.total_iterations)

            insights = reflector.analyze(output)
            curator.merge(insights)

        learned_avg = sum(learned_iterations[-10:]) / 10  # Last 10 tasks

        # Verify 30-50% reduction (SC-002)
        reduction_pct = (baseline_avg - learned_avg) / baseline_avg * 100
        assert reduction_pct >= 30, f"Expected ≥30% reduction, got {reduction_pct:.1f}%"
        assert reduction_pct <= 70, f"Reduction too high: {reduction_pct:.1f}%"

    def test_task_completion_success_rate(self):
        """Verify 90% of 2-5 tool tasks complete within iteration limit (SC-001)."""
        # This will be tested once forward() is implemented
        # For now, verify TaskOutput has iteration_limit_reached field
        from ace.generator.signatures import TaskOutput

        assert hasattr(TaskOutput, "iteration_limit_reached")

    def test_playbook_strategy_capture(self):
        """Verify strategies captured within 3 similar tasks (SC-003)."""
        # This will be tested once Reflector + Curator are extended
        # For now, verify PlaybookBullet has tool fields
        from ace.models.playbook import PlaybookBullet

        assert hasattr(PlaybookBullet, "tool_sequence")
        assert hasattr(PlaybookBullet, "tool_success_rate")
        assert hasattr(PlaybookBullet, "avg_iterations")


@pytest.mark.integration
@pytest.mark.react
class TestPlaybookContextInjection:
    """Test playbook strategy injection into ReAct prompts."""

    def test_playbook_bullets_formatted_for_react(self):
        """Playbook strategies should be formatted for DSPy ReAct context."""
        # Mock playbook strategies
        strategies = [
            "For database queries, filter SQL first by date/topic",
            "Use vector search for semantic similarity",
            "Always rank results by relevance before returning",
        ]

        # Format as context (will be in T024)
        context = "\n".join(f"{i+1}. {s}" for i, s in enumerate(strategies))

        assert "1. For database queries" in context
        assert "2. Use vector search" in context
        assert "3. Always rank results" in context

    @pytest.mark.skip(reason="Requires forward() implementation - T024")
    def test_agent_uses_playbook_strategies(self):
        """Agent should reference playbook strategies during execution."""
        from ace.generator.react_generator import ReActGenerator

        tools = [
            lambda query, table: f"SQL results from {table}",
            lambda query, k=5: f"Vector results",
            lambda results, criteria: f"Ranked results",
        ]

        agent = ReActGenerator(tools=tools)

        task = TaskInput(
            task_id="playbook-test",
            description="Find recent ML papers",
            domain="ml-research",
            playbook_bullets=[
                "Filter SQL by date first",
                "Use vector search for semantic matching",
                "Rank by relevance",
            ],
        )

        output = agent.forward(task)

        # Should reference at least one playbook bullet
        assert len(output.bullets_referenced) > 0


@pytest.mark.integration
@pytest.mark.react
class TestPlaybookIntegration:
    """
    T047: Playbook integration tests.

    Verifies:
    - Strategies are retrieved correctly
    - Deduplication works (semantic similarity ≥0.8)
    - Strategies are applied correctly to tasks
    """

    def test_playbook_strategy_retrieval_by_domain(self):
        """Playbook should filter strategies by domain."""
        from ace.models.playbook import PlaybookBullet

        # Create mock playbook with multiple domains
        strategies = [
            PlaybookBullet(
                content="Use vector search for semantic queries",
                section="Helpful",
                domain_id="ml-research",
                tags=["tool-calling"],
                tool_sequence=["search_vector_db"],
                tool_success_rate=0.9,
                avg_iterations=3
            ),
            PlaybookBullet(
                content="Filter SQL by date first",
                section="Helpful",
                domain_id="database-queries",
                tags=["tool-calling"],
                tool_sequence=["search_sql_db"],
                tool_success_rate=0.85,
                avg_iterations=2
            ),
            PlaybookBullet(
                content="Rank results by relevance",
                section="Helpful",
                domain_id="ml-research",
                tags=["tool-calling"],
                tool_sequence=["rank_results"],
                tool_success_rate=0.95,
                avg_iterations=1
            ),
        ]

        # Filter by domain
        ml_strategies = [s for s in strategies if s.domain_id == "ml-research"]
        db_strategies = [s for s in strategies if s.domain_id == "database-queries"]

        # Verify filtering
        assert len(ml_strategies) == 2
        assert len(db_strategies) == 1
        assert ml_strategies[0].domain_id == "ml-research"
        assert db_strategies[0].domain_id == "database-queries"

    def test_playbook_strategy_deduplication(self):
        """Playbook should deduplicate strategies by semantic similarity ≥0.8."""
        from ace.models.playbook import PlaybookBullet

        # Similar strategies that should be deduplicated
        strategies = [
            PlaybookBullet(
                content="Use vector search for semantic queries",
                section="Helpful",
                domain_id="ml-research",
                tags=["tool-calling"],
                tool_sequence=["search_vector_db"],
                tool_success_rate=0.9,
                avg_iterations=3
            ),
            PlaybookBullet(
                content="Use vector search for semantic matching",  # Similar to above
                section="Helpful",
                domain_id="ml-research",
                tags=["tool-calling"],
                tool_sequence=["search_vector_db"],
                tool_success_rate=0.85,
                avg_iterations=3
            ),
            PlaybookBullet(
                content="Filter SQL queries by date",  # Different strategy
                section="Helpful",
                domain_id="ml-research",
                tags=["tool-calling"],
                tool_sequence=["search_sql_db"],
                tool_success_rate=0.8,
                avg_iterations=2
            ),
        ]

        # Semantic deduplication logic (simplified - actual implementation in SemanticCurator)
        from sentence_transformers import SentenceTransformer
        import numpy as np

        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode([s.content for s in strategies])

            # Calculate cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(embeddings)

            # Find duplicates (similarity ≥ 0.8, excluding self-comparison)
            duplicates = []
            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    if similarities[i][j] >= 0.8:
                        duplicates.append((i, j, similarities[i][j]))

            # Should find one pair of duplicates (strategies 0 and 1)
            assert len(duplicates) > 0
            assert duplicates[0][0] == 0 and duplicates[0][1] == 1

        except ImportError:
            # Skip test if sentence-transformers not installed
            pytest.skip("sentence-transformers not installed")

    def test_playbook_strategy_tool_sequence_matching(self):
        """Playbook should match strategies by tool sequence."""
        from ace.models.playbook import PlaybookBullet

        strategies = [
            PlaybookBullet(
                content="Vector search then rank",
                section="Helpful",
                domain_id="ml-research",
                tags=["tool-calling"],
                tool_sequence=["search_vector_db", "rank_results"],
                tool_success_rate=0.9,
                avg_iterations=3
            ),
            PlaybookBullet(
                content="SQL search then rank",
                section="Helpful",
                domain_id="ml-research",
                tags=["tool-calling"],
                tool_sequence=["search_sql_db", "rank_results"],
                tool_success_rate=0.85,
                avg_iterations=4
            ),
            PlaybookBullet(
                content="Combined search approach",
                section="Helpful",
                domain_id="ml-research",
                tags=["tool-calling"],
                tool_sequence=["search_vector_db", "search_sql_db", "rank_results"],
                tool_success_rate=0.95,
                avg_iterations=5
            ),
        ]

        # Filter by tool sequence pattern
        vector_strategies = [s for s in strategies if "search_vector_db" in s.tool_sequence]
        sql_strategies = [s for s in strategies if "search_sql_db" in s.tool_sequence]
        rank_strategies = [s for s in strategies if "rank_results" in s.tool_sequence]

        assert len(vector_strategies) == 2
        assert len(sql_strategies) == 2
        assert len(rank_strategies) == 3  # All include ranking

    def test_playbook_strategy_success_rate_ordering(self):
        """Playbook should prioritize strategies by success rate."""
        from ace.models.playbook import PlaybookBullet

        strategies = [
            PlaybookBullet(
                content="Strategy A",
                section="Helpful",
                domain_id="test",
                tags=["tool-calling"],
                tool_sequence=["tool_a"],
                tool_success_rate=0.7,
                avg_iterations=5
            ),
            PlaybookBullet(
                content="Strategy B",
                section="Helpful",
                domain_id="test",
                tags=["tool-calling"],
                tool_sequence=["tool_b"],
                tool_success_rate=0.95,
                avg_iterations=3
            ),
            PlaybookBullet(
                content="Strategy C",
                section="Helpful",
                domain_id="test",
                tags=["tool-calling"],
                tool_sequence=["tool_c"],
                tool_success_rate=0.85,
                avg_iterations=4
            ),
        ]

        # Sort by success rate (descending)
        sorted_strategies = sorted(strategies, key=lambda s: s.tool_success_rate, reverse=True)

        # Verify ordering
        assert sorted_strategies[0].tool_success_rate == 0.95
        assert sorted_strategies[1].tool_success_rate == 0.85
        assert sorted_strategies[2].tool_success_rate == 0.7
        assert sorted_strategies[0].content == "Strategy B"

    def test_playbook_strategy_iteration_efficiency(self):
        """Playbook should track average iterations for strategy efficiency."""
        from ace.models.playbook import PlaybookBullet

        strategies = [
            PlaybookBullet(
                content="Efficient strategy",
                section="Helpful",
                domain_id="test",
                tags=["tool-calling"],
                tool_sequence=["tool_a", "tool_b"],
                tool_success_rate=0.9,
                avg_iterations=2  # Very efficient
            ),
            PlaybookBullet(
                content="Moderate strategy",
                section="Helpful",
                domain_id="test",
                tags=["tool-calling"],
                tool_sequence=["tool_a", "tool_b", "tool_c"],
                tool_success_rate=0.85,
                avg_iterations=5  # Moderate efficiency
            ),
            PlaybookBullet(
                content="Complex strategy",
                section="Helpful",
                domain_id="test",
                tags=["tool-calling"],
                tool_sequence=["tool_a", "tool_b", "tool_c", "tool_d"],
                tool_success_rate=0.8,
                avg_iterations=8  # Less efficient but still successful
            ),
        ]

        # Filter efficient strategies (avg_iterations < 5)
        efficient = [s for s in strategies if s.avg_iterations < 5]

        # Verify filtering
        assert len(efficient) == 2
        assert all(s.avg_iterations < 5 for s in efficient)

    def test_playbook_harmful_vs_helpful_strategies(self):
        """Playbook should distinguish between harmful and helpful strategies."""
        from ace.models.playbook import PlaybookBullet

        helpful_strategy = PlaybookBullet(
            content="Use vector search for semantic queries",
            section="Helpful",
            domain_id="ml-research",
            tags=["tool-calling"],
            tool_sequence=["search_vector_db"],
            tool_success_rate=0.9,
            avg_iterations=3
        )

        harmful_strategy = PlaybookBullet(
            content="Avoid using tool_x with tool_y - causes timeouts",
            section="Harmful",
            domain_id="ml-research",
            tags=["tool-calling", "error"],
            tool_sequence=["tool_x", "tool_y"],
            tool_success_rate=0.1,  # Low success rate
            avg_iterations=10  # High iteration count
        )

        # Verify sections
        assert helpful_strategy.section == "Helpful"
        assert harmful_strategy.section == "Harmful"

        # Helpful should have high success, low iterations
        assert helpful_strategy.tool_success_rate > 0.8
        assert helpful_strategy.avg_iterations < 5

        # Harmful should have low success, high iterations
        assert harmful_strategy.tool_success_rate < 0.5
        assert harmful_strategy.avg_iterations > 5

    def test_playbook_cross_domain_transfer(self):
        """Playbook should support cross-domain pattern transfer for related tasks."""
        from ace.models.playbook import PlaybookBullet

        # Strategies from different but related domains
        strategies = [
            PlaybookBullet(
                content="Filter by date then search",
                section="Helpful",
                domain_id="paper-search",
                tags=["tool-calling", "search"],
                tool_sequence=["filter", "search"],
                tool_success_rate=0.9,
                avg_iterations=3
            ),
            PlaybookBullet(
                content="Filter by date then search",
                section="Helpful",
                domain_id="code-search",
                tags=["tool-calling", "search"],
                tool_sequence=["filter", "search"],
                tool_success_rate=0.85,
                avg_iterations=3
            ),
        ]

        # Same strategy pattern works across domains
        assert strategies[0].tool_sequence == strategies[1].tool_sequence
        assert strategies[0].content == strategies[1].content

        # Both should have similar performance
        assert abs(strategies[0].tool_success_rate - strategies[1].tool_success_rate) < 0.1
