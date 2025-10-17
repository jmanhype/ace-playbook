"""
Integration tests for ReAct + Reflector workflow

Tests T013: Verify tool usage patterns are extracted and analyzed by Reflector
"""

import pytest
from typing import List
from ace.generator.react_generator import ReActGenerator
from ace.generator.signatures import TaskInput, TaskOutput, ReasoningStep


@pytest.mark.integration
@pytest.mark.react
class TestReActReflectorIntegration:
    """Test integration between ReActGenerator and GroundedReflector."""

    @pytest.fixture
    def sample_tools(self):
        """Sample tools for testing."""

        def search_database(query: str, table: str = "default") -> str:
            """Search database for records."""
            return f"Found 10 results for '{query}' in {table}"

        def filter_results(results: str, criteria: str) -> str:
            """Filter results by criteria."""
            return f"Filtered {results} by {criteria}"

        return [search_database, filter_results]

    @pytest.fixture
    def agent_with_tools(self, sample_tools):
        """ReActGenerator with sample tools."""
        return ReActGenerator(tools=sample_tools, model="gpt-4", max_iters=10)

    def test_task_output_contains_tool_usage(self, agent_with_tools):
        """TaskOutput should contain structured trace with tool calls."""
        # This test will fail until forward() is implemented
        # For now, verify the output structure exists

        from ace.generator.signatures import TaskOutput

        # Check TaskOutput has required fields
        assert hasattr(TaskOutput, "structured_trace")
        assert hasattr(TaskOutput, "tools_used")
        assert hasattr(TaskOutput, "total_iterations")

    def test_reasoning_step_captures_tool_call(self):
        """ReasoningStep should capture tool execution details."""
        step = ReasoningStep(
            iteration=1,
            thought="I need to search the database",
            action="call_tool",
            tool_name="search_database",
            tool_args={"query": "test", "table": "users"},
            observation="Found 10 results",
            timestamp=1234567890.0,
            duration_ms=45.2,
        )

        assert step.iteration == 1
        assert step.action == "call_tool"
        assert step.tool_name == "search_database"
        assert step.tool_args == {"query": "test", "table": "users"}
        assert step.observation == "Found 10 results"

    def test_reflector_extracts_tool_sequence(self):
        """Reflector should extract tool sequence from structured trace."""
        # Mock TaskOutput with tool usage
        trace = [
            ReasoningStep(
                iteration=1,
                thought="Search database",
                action="call_tool",
                tool_name="search_database",
                tool_args={"query": "test"},
                observation="Found results",
            ),
            ReasoningStep(
                iteration=2,
                thought="Filter results",
                action="call_tool",
                tool_name="filter_results",
                tool_args={"results": "...", "criteria": "recent"},
                observation="Filtered to 3 results",
            ),
            ReasoningStep(
                iteration=3, thought="Done", action="finish", observation="Final answer"
            ),
        ]

        # Extract tool sequence (this logic will be in Reflector)
        tool_sequence = [
            step.tool_name for step in trace if step.action == "call_tool" and step.tool_name
        ]

        assert tool_sequence == ["search_database", "filter_results"]
        assert len(tool_sequence) == 2

    def test_reflector_creates_tool_calling_strategy(self):
        """Reflector should create ToolCallingStrategy bullets from tool usage."""
        # This will be implemented in T022
        # For now, verify the PlaybookBullet model has tool fields

        from ace.models.playbook import PlaybookBullet

        # Check PlaybookBullet has tool_sequence field
        assert hasattr(PlaybookBullet, "tool_sequence")
        assert hasattr(PlaybookBullet, "tool_success_rate")
        assert hasattr(PlaybookBullet, "avg_iterations")

    @pytest.mark.skip(reason="Requires forward() implementation - T016")
    def test_full_react_reflector_flow(self, agent_with_tools):
        """Full flow: ReAct execution → Reflector analysis → Strategy creation."""
        # This test will be enabled once T016, T022, T023 are complete

        task = TaskInput(
            task_id="test-001",
            description="Search for recent users and filter by active status",
            domain="user_management",
        )

        # Execute task
        output = agent_with_tools.forward(task)

        # Verify output structure
        assert output.task_id == "test-001"
        assert len(output.structured_trace) > 0
        assert len(output.tools_used) > 0
        assert output.total_iterations > 0

        # Reflector analysis (will be implemented in T022)
        from ace.reflector.grounded_reflector import GroundedReflector

        reflector = GroundedReflector()
        insights = reflector.analyze(output)

        # Should extract tool sequence
        assert any("tool_sequence" in str(insight) for insight in insights)


@pytest.mark.integration
@pytest.mark.react
class TestToolUsagePatternExtraction:
    """Test extraction of tool usage patterns from execution traces."""

    def test_extract_successful_pattern(self):
        """Should identify successful tool sequences."""
        trace = [
            ReasoningStep(
                iteration=1,
                thought="Step 1",
                action="call_tool",
                tool_name="tool_a",
                tool_args={},
                observation="Success",
            ),
            ReasoningStep(
                iteration=2,
                thought="Step 2",
                action="call_tool",
                tool_name="tool_b",
                tool_args={},
                observation="Success",
            ),
            ReasoningStep(iteration=3, thought="Done", action="finish", observation="Answer"),
        ]

        # Successful execution (didn't hit max iterations)
        task_output = TaskOutput(
            task_id="test",
            reasoning_trace=["step1", "step2", "done"],
            answer="Final answer",
            confidence=0.9,
            structured_trace=trace,
            tools_used=["tool_a", "tool_b"],
            total_iterations=3,
            iteration_limit_reached=False,
        )

        assert task_output.iteration_limit_reached is False
        assert len(task_output.tools_used) == 2
        assert task_output.tools_used == ["tool_a", "tool_b"]

    def test_extract_failed_pattern_max_iters(self):
        """Should identify failed sequences (hit max iterations)."""
        trace = [
            ReasoningStep(
                iteration=i,
                thought=f"Iteration {i}",
                action="call_tool",
                tool_name=f"tool_{i}",
                tool_args={},
                observation="...",
            )
            for i in range(1, 11)
        ]

        task_output = TaskOutput(
            task_id="test",
            reasoning_trace=["..." for _ in range(10)],
            answer="Partial answer",
            confidence=0.3,
            structured_trace=trace,
            tools_used=[f"tool_{i}" for i in range(1, 11)],
            total_iterations=10,
            iteration_limit_reached=True,  # Hit max iterations
        )

        assert task_output.iteration_limit_reached is True
        assert task_output.total_iterations == 10
        # This pattern should be marked as "Harmful" in playbook
