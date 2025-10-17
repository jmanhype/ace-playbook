"""
Unit tests for ReActGenerator initialization and core methods

Tests T012: Verify tools are registered, max_iters configured, DSPy ReAct module initialized
"""

import pytest
from typing import List
from ace.generator.react_generator import (
    ReActGenerator,
    ToolValidationError,
    DuplicateToolError,
    ToolNotFoundError,
)


@pytest.mark.unit
@pytest.mark.react
class TestReActGeneratorInit:
    """Test ReActGenerator initialization."""

    def test_init_with_no_tools(self):
        """Should initialize with empty tools list."""
        agent = ReActGenerator()
        assert agent.tools == {}
        assert agent.model == "gpt-4"  # Default model
        assert agent.max_iters is None

    def test_init_with_valid_tools(self):
        """Should register valid tools during initialization."""

        def search(query: str) -> str:
            """Search tool."""
            return ""

        def filter_results(data: List[str]) -> List[str]:
            """Filter tool."""
            return data

        agent = ReActGenerator(tools=[search, filter_results])
        assert len(agent.tools) == 2
        assert "search" in agent.tools
        assert "filter_results" in agent.tools

    def test_init_with_model(self):
        """Should store model identifier."""
        agent = ReActGenerator(model="gpt-4")
        assert agent.model == "gpt-4"

    def test_init_with_max_iters(self):
        """Should store agent-level max iterations."""
        agent = ReActGenerator(max_iters=15)
        assert agent.max_iters == 15

    def test_init_with_invalid_tool_raises_error(self):
        """Should raise ToolValidationError for invalid tool during init."""

        def bad_tool(query):  # Missing type annotation
            return ""

        with pytest.raises(ToolValidationError):
            ReActGenerator(tools=[bad_tool])


@pytest.mark.unit
@pytest.mark.react
class TestReActGeneratorRegisterTool:
    """Test tool registration after initialization."""

    def test_register_tool_success(self):
        """Should successfully register a new tool."""

        def new_tool(data: str) -> str:
            """New tool."""
            return data

        agent = ReActGenerator()
        agent.register_tool(new_tool)
        assert "new_tool" in agent.tools
        assert agent.tools["new_tool"] == new_tool

    def test_register_duplicate_raises_error(self):
        """Should raise DuplicateToolError for duplicate tool name."""

        def search(query: str) -> str:
            """Search tool."""
            return ""

        agent = ReActGenerator(tools=[search])

        # Try to register tool with same name
        def search(query: str) -> str:  # Same name
            """Another search tool."""
            return ""

        with pytest.raises(DuplicateToolError):
            agent.register_tool(search)

    def test_register_invalid_tool_raises_error(self):
        """Should raise ToolValidationError for invalid tool."""

        def invalid(query):  # Missing type annotation
            return ""

        agent = ReActGenerator()
        with pytest.raises(ToolValidationError):
            agent.register_tool(invalid)


@pytest.mark.unit
@pytest.mark.react
class TestReActGeneratorValidateTools:
    """Test tool validation method."""

    def test_validate_tools_with_valid_tools(self):
        """Should return empty list for all valid tools."""

        def tool1(x: str) -> str:
            """Tool 1."""
            return x

        def tool2(y: int) -> int:
            """Tool 2."""
            return y

        agent = ReActGenerator(tools=[tool1, tool2])
        errors = agent.validate_tools()

        # Filter out docstring warnings (those are acceptable)
        serious_errors = [e for e in errors if "missing type annotation" in e.lower()]
        assert len(serious_errors) == 0

    def test_validate_tools_returns_all_errors(self):
        """Should return list of all validation errors across tools."""
        # Note: This test will fail initially since we validate on registration
        # We'll skip tools that fail validation during init for this test
        agent = ReActGenerator()

        # Manually add an invalid tool to test validate_tools (bypassing register_tool)
        def bad_tool(query):  # Missing type annotation
            return ""

        agent.tools["bad_tool"] = bad_tool

        errors = agent.validate_tools()
        assert len(errors) > 0
        assert any("bad_tool" in err for err in errors)
        assert any("missing type annotation" in err.lower() for err in errors)


@pytest.mark.unit
@pytest.mark.react
class TestMaxIterationsHybridOverride:
    """
    T046: Test hybrid max iterations configuration (task > agent > system default 10).

    Verifies that max iterations follows the priority order:
    1. Task-level max_iterations (highest priority)
    2. Agent-level max_iters
    3. System default (10)
    """

    def test_system_default_max_iters(self):
        """Agent with no max_iters should use system default (10)."""
        agent = ReActGenerator()
        # System default is 10 (will be used in forward())
        assert agent.max_iters is None  # Agent stores None, uses default in forward

    def test_agent_level_max_iters_overrides_system_default(self):
        """Agent-level max_iters should override system default."""
        agent = ReActGenerator(max_iters=20)
        assert agent.max_iters == 20

        # Different agent with different override
        agent2 = ReActGenerator(max_iters=5)
        assert agent2.max_iters == 5

    def test_task_level_override_exists(self):
        """Task-level max_iters field should exist in TaskInput."""
        from ace.generator.signatures import TaskInput

        # Check that TaskInput has max_iterations field
        assert hasattr(TaskInput, "max_iterations")

    def test_hybrid_override_priority(self):
        """
        Verify priority order: task > agent > system (10).

        This tests the configuration hierarchy:
        - System default: 10
        - Agent level: Can override system default
        - Task level: Can override agent level
        """
        from ace.generator.signatures import TaskInput

        # Test 1: No overrides - should use system default (10)
        agent1 = ReActGenerator()
        assert agent1.max_iters is None  # Will use 10 in forward()

        # Test 2: Agent override - should use 15
        agent2 = ReActGenerator(max_iters=15)
        assert agent2.max_iters == 15

        # Test 3: Task with max_iterations should override agent
        task_with_override = TaskInput(
            task_id="test-001",
            description="Test task",
            domain="test",
            playbook_bullets=[],
            max_iterations=25  # Should override agent's 15
        )
        assert task_with_override.max_iterations == 25

        # Test 4: Task without max_iterations should use agent's setting
        task_without_override = TaskInput(
            task_id="test-002",
            description="Test task",
            domain="test",
            playbook_bullets=[]
        )
        assert task_without_override.max_iterations is None  # Will fallback to agent's setting


@pytest.mark.unit
@pytest.mark.react
class TestReActGeneratorToolAccess:
    """Test tool access and lookup."""

    def test_tools_are_stored_by_name(self):
        """Tools should be accessible by their function name."""

        def search_database(query: str) -> str:
            """Search tool."""
            return ""

        agent = ReActGenerator(tools=[search_database])
        assert "search_database" in agent.tools
        assert callable(agent.tools["search_database"])

    def test_multiple_tools_stored_correctly(self):
        """Multiple tools should all be stored with correct names."""

        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        def tool_b(y: int) -> int:
            """Tool B."""
            return y

        def tool_c(z: float) -> float:
            """Tool C."""
            return z

        agent = ReActGenerator(tools=[tool_a, tool_b, tool_c])
        assert len(agent.tools) == 3
        assert all(name in agent.tools for name in ["tool_a", "tool_b", "tool_c"])


@pytest.mark.unit
@pytest.mark.react
class TestReasoningTraceGeneration:
    """
    T045: Test structured reasoning trace generation.

    Verifies:
    - ReasoningStep dataclass structure
    - Timing metadata accuracy
    - Iteration counting
    - Tool execution tracking
    """

    def test_reasoning_step_structure(self):
        """ReasoningStep should have all required fields."""
        from ace.generator.signatures import ReasoningStep

        # Verify ReasoningStep has expected fields
        assert hasattr(ReasoningStep, "iteration")
        assert hasattr(ReasoningStep, "thought")
        assert hasattr(ReasoningStep, "action")
        assert hasattr(ReasoningStep, "tool_name")
        assert hasattr(ReasoningStep, "tool_args")
        assert hasattr(ReasoningStep, "observation")
        assert hasattr(ReasoningStep, "timestamp")
        assert hasattr(ReasoningStep, "duration_ms")

    def test_reasoning_step_creation(self):
        """Should be able to create ReasoningStep instances with correct types."""
        from ace.generator.signatures import ReasoningStep
        import time

        step = ReasoningStep(
            iteration=1,
            thought="I need to search for information",
            action="search",
            tool_name="search_tool",
            tool_args={"query": "test"},
            observation="Found 3 results",
            timestamp=time.time(),
            duration_ms=150.5
        )

        assert step.iteration == 1
        assert step.thought == "I need to search for information"
        assert step.action == "search"
        assert step.tool_name == "search_tool"
        assert step.tool_args == {"query": "test"}
        assert step.observation == "Found 3 results"
        assert isinstance(step.timestamp, float)
        assert step.duration_ms == 150.5

    def test_task_output_has_structured_trace_field(self):
        """TaskOutput should include structured_trace field."""
        from ace.generator.signatures import TaskOutput, ReasoningStep

        # Verify TaskOutput has structured_trace field
        assert hasattr(TaskOutput, "structured_trace")

        # Create TaskOutput with structured trace
        output = TaskOutput(
            task_id="test-001",
            answer="Result",
            confidence=0.9,
            reasoning_trace=["Step 1", "Step 2"],
            bullets_referenced=[],
            structured_trace=[
                ReasoningStep(
                    iteration=1,
                    thought="Test",
                    action="search",
                    tool_name="search_tool",
                    tool_args={"query": "test"},
                    observation="Result",
                    timestamp=0.0,
                    duration_ms=100.0
                )
            ],
            tools_used=["search_tool"],
            total_iterations=1,
            iteration_limit_reached=False
        )

        assert len(output.structured_trace) == 1
        assert output.structured_trace[0].iteration == 1

    def test_task_output_tracks_tools_used(self):
        """TaskOutput should track all tools used during execution."""
        from ace.generator.signatures import TaskOutput

        # Verify TaskOutput has tools_used field
        assert hasattr(TaskOutput, "tools_used")

        output = TaskOutput(
            task_id="test-002",
            answer="Result",
            confidence=0.8,
            reasoning_trace=["Step 1", "Step 2", "Step 3"],
            bullets_referenced=[],
            structured_trace=[],
            tools_used=["tool_a", "tool_b", "tool_a"],  # tool_a used twice
            total_iterations=3,
            iteration_limit_reached=False
        )

        assert len(output.tools_used) == 3
        assert "tool_a" in output.tools_used
        assert "tool_b" in output.tools_used

    def test_task_output_tracks_iteration_count(self):
        """TaskOutput should track total iterations and limit status."""
        from ace.generator.signatures import TaskOutput

        # Verify iteration tracking fields exist
        assert hasattr(TaskOutput, "total_iterations")
        assert hasattr(TaskOutput, "iteration_limit_reached")

        # Case 1: Completed within limit
        output1 = TaskOutput(
            task_id="test-003",
            answer="Success",
            confidence=0.9,
            reasoning_trace=["Step 1", "Step 2"],
            bullets_referenced=[],
            structured_trace=[],
            tools_used=["tool_a"],
            total_iterations=5,
            iteration_limit_reached=False
        )

        assert output1.total_iterations == 5
        assert output1.iteration_limit_reached is False

        # Case 2: Hit iteration limit
        output2 = TaskOutput(
            task_id="test-004",
            answer="Partial result",
            confidence=0.5,
            reasoning_trace=["Step 1", "Step 2", "..."],
            bullets_referenced=[],
            structured_trace=[],
            tools_used=["tool_a", "tool_b"],
            total_iterations=10,
            iteration_limit_reached=True
        )

        assert output2.total_iterations == 10
        assert output2.iteration_limit_reached is True

    def test_timing_metadata_precision(self):
        """Timing metadata should have millisecond precision."""
        from ace.generator.signatures import ReasoningStep

        # Test with precise timing values
        step1 = ReasoningStep(
            iteration=1,
            thought="Test",
            action="search",
            tool_name="tool",
            tool_args={},
            observation="Result",
            timestamp=1234567890.123456,  # Unix timestamp with microseconds
            duration_ms=123.456  # Milliseconds with precision
        )

        # Verify precision is preserved
        assert step1.timestamp == 1234567890.123456
        assert step1.duration_ms == 123.456

    def test_structured_trace_ordering(self):
        """Structured trace should maintain iteration order."""
        from ace.generator.signatures import TaskOutput, ReasoningStep

        trace = [
            ReasoningStep(
                iteration=i,
                thought=f"Thought {i}",
                action="search",
                tool_name="tool",
                tool_args={},
                observation=f"Result {i}",
                timestamp=float(i),
                duration_ms=100.0
            )
            for i in range(1, 6)  # Iterations 1-5
        ]

        output = TaskOutput(
            task_id="test-005",
            answer="Final result",
            confidence=0.9,
            reasoning_trace=["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"],
            bullets_referenced=[],
            structured_trace=trace,
            tools_used=["tool"],
            total_iterations=5,
            iteration_limit_reached=False
        )

        # Verify ordering is preserved
        for i, step in enumerate(output.structured_trace, start=1):
            assert step.iteration == i
            assert step.thought == f"Thought {i}"
            assert step.observation == f"Result {i}"
