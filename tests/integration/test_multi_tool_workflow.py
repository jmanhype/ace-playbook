"""
Integration Tests for Multi-Tool Workflow Agent (User Story 2)

Tests:
- T026: Multi-tool orchestration with heterogeneous tools
- T027: Tool failure handling and adaptation

Note: Tests focus on tool registration, validation, execution wrapper, and
reflector integration with multi-tool patterns.
"""

import pytest
from typing import Dict, Any, List

from ace.generator.react_generator import ReActGenerator, ToolValidationError, DuplicateToolError
from ace.generator.signatures import TaskInput, ReasoningStep
from ace.reflector.grounded_reflector import InsightCandidate, ReflectorOutput
from ace.reflector.signatures import ReflectorInput, InsightSection
from ace.curator.semantic_curator import SemanticCurator
from ace.curator.curator_models import CuratorInput
from ace.models.playbook import PlaybookBullet, PlaybookStage


# ============================================================================
# HETEROGENEOUS MOCK TOOLS
# ============================================================================

def mock_api_fetch(endpoint: str) -> Dict[str, Any]:
    """Mock API tool - fetches data from endpoint."""
    return {
        "endpoint": endpoint,
        "data": {"value": 42, "status": "success"},
        "source": "api"
    }


def mock_calculator(expression: str) -> float:
    """Mock calculator tool - evaluates expression."""
    return 123.45


def mock_formatter(data: Any, format_type: str = "json") -> str:
    """Mock formatter tool - formats data."""
    return f"[{format_type.upper()}] {str(data)}"


def failing_api_fetch(endpoint: str) -> Dict[str, Any]:
    """API tool that always fails."""
    raise TimeoutError(f"API timeout for: {endpoint}")


def invalid_tool_no_params():
    """Tool with no parameters - should fail validation."""
    return "invalid"


def invalid_tool_no_types(param1, param2):
    """Tool with no type annotations - should fail validation."""
    return "invalid"


# ============================================================================
# MOCK REFLECTOR FOR MULTI-TOOL TESTING
# ============================================================================

class MultiToolMockReflector:
    """Mock reflector that analyzes multi-tool usage patterns."""

    def __call__(self, reflector_input: ReflectorInput) -> ReflectorOutput:
        """Generate insights based on tool usage."""
        insights = []

        # Success pattern with multiple tools
        if reflector_input.tools_used and len(reflector_input.tools_used) >= 2:
            tool_chain = " → ".join(reflector_input.tools_used)
            insights.append(
                InsightCandidate(
                    content=f"Multi-tool workflow: {tool_chain}",
                    section=InsightSection.HELPFUL,
                    confidence=0.9,
                    rationale="Successful multi-tool orchestration",
                    tags=["multi-tool", "orchestration"],
                    referenced_steps=[0, 1],
                    tool_sequence=reflector_input.tools_used,
                    tool_success_rate=1.0,
                    avg_iterations=reflector_input.total_iterations,
                    avg_execution_time_ms=150.0,
                )
            )

        # Failure pattern with adaptation
        if reflector_input.error_messages and len(reflector_input.tools_used) > 1:
            insights.append(
                InsightCandidate(
                    content=f"After '{reflector_input.tools_used[0]}' failed, adapted to '{reflector_input.tools_used[1]}'",
                    section=InsightSection.HELPFUL,
                    confidence=0.85,
                    rationale="Tool adaptation after failure",
                    tags=["adaptation", "error-recovery"],
                    referenced_steps=[0, 1],
                    tool_sequence=reflector_input.tools_used[:2],
                    tool_success_rate=1.0,
                    avg_iterations=reflector_input.total_iterations,
                    avg_execution_time_ms=200.0,
                )
            )

        return ReflectorOutput(
            task_id=reflector_input.task_id,
            insights=insights,
            analysis_summary="Multi-tool workflow analysis",
            referenced_steps=list(range(len(reflector_input.reasoning_trace))),
            confidence_score=0.85,
            feedback_types_used=[],
            requires_human_review=False,
            contradicts_existing=[],
        )


# ============================================================================
# T026: MULTI-TOOL ORCHESTRATION TEST
# ============================================================================

def test_multi_tool_orchestration_tool_registration():
    """
    T026: Verify agent can register and validate heterogeneous tools.

    Tests that ReActGenerator properly registers different tool types
    (API, calculator, formatter) and validates their signatures.
    """
    # Initialize agent with heterogeneous tools
    agent = ReActGenerator(
        tools=[mock_api_fetch, mock_calculator, mock_formatter],
        model="gpt-4",
        max_iters=10
    )

    # Verify tools are registered
    assert len(agent.tools) == 3
    assert "mock_api_fetch" in agent.tools
    assert "mock_calculator" in agent.tools
    assert "mock_formatter" in agent.tools

    # Verify each tool is callable
    assert callable(agent.tools["mock_api_fetch"])
    assert callable(agent.tools["mock_calculator"])
    assert callable(agent.tools["mock_formatter"])

    # Verify tools can be executed
    result, error = agent._execute_tool_with_timeout("mock_api_fetch", {"endpoint": "/test"})
    assert error is None
    assert result["endpoint"] == "/test"
    assert result["source"] == "api"

    result, error = agent._execute_tool_with_timeout("mock_calculator", {"expression": "10+5"})
    assert error is None
    assert isinstance(result, float)

    result, error = agent._execute_tool_with_timeout("mock_formatter", {"data": "test", "format_type": "json"})
    assert error is None
    assert "[JSON]" in result


def test_multi_tool_validation_rejects_invalid_tools():
    """
    T026: Verify tool validation rejects invalid tool signatures.
    """
    # Tool with no parameters should fail
    with pytest.raises(ToolValidationError) as exc_info:
        agent = ReActGenerator(tools=[invalid_tool_no_params])
    assert "must have at least one parameter" in str(exc_info.value)

    # Tool with no type annotations should fail
    with pytest.raises(ToolValidationError) as exc_info:
        agent = ReActGenerator(tools=[invalid_tool_no_types])
    assert "missing type annotation" in str(exc_info.value)


def test_multi_tool_prevents_duplicate_registration():
    """
    T026: Verify duplicate tool names are rejected.
    """
    agent = ReActGenerator(tools=[mock_api_fetch])

    # Attempting to register duplicate should fail
    with pytest.raises(DuplicateToolError) as exc_info:
        agent.register_tool(mock_api_fetch)
    assert "already registered" in str(exc_info.value)


# ============================================================================
# T027: TOOL FAILURE HANDLING TEST
# ============================================================================

def test_tool_failure_handling_with_timeout():
    """
    T027: Verify tool execution wrapper handles failures gracefully.

    Tests that the execution wrapper catches exceptions and returns
    formatted error messages.
    """
    agent = ReActGenerator(tools=[failing_api_fetch], model="gpt-4")

    # Execute failing tool
    result, error = agent._execute_tool_with_timeout("failing_api_fetch", {"endpoint": "/test"})

    # Verify failure is captured
    assert result is None
    assert error is not None
    assert "API timeout" in error or "Tool execution failed" in error


def test_tool_failure_with_invalid_arguments():
    """
    T027: Verify tool execution handles invalid arguments.
    """
    agent = ReActGenerator(tools=[mock_calculator], model="gpt-4")

    # Call with wrong argument name
    result, error = agent._execute_tool_with_timeout("mock_calculator", {"wrong_arg": "test"})

    # Verify error is captured
    assert result is None
    assert error is not None
    assert "Invalid arguments" in error or "unexpected keyword argument" in error.lower()


def test_tool_failure_with_unknown_tool():
    """
    T027: Verify execution wrapper handles unknown tool names.
    """
    agent = ReActGenerator(tools=[mock_api_fetch], model="gpt-4")

    # Try to execute unregistered tool
    result, error = agent._execute_tool_with_timeout("unknown_tool", {})

    # Verify error is captured
    assert result is None
    assert error is not None
    assert "not found" in error


# ============================================================================
# REFLECTOR INTEGRATION WITH MULTI-TOOL PATTERNS
# ============================================================================

def test_reflector_captures_multi_tool_pattern():
    """
    T026/T027: Verify reflector captures multi-tool orchestration patterns.

    Tests that the reflector properly analyzes tool sequences and
    generates appropriate insights.
    """
    reflector = MultiToolMockReflector()

    # Create reflector input with multi-tool sequence
    reflector_input = ReflectorInput(
        task_id="multi-001",
        reasoning_trace=["Step 1", "Step 2", "Step 3"],
        answer="Result",
        confidence=0.9,
        bullets_referenced=[],
        ground_truth="",
        test_results="",
        error_messages=[],
        performance_metrics="",
        domain="multi-tool-test",
        structured_trace=[],
        tools_used=["mock_api_fetch", "mock_calculator", "mock_formatter"],
        total_iterations=3,
        iteration_limit_reached=False,
    )

    # Execute reflection
    reflector_output = reflector(reflector_input)

    # Verify multi-tool pattern is captured
    assert len(reflector_output.insights) > 0

    helpful_insights = [i for i in reflector_output.insights if i.section == InsightSection.HELPFUL]
    assert len(helpful_insights) > 0

    # Verify tool sequence is documented
    multi_tool_insight = helpful_insights[0]
    assert multi_tool_insight.tool_sequence == ["mock_api_fetch", "mock_calculator", "mock_formatter"]
    assert "multi-tool" in multi_tool_insight.tags or "orchestration" in multi_tool_insight.tags
    assert multi_tool_insight.tool_success_rate == 1.0


def test_reflector_captures_tool_adaptation():
    """
    T027: Verify reflector captures tool adaptation after failure.

    Tests that the reflector identifies when an agent switches tools
    after encountering an error.
    """
    reflector = MultiToolMockReflector()

    # Create reflector input with failure and adaptation
    reflector_input = ReflectorInput(
        task_id="adapt-001",
        reasoning_trace=["Step 1", "Step 2", "Step 3"],
        answer="Result after adaptation",
        confidence=0.8,
        bullets_referenced=[],
        ground_truth="",
        test_results="",
        error_messages=["API timeout for: /data"],
        performance_metrics="",
        domain="error-recovery",
        structured_trace=[],
        tools_used=["failing_api_fetch", "mock_api_fetch"],  # Adapted to backup
        total_iterations=3,
        iteration_limit_reached=False,
    )

    # Execute reflection
    reflector_output = reflector(reflector_input)

    # Verify adaptation pattern is captured
    assert len(reflector_output.insights) > 0

    adaptation_insights = [
        i for i in reflector_output.insights
        if "adaptation" in i.tags or "error-recovery" in i.tags
    ]
    assert len(adaptation_insights) > 0

    # Verify adaptation strategy is documented
    adaptation = adaptation_insights[0]
    assert adaptation.section == InsightSection.HELPFUL
    assert "failing_api_fetch" in adaptation.content
    assert "mock_api_fetch" in adaptation.content
    assert adaptation.tool_sequence == ["failing_api_fetch", "mock_api_fetch"]


# ============================================================================
# CURATOR INTEGRATION WITH MULTI-TOOL STRATEGIES
# ============================================================================

def test_curator_stores_multi_tool_strategies():
    """
    T026/T027: Verify curator properly stores multi-tool strategies.

    Tests that multi-tool patterns are correctly added to the playbook
    with all metadata preserved.
    """
    curator = SemanticCurator()
    reflector = MultiToolMockReflector()

    # Create multi-tool execution data
    reflector_input = ReflectorInput(
        task_id="curator-001",
        reasoning_trace=["Step 1", "Step 2"],
        answer="Success",
        confidence=0.9,
        bullets_referenced=[],
        ground_truth="",
        test_results="",
        error_messages=[],
        performance_metrics="",
        domain="multi-tool-curator-test",
        structured_trace=[],
        tools_used=["mock_api_fetch", "mock_calculator"],
        total_iterations=2,
        iteration_limit_reached=False,
    )

    # Get insights
    reflector_output = reflector(reflector_input)

    # Convert to curator format
    insights_dicts = [
        {
            "content": ins.content,
            "section": ins.section.value,
            "tags": ins.tags,
            "tool_sequence": ins.tool_sequence,
            "tool_success_rate": ins.tool_success_rate,
            "avg_iterations": ins.avg_iterations,
            "avg_execution_time_ms": ins.avg_execution_time_ms,
        }
        for ins in reflector_output.insights
    ]

    # Apply to curator
    curator_input = CuratorInput(
        task_id="curator-001",
        domain_id="multi-tool-curator-test",
        insights=insights_dicts,
        current_playbook=[],
        target_stage=PlaybookStage.SHADOW,
    )

    curator_output = curator.apply_delta(curator_input)

    # Verify multi-tool strategy is stored
    assert len(curator_output.updated_playbook) > 0

    multi_tool_bullets = [
        b for b in curator_output.updated_playbook
        if b.tool_sequence and len(b.tool_sequence) >= 2
    ]
    assert len(multi_tool_bullets) > 0

    # Verify metadata is preserved
    strategy = multi_tool_bullets[0]
    assert strategy.tool_sequence == ["mock_api_fetch", "mock_calculator"]
    assert strategy.tool_success_rate == 1.0
    assert strategy.avg_iterations == 2
    assert strategy.avg_execution_time_ms == 150.0
    assert strategy.domain_id == "multi-tool-curator-test"
