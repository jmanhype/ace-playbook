"""
Backward Compatibility Tests (T052)

Verifies ReActGenerator works as drop-in replacement for CoTGenerator
when no tools are provided.
"""

import pytest
import dspy
import os
from ace.generator.react_generator import ReActGenerator
from ace.generator.cot_generator import CoTGenerator
from ace.generator.signatures import TaskInput


def has_valid_api_key():
    """Check if a valid OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key and not api_key.startswith("your_") and api_key.startswith("sk-")


@pytest.fixture(scope="module", autouse=True)
def configure_dspy():
    """Configure DSPy with LM for testing."""
    if not has_valid_api_key():
        pytest.skip("Backward compatibility tests require a valid OPENAI_API_KEY")

    api_key = os.getenv("OPENAI_API_KEY")
    lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
    dspy.configure(lm=lm)
    yield
    # Cleanup after module
    dspy.configure(lm=None)


@pytest.mark.integration
@pytest.mark.backward_compat
class TestReActBackwardCompatibility:
    """Test ReActGenerator backward compatibility with CoTGenerator."""

    def test_react_without_tools_behaves_like_cot(self):
        """
        ReActGenerator without tools should work like CoTGenerator.

        Both should accept same TaskInput and return compatible TaskOutput.
        """
        # Initialize both generators without tools
        react_agent = ReActGenerator(model="gpt-4", max_iters=10)
        cot_agent = CoTGenerator(model="gpt-4")

        # Same task for both
        task = TaskInput(
            task_id="compat-001",
            description="What is 2 + 2?",
            domain="arithmetic",
            playbook_bullets=[]
        )

        # Both should execute without errors
        react_output = react_agent.forward(task)
        cot_output = cot_agent(task)  # CoTGenerator uses __call__

        # Verify both return TaskOutput with required fields
        assert hasattr(react_output, 'task_id')
        assert hasattr(react_output, 'answer')
        assert hasattr(react_output, 'confidence')
        assert hasattr(react_output, 'reasoning_trace')

        assert hasattr(cot_output, 'task_id')
        assert hasattr(cot_output, 'answer')
        assert hasattr(cot_output, 'confidence')
        assert hasattr(cot_output, 'reasoning_trace')

        # Task IDs should match input
        assert react_output.task_id == "compat-001"
        assert cot_output.task_id == "compat-001"

    def test_react_without_tools_no_tool_fields(self):
        """
        ReActGenerator without tools should have empty tool-related fields.
        """
        agent = ReActGenerator(model="gpt-4")

        task = TaskInput(
            task_id="notool-001",
            description="Simple reasoning task",
            domain="general",
            playbook_bullets=[]
        )

        output = agent.forward(task)

        # Tool-related fields should be empty/default
        assert output.tools_used == []
        assert output.total_iterations >= 0
        assert output.iteration_limit_reached is False

    def test_react_accepts_cot_style_playbook_bullets(self):
        """
        ReActGenerator should accept CoT-style playbook bullets.
        """
        agent = ReActGenerator(model="gpt-4")

        # CoT-style task with playbook bullets
        task = TaskInput(
            task_id="cot-style-001",
            description="Solve this problem",
            domain="general",
            playbook_bullets=[
                "Break down the problem into steps",
                "Use logical reasoning",
                "Check your work"
            ]
        )

        output = agent.forward(task)

        # Should execute without errors
        assert output.task_id == "cot-style-001"
        assert isinstance(output.reasoning_trace, list)

    def test_react_max_iters_compatible_with_cot(self):
        """
        ReActGenerator max_iters should work like CoTGenerator.
        """
        # Both use max_iters parameter
        react_agent = ReActGenerator(max_iters=5)

        # CoTGenerator doesn't have max_iters, but ReActGenerator should still work
        assert react_agent.max_iters == 5

        task = TaskInput(
            task_id="iters-001",
            description="Test task",
            domain="test",
            playbook_bullets=[]
        )

        output = react_agent.forward(task)

        # Should complete within iteration limit
        assert output.total_iterations <= 5

    def test_react_model_parameter_compatible(self):
        """
        ReActGenerator model parameter should work like CoTGenerator.
        """
        react_agent = ReActGenerator(model="gpt-4o-mini")
        cot_agent = CoTGenerator(model="gpt-4o-mini")

        # Both should store model identifier
        assert react_agent.model == "gpt-4o-mini"
        assert cot_agent.model == "gpt-4o-mini"

    def test_react_with_empty_tools_list(self):
        """
        ReActGenerator with empty tools list should work like CoTGenerator.
        """
        agent = ReActGenerator(tools=[], model="gpt-4")

        task = TaskInput(
            task_id="empty-tools-001",
            description="Task with no tools",
            domain="general",
            playbook_bullets=[]
        )

        output = agent.forward(task)

        # Should execute successfully
        assert output.task_id == "empty-tools-001"
        assert output.tools_used == []
        assert len(agent.tools) == 0

    def test_react_output_compatible_with_reflector(self):
        """
        ReActGenerator output should be compatible with existing Reflector.
        """
        from ace.reflector.grounded_reflector import GroundedReflector
        from ace.reflector.signatures import ReflectorInput

        agent = ReActGenerator(model="gpt-4")

        task = TaskInput(
            task_id="reflector-compat-001",
            description="Test reflector compatibility",
            domain="test",
            playbook_bullets=[]
        )

        output = agent.forward(task)

        # Should be able to create ReflectorInput from output
        reflector_input = ReflectorInput(
            task_id=output.task_id,
            reasoning_trace=output.reasoning_trace,
            answer=output.answer,
            confidence=output.confidence,
            bullets_referenced=getattr(output, 'bullets_referenced', []),
            ground_truth="",
            test_results="",
            error_messages=[],
            performance_metrics="",
            domain=task.domain,
            structured_trace=getattr(output, 'structured_trace', []),
            tools_used=getattr(output, 'tools_used', []),
            total_iterations=getattr(output, 'total_iterations', 0),
            iteration_limit_reached=getattr(output, 'iteration_limit_reached', False),
        )

        # Should create successfully
        assert reflector_input.task_id == "reflector-compat-001"

    def test_react_without_tools_returns_pydantic_model(self):
        """
        ReActGenerator should return Pydantic TaskOutput like CoTGenerator.
        """
        agent = ReActGenerator(model="gpt-4")

        task = TaskInput(
            task_id="pydantic-001",
            description="Test output type",
            domain="test",
            playbook_bullets=[]
        )

        output = agent.forward(task)

        # Should be a Pydantic model (from cot_generator)
        from ace.generator.cot_generator import TaskOutput as PydanticTaskOutput
        assert isinstance(output, PydanticTaskOutput)

        # Should have all standard fields
        assert hasattr(output, 'task_id')
        assert hasattr(output, 'reasoning_trace')
        assert hasattr(output, 'answer')
        assert hasattr(output, 'confidence')
        assert hasattr(output, 'bullets_referenced')
        assert hasattr(output, 'latency_ms')
        assert hasattr(output, 'model_name')


@pytest.mark.integration
@pytest.mark.backward_compat
class TestReActAddedFeatures:
    """Test that ReActGenerator adds features without breaking compatibility."""

    def test_react_adds_tool_support_optionally(self):
        """
        ReActGenerator adds tool support as optional feature.
        """
        # Works without tools (backward compat)
        agent_no_tools = ReActGenerator(model="gpt-4")
        assert len(agent_no_tools.tools) == 0

        # Works with tools (new feature)
        def sample_tool(query: str) -> str:
            """Sample tool."""
            return "result"

        agent_with_tools = ReActGenerator(tools=[sample_tool], model="gpt-4")
        assert len(agent_with_tools.tools) == 1
        assert "sample_tool" in agent_with_tools.tools

    def test_react_adds_structured_trace_optionally(self):
        """
        ReActGenerator adds structured_trace without breaking existing code.
        """
        agent = ReActGenerator(model="gpt-4")

        task = TaskInput(
            task_id="trace-001",
            description="Test",
            domain="test",
            playbook_bullets=[]
        )

        output = agent.forward(task)

        # New field is added but optional
        assert hasattr(output, 'structured_trace')
        # Without tools, structured trace should be empty
        assert output.structured_trace == []

    def test_react_adds_performance_metadata_optionally(self):
        """
        ReActGenerator adds performance metadata without breaking existing code.
        """
        agent = ReActGenerator(model="gpt-4")

        task = TaskInput(
            task_id="perf-001",
            description="Test",
            domain="test",
            playbook_bullets=[]
        )

        output = agent.forward(task)

        # New metadata field is added
        assert hasattr(output, 'metadata')
        assert isinstance(output.metadata, dict)
        assert 'performance' in output.metadata


@pytest.mark.integration
@pytest.mark.backward_compat
class TestMigrationPath:
    """Test migration path from CoTGenerator to ReActGenerator."""

    def test_direct_replacement_works(self):
        """
        CoTGenerator can be replaced with ReActGenerator directly.
        """
        # Old code pattern (CoT)
        old_agent = CoTGenerator(model="gpt-4")

        # New code pattern (ReAct, drop-in replacement)
        new_agent = ReActGenerator(model="gpt-4")

        # Same task
        task = TaskInput(
            task_id="migration-001",
            description="Test migration",
            domain="test",
            playbook_bullets=["Use step-by-step reasoning"]
        )

        # Both work
        old_output = old_agent(task)  # CoTGenerator uses __call__
        new_output = new_agent.forward(task)

        # Both have same interface
        assert old_output.task_id == new_output.task_id == "migration-001"
        assert isinstance(old_output.reasoning_trace, list)
        assert isinstance(new_output.reasoning_trace, list)

    def test_gradual_migration_with_tools(self):
        """
        Tools can be added gradually after migration.
        """
        # Step 1: Replace CoTGenerator with ReActGenerator (no tools)
        agent = ReActGenerator(model="gpt-4")
        assert len(agent.tools) == 0

        # Step 2: Add tools one by one
        def tool1(x: str) -> str:
            """Tool 1."""
            return x

        agent.register_tool(tool1)
        assert len(agent.tools) == 1

        def tool2(y: int) -> int:
            """Tool 2."""
            return y

        agent.register_tool(tool2)
        assert len(agent.tools) == 2

        # Agent works with incremental tool additions
        task = TaskInput(
            task_id="gradual-001",
            description="Test",
            domain="test",
            playbook_bullets=[]
        )
        output = agent.forward(task)
        assert output.task_id == "gradual-001"
