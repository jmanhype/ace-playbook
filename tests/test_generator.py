"""
Unit Tests for Generator Module

Tests for CoTGenerator, trace formatting, bullet reference extraction,
and playbook context injection.

Coverage:
- T042: Unit tests for trace formatting
- Reasoning trace parsing (1-20 step validation)
- Bullet reference detection
- Playbook context formatting
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from ace.generator import (
    CoTGenerator,
    TaskInput,
    TaskOutput,
    create_cot_generator
)


class TestCoTGenerator:
    """Test suite for CoTGenerator class."""

    def test_initialization(self):
        """Test CoTGenerator can be initialized with default params."""
        generator = CoTGenerator()

        assert generator.model == "gpt-4-turbo"
        assert generator.temperature == 0.7
        assert generator.max_tokens == 2000

    def test_initialization_with_custom_params(self):
        """Test CoTGenerator initialization with custom parameters."""
        generator = CoTGenerator(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1000
        )

        assert generator.model == "gpt-3.5-turbo"
        assert generator.temperature == 0.5
        assert generator.max_tokens == 1000

    def test_factory_function(self):
        """Test create_cot_generator factory function."""
        generator = create_cot_generator(model="gpt-4", temperature=0.8)

        assert isinstance(generator, CoTGenerator)
        assert generator.model == "gpt-4"
        assert generator.temperature == 0.8


class TestPlaybookContextFormatting:
    """Test suite for playbook context injection (T041)."""

    def test_format_empty_playbook(self):
        """Test formatting with no playbook bullets."""
        generator = CoTGenerator()
        context = generator.format_playbook_context([])

        assert context == "No playbook strategies available."

    def test_format_single_bullet(self):
        """Test formatting with single playbook bullet."""
        generator = CoTGenerator()
        bullets = ["Break problems into smaller steps"]
        bullet_ids = ["bullet-abc123"]

        context = generator.format_playbook_context(bullets, bullet_ids)

        assert "Strategy 1 [bullet-abc123]" in context
        assert "Break problems into smaller steps" in context

    def test_format_multiple_bullets(self):
        """Test formatting with multiple playbook bullets."""
        generator = CoTGenerator()
        bullets = [
            "Break problems into smaller steps",
            "Verify intermediate results",
            "Use explicit variable names"
        ]
        bullet_ids = ["bullet-001", "bullet-002", "bullet-003"]

        context = generator.format_playbook_context(bullets, bullet_ids)

        assert "Strategy 1 [bullet-001]: Break problems into smaller steps" in context
        assert "Strategy 2 [bullet-002]: Verify intermediate results" in context
        assert "Strategy 3 [bullet-003]: Use explicit variable names" in context

    def test_format_without_bullet_ids(self):
        """Test formatting generates placeholder IDs when none provided."""
        generator = CoTGenerator()
        bullets = ["Strategy A", "Strategy B"]

        context = generator.format_playbook_context(bullets)

        assert "[bullet-000]" in context
        assert "[bullet-001]" in context
        assert "Strategy A" in context
        assert "Strategy B" in context


class TestReasoningTraceFormatting:
    """Test suite for reasoning trace parsing (T040)."""

    def test_parse_multiline_reasoning(self):
        """Test parsing multi-line reasoning into list of steps."""
        generator = CoTGenerator()
        reasoning_text = """Step 1: Identify the problem
Step 2: Break into parts
Step 3: Solve each part
Step 4: Combine results"""

        trace = generator.parse_reasoning_trace(reasoning_text)

        assert len(trace) == 4
        assert trace[0] == "Step 1: Identify the problem"
        assert trace[1] == "Step 2: Break into parts"
        assert trace[2] == "Step 3: Solve each part"
        assert trace[3] == "Step 4: Combine results"

    def test_parse_reasoning_removes_empty_lines(self):
        """Test that empty lines are filtered out."""
        generator = CoTGenerator()
        reasoning_text = """Step 1: First step

Step 2: Second step


Step 3: Third step"""

        trace = generator.parse_reasoning_trace(reasoning_text)

        assert len(trace) == 3
        assert "" not in trace

    def test_parse_reasoning_truncates_at_max_steps(self):
        """Test that reasoning is truncated at max_steps limit."""
        generator = CoTGenerator()

        # Generate 25 steps
        steps = [f"Step {i}: Action {i}" for i in range(1, 26)]
        reasoning_text = "\n".join(steps)

        trace = generator.parse_reasoning_trace(reasoning_text, max_steps=20)

        assert len(trace) == 20  # Truncated to max
        assert trace[0] == "Step 1: Action 1"
        assert trace[19] == "Step 20: Action 20"

    def test_parse_empty_reasoning(self):
        """Test parsing empty reasoning returns fallback message."""
        generator = CoTGenerator()
        reasoning_text = ""

        trace = generator.parse_reasoning_trace(reasoning_text)

        assert len(trace) == 1
        assert trace[0] == "No explicit reasoning steps provided"

    def test_parse_reasoning_single_step(self):
        """Test parsing single-step reasoning."""
        generator = CoTGenerator()
        reasoning_text = "The answer is 42 because that's the answer to everything"

        trace = generator.parse_reasoning_trace(reasoning_text)

        assert len(trace) == 1
        assert trace[0] == "The answer is 42 because that's the answer to everything"


class TestBulletReferenceExtraction:
    """Test suite for bullet reference detection (T040)."""

    def test_extract_single_bullet_reference(self):
        """Test extracting single bullet ID from reasoning."""
        generator = CoTGenerator()
        reasoning = "Using strategy [bullet-abc123], I will break this problem down."

        refs = generator.extract_bullet_references(reasoning)

        assert len(refs) == 1
        assert refs[0] == "bullet-abc123"

    def test_extract_multiple_bullet_references(self):
        """Test extracting multiple bullet IDs from reasoning."""
        generator = CoTGenerator()
        reasoning = """
        Step 1: Apply [bullet-001] to break the problem.
        Step 2: Use [bullet-002] to verify results.
        Step 3: Combine with [bullet-003] for final answer.
        """

        refs = generator.extract_bullet_references(reasoning)

        assert len(refs) == 3
        assert "bullet-001" in refs
        assert "bullet-002" in refs
        assert "bullet-003" in refs

    def test_extract_duplicate_bullet_references(self):
        """Test that duplicate references are deduplicated."""
        generator = CoTGenerator()
        reasoning = """
        Using [bullet-abc] to start.
        Applying [bullet-abc] again for verification.
        Final check with [bullet-abc].
        """

        refs = generator.extract_bullet_references(reasoning)

        assert len(refs) == 1  # Deduplicated
        assert refs[0] == "bullet-abc"

    def test_extract_no_bullet_references(self):
        """Test reasoning with no bullet references returns empty list."""
        generator = CoTGenerator()
        reasoning = "I will solve this step by step without referencing strategies."

        refs = generator.extract_bullet_references(reasoning)

        assert len(refs) == 0

    def test_extract_bullet_references_with_hyphens(self):
        """Test extracting bullet IDs with multiple hyphens."""
        generator = CoTGenerator()
        reasoning = "Using [bullet-abc-123-def] and [bullet-xyz-789]."

        refs = generator.extract_bullet_references(reasoning)

        assert len(refs) == 2
        assert "bullet-abc-123-def" in refs
        assert "bullet-xyz-789" in refs


class TestTaskInputValidation:
    """Test suite for TaskInput validation."""

    def test_validate_valid_task_input(self):
        """Test validation passes for valid TaskInput."""
        generator = CoTGenerator()
        task = TaskInput(
            task_id="task-001",
            description="Solve this problem",
            domain="arithmetic",
            playbook_bullets=["Strategy 1"],
            max_reasoning_steps=10
        )

        # Should not raise
        generator.validate_task_input(task)

    def test_validate_missing_task_id(self):
        """Test validation fails when task_id is missing."""
        generator = CoTGenerator()
        task = TaskInput(
            task_id="",
            description="Solve this problem"
        )

        with pytest.raises(ValueError, match="task_id is required"):
            generator.validate_task_input(task)

    def test_validate_missing_description(self):
        """Test validation fails when description is missing."""
        generator = CoTGenerator()
        task = TaskInput(
            task_id="task-001",
            description=""
        )

        with pytest.raises(ValueError, match="task description is required"):
            generator.validate_task_input(task)

    def test_validate_invalid_max_reasoning_steps_too_low(self):
        """Test validation fails when max_reasoning_steps < 1."""
        generator = CoTGenerator()
        task = TaskInput(
            task_id="task-001",
            description="Problem",
            max_reasoning_steps=0
        )

        with pytest.raises(ValueError, match="max_reasoning_steps must be between 1 and 20"):
            generator.validate_task_input(task)

    def test_validate_invalid_max_reasoning_steps_too_high(self):
        """Test validation fails when max_reasoning_steps > 20."""
        generator = CoTGenerator()
        task = TaskInput(
            task_id="task-001",
            description="Problem",
            max_reasoning_steps=25
        )

        with pytest.raises(ValueError, match="max_reasoning_steps must be between 1 and 20"):
            generator.validate_task_input(task)


class TestCoTGeneratorExecution:
    """Test suite for end-to-end CoTGenerator execution (mocked)."""

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    def test_execute_task_with_mock_predictor(self, mock_cot_class):
        """Test full task execution with mocked DSPy predictor."""
        # Mock the predictor response
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.reasoning = "Step 1: Analyze\nStep 2: Calculate\nStep 3: Answer"
        mock_prediction.answer = "42"
        mock_prediction.confidence = 0.95

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor

        # Create generator and execute task
        generator = CoTGenerator(model="gpt-4-turbo")
        generator.predictor = mock_predictor

        task = TaskInput(
            task_id="task-001",
            description="What is 6 * 7?",
            domain="arithmetic",
            playbook_bullets=["Break into steps"],
            max_reasoning_steps=10
        )

        output = generator(task)

        # Assertions
        assert output.task_id == "task-001"
        assert len(output.reasoning_trace) == 3
        assert output.answer == "42"
        assert output.confidence == 0.95
        assert output.latency_ms is not None  # May be 0 for very fast mocked execution
        assert output.latency_ms >= 0
        assert output.model_name == "gpt-4-turbo"

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    def test_execute_task_extracts_bullet_references(self, mock_cot_class):
        """Test that bullet references are extracted from reasoning."""
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.reasoning = """
        Step 1: Using [bullet-abc123] to break down problem
        Step 2: Apply [bullet-def456] for verification
        """
        mock_prediction.answer = "Result"
        mock_prediction.confidence = 0.9

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor

        generator = CoTGenerator()
        generator.predictor = mock_predictor

        task = TaskInput(
            task_id="task-002",
            description="Test task",
            playbook_bullets=["Strategy A", "Strategy B"]
        )

        output = generator(task)

        assert len(output.bullets_referenced) == 2
        assert "bullet-abc123" in output.bullets_referenced
        assert "bullet-def456" in output.bullets_referenced

    def test_execute_task_with_invalid_input_raises(self):
        """Test that invalid TaskInput raises ValueError."""
        generator = CoTGenerator()

        invalid_task = TaskInput(
            task_id="",  # Empty task_id
            description="Problem"
        )

        with pytest.raises(ValueError):
            generator(invalid_task)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
