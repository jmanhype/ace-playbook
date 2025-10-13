"""
Integration Tests for Generator Module

End-to-end tests for CoTGenerator with playbook integration.
Tests the full workflow: playbook retrieval → context injection → generation → trace validation.

Coverage:
- T043: Integration test for reasoning traces
- End-to-end generator execution with real playbook
- Bullet reference validation
- Reasoning trace structure validation
"""

import pytest
from unittest.mock import Mock, patch
import uuid
from datetime import datetime

from ace.generator import CoTGenerator, TaskInput, TaskOutput
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.utils.database import get_session


@pytest.fixture
def generator():
    """Create CoTGenerator instance for testing."""
    return CoTGenerator(model="gpt-4-turbo", temperature=0.7)


@pytest.fixture
def sample_playbook_bullets(test_db_session):
    """Create sample playbook bullets for integration testing."""
    repo = PlaybookRepository(test_db_session)

    bullets = [
        PlaybookBullet(
            id=str(uuid.uuid4()),
            domain_id="test-arithmetic",
            content="Break arithmetic problems into step-by-step calculations",
            section="Helpful",
            helpful_count=5,
            harmful_count=0,
            tags=["arithmetic", "decomposition"],
            embedding=[0.1] * 384,  # Placeholder embedding
            created_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
            stage=PlaybookStage.PROD
        ),
        PlaybookBullet(
            id=str(uuid.uuid4()),
            domain_id="test-arithmetic",
            content="Verify intermediate results before proceeding",
            section="Helpful",
            helpful_count=3,
            harmful_count=0,
            tags=["verification", "accuracy"],
            embedding=[0.2] * 384,
            created_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
            stage=PlaybookStage.PROD
        ),
        PlaybookBullet(
            id=str(uuid.uuid4()),
            domain_id="test-arithmetic",
            content="Use explicit variable names for clarity",
            section="Helpful",
            helpful_count=2,
            harmful_count=0,
            tags=["clarity", "naming"],
            embedding=[0.3] * 384,
            created_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
            stage=PlaybookStage.PROD
        )
    ]

    for bullet in bullets:
        test_db_session.add(bullet)
    test_db_session.commit()

    return bullets


class TestCoTGeneratorIntegration:
    """Integration tests for CoTGenerator with playbook context."""

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    def test_generator_with_playbook_bullets(
        self,
        mock_cot_class,
        generator,
        sample_playbook_bullets
    ):
        """
        Test full generator workflow with playbook bullets.

        T043: Verify reasoning traces show strategy consultation and decision-making.
        """
        # Mock the DSPy predictor
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.reasoning = """
        Step 1: Using strategy [bullet-001] to break down 37 * 42
        Step 2: Calculate 37 * 40 = 1480
        Step 3: Calculate 37 * 2 = 74
        Step 4: Apply verification [bullet-002] - sum results
        Step 5: 1480 + 74 = 1554
        """
        mock_prediction.answer = "1554"
        mock_prediction.confidence = 0.95

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor
        generator.predictor = mock_predictor

        # Create task with playbook bullets
        bullet_contents = [b.content for b in sample_playbook_bullets]
        bullet_ids = [b.id for b in sample_playbook_bullets]

        task = TaskInput(
            task_id=str(uuid.uuid4()),
            description="Calculate 37 * 42",
            domain="arithmetic",
            playbook_bullets=bullet_contents,
            max_reasoning_steps=10
        )

        # Execute generator
        output = generator(task)

        # Assertions - T043 acceptance criteria
        assert isinstance(output, TaskOutput)
        assert output.task_id == task.task_id

        # Verify reasoning trace structure
        assert len(output.reasoning_trace) >= 3, "Should have at least 3 reasoning steps"
        assert len(output.reasoning_trace) <= 20, "Should not exceed 20 steps"

        # Verify trace shows strategy consultation
        reasoning_text = "\n".join(output.reasoning_trace)
        assert "strategy" in reasoning_text.lower() or "bullet" in reasoning_text.lower(), \
            "Reasoning should reference playbook strategies"

        # Verify bullet references are extracted
        assert len(output.bullets_referenced) >= 1, \
            "Should reference at least 1 playbook bullet"

        # Verify answer and confidence
        assert output.answer == "1554"
        assert 0.0 <= output.confidence <= 1.0

        # Verify performance metrics
        assert output.latency_ms > 0
        assert output.model_name == "gpt-4-turbo"

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    def test_generator_without_playbook_bullets(self, mock_cot_class, generator):
        """
        Test generator with empty playbook (no strategies available).

        Should still produce valid output but with no bullet references.
        """
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.reasoning = "Step 1: Solve directly\nStep 2: Answer is 42"
        mock_prediction.answer = "42"
        mock_prediction.confidence = 0.8

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor
        generator.predictor = mock_predictor

        task = TaskInput(
            task_id=str(uuid.uuid4()),
            description="What is 6 * 7?",
            domain="arithmetic",
            playbook_bullets=[],  # Empty playbook
            max_reasoning_steps=10
        )

        output = generator(task)

        assert isinstance(output, TaskOutput)
        assert len(output.reasoning_trace) >= 1
        assert output.answer == "42"
        assert len(output.bullets_referenced) == 0  # No bullets to reference

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    def test_generator_explicit_bullet_references(self, mock_cot_class, generator):
        """
        Test that bullet IDs are correctly extracted from reasoning trace.

        T043: Verify bullets_referenced list populated with IDs.
        """
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.reasoning = """
        I will apply [bullet-abc123] for decomposition.
        Then use [bullet-def456] for verification.
        Finally apply [bullet-ghi789] for clarity.
        """
        mock_prediction.answer = "Complete"
        mock_prediction.confidence = 0.9

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor
        generator.predictor = mock_predictor

        task = TaskInput(
            task_id=str(uuid.uuid4()),
            description="Test task",
            domain="test",
            playbook_bullets=["Strategy 1", "Strategy 2", "Strategy 3"]
        )

        output = generator(task)

        # Verify all three bullet IDs are extracted
        assert len(output.bullets_referenced) == 3
        assert "bullet-abc123" in output.bullets_referenced
        assert "bullet-def456" in output.bullets_referenced
        assert "bullet-ghi789" in output.bullets_referenced

    def test_generator_playbook_context_formatting(self, generator, sample_playbook_bullets):
        """
        Test that playbook context is correctly formatted for injection.

        T041: Verify context injection produces numbered list with IDs.
        """
        bullet_contents = [b.content for b in sample_playbook_bullets]
        bullet_ids = [b.id for b in sample_playbook_bullets]

        context = generator.format_playbook_context(bullet_contents, bullet_ids)

        # Verify numbered format
        assert "Strategy 1" in context
        assert "Strategy 2" in context
        assert "Strategy 3" in context

        # Verify IDs are included
        for bullet_id in bullet_ids:
            assert f"[{bullet_id}]" in context

        # Verify content is included
        for content in bullet_contents:
            assert content in context

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    def test_generator_respects_max_reasoning_steps(self, mock_cot_class, generator):
        """
        Test that reasoning trace is truncated at max_reasoning_steps.

        T040: Enforce 1-20 step limit.
        """
        # Generate 30 steps in mock response
        mock_predictor = Mock()
        mock_prediction = Mock()
        steps = [f"Step {i}: Action {i}" for i in range(1, 31)]
        mock_prediction.reasoning = "\n".join(steps)
        mock_prediction.answer = "Result"
        mock_prediction.confidence = 0.85

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor
        generator.predictor = mock_predictor

        task = TaskInput(
            task_id=str(uuid.uuid4()),
            description="Long reasoning task",
            domain="test",
            max_reasoning_steps=15  # Limit to 15 steps
        )

        output = generator(task)

        # Verify truncation
        assert len(output.reasoning_trace) == 15, \
            "Should truncate at max_reasoning_steps=15"
        assert output.reasoning_trace[0] == "Step 1: Action 1"
        assert output.reasoning_trace[14] == "Step 15: Action 15"

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    def test_generator_handles_multiline_reasoning(self, mock_cot_class, generator):
        """
        Test parsing of complex multiline reasoning with various formats.

        T040: Parse different reasoning formats into structured trace.
        """
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.reasoning = """
        First, I need to understand the problem.

        Then I'll break it down:
        - Part A: Calculate the base
        - Part B: Add the modifier
        - Part C: Verify the result

        Finally, I'll combine everything to get the answer.
        """
        mock_prediction.answer = "Final result"
        mock_prediction.confidence = 0.9

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor
        generator.predictor = mock_predictor

        task = TaskInput(
            task_id=str(uuid.uuid4()),
            description="Complex task",
            domain="test"
        )

        output = generator(task)

        # Should parse into steps, removing empty lines
        assert len(output.reasoning_trace) >= 3
        assert "" not in output.reasoning_trace
        assert all(isinstance(step, str) for step in output.reasoning_trace)


class TestGeneratorErrorHandling:
    """Test error handling in CoTGenerator."""

    def test_generator_raises_on_invalid_task_id(self, generator):
        """Test that missing task_id raises ValueError."""
        invalid_task = TaskInput(
            task_id="",
            description="Valid description"
        )

        with pytest.raises(ValueError, match="task_id is required"):
            generator(invalid_task)

    def test_generator_raises_on_missing_description(self, generator):
        """Test that missing description raises ValueError."""
        invalid_task = TaskInput(
            task_id="task-001",
            description=""
        )

        with pytest.raises(ValueError, match="task description is required"):
            generator(invalid_task)

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    def test_generator_raises_on_execution_failure(self, mock_cot_class, generator):
        """Test that DSPy errors are caught and wrapped in RuntimeError."""
        mock_predictor = Mock()
        mock_predictor.side_effect = Exception("DSPy internal error")

        mock_cot_class.return_value = mock_predictor
        generator.predictor = mock_predictor

        task = TaskInput(
            task_id="task-001",
            description="Valid task"
        )

        with pytest.raises(RuntimeError, match="CoT generation failed"):
            generator(task)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
