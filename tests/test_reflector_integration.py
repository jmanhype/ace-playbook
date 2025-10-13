"""
Integration Tests for Reflector with Generator

Tests end-to-end feedback learning workflow: Generator → Reflector with
objective labeling based on execution feedback.

Coverage:
- T051: Integration test for feedback learning
- Generator→Reflector workflow
- Correct answer → Helpful insights
- Incorrect answer → Harmful insights
- Test pass/fail → insight labeling
"""

import pytest
from unittest.mock import Mock, patch
import json

from ace.generator import CoTGenerator, TaskInput, TaskOutput
from ace.reflector import (
    GroundedReflector,
    ReflectorInput,
    InsightSection,
    FeedbackType
)


class TestGeneratorReflectorIntegration:
    """Integration tests for Generator→Reflector workflow."""

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    @patch('ace.reflector.grounded_reflector.dspy.ChainOfThought')
    def test_correct_answer_generates_helpful_insights(
        self,
        mock_reflector_cot,
        mock_generator_cot
    ):
        """
        T051: Test that correct answer generates Helpful insights.

        Workflow:
        1. Generator produces correct answer with reasoning
        2. Reflector compares with ground truth
        3. Extracts Helpful insights from successful reasoning
        """
        # Mock Generator predictor
        generator_predictor = Mock()
        generator_prediction = Mock()
        generator_prediction.reasoning = """Step 1: Break down 6 * 7
Step 2: Calculate: 6 * 7 = 42
Step 3: Verify result"""
        generator_prediction.answer = "42"
        generator_prediction.confidence = 0.95

        generator_predictor.return_value = generator_prediction
        mock_generator_cot.return_value = generator_predictor

        # Mock Reflector predictor
        reflector_predictor = Mock()
        reflector_prediction = Mock()
        reflector_prediction.analysis = "Correct answer. Breaking down arithmetic was effective."
        reflector_prediction.helpful_insights = """Breaking arithmetic into steps
Verifying intermediate calculations"""
        reflector_prediction.harmful_insights = ""
        reflector_prediction.confidence = 0.9

        reflector_predictor.return_value = reflector_prediction
        mock_reflector_cot.return_value = reflector_predictor

        # Create modules
        generator = CoTGenerator(model="gpt-4-turbo")
        generator.predictor = generator_predictor

        reflector = GroundedReflector(model="gpt-4o-mini")
        reflector.predictor = reflector_predictor

        # Execute Generator
        task = TaskInput(
            task_id="task-001",
            description="Calculate 6 * 7",
            domain="arithmetic",
            playbook_bullets=["Break problems into steps"],
            max_reasoning_steps=10
        )

        generator_output = generator(task)

        # Execute Reflector with ground truth
        reflector_input = ReflectorInput(
            task_id=generator_output.task_id,
            reasoning_trace=generator_output.reasoning_trace,
            answer=generator_output.answer,
            confidence=generator_output.confidence,
            bullets_referenced=generator_output.bullets_referenced,
            ground_truth="42",  # Correct answer
            domain="arithmetic"
        )

        reflector_output = reflector(reflector_input)

        # Assertions
        assert reflector_output.task_id == "task-001"
        assert len(reflector_output.insights) == 2
        assert all(i.section == InsightSection.HELPFUL for i in reflector_output.insights)
        assert "Breaking arithmetic into steps" in reflector_output.insights[0].content
        assert FeedbackType.GROUND_TRUTH in reflector_output.feedback_types_used

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    @patch('ace.reflector.grounded_reflector.dspy.ChainOfThought')
    def test_incorrect_answer_generates_harmful_insights(
        self,
        mock_reflector_cot,
        mock_generator_cot
    ):
        """
        T051: Test that incorrect answer generates Harmful insights.

        Workflow:
        1. Generator produces incorrect answer
        2. Reflector detects mismatch with ground truth
        3. Extracts Harmful insights from failed reasoning
        """
        # Mock Generator predictor (produces wrong answer)
        generator_predictor = Mock()
        generator_prediction = Mock()
        generator_prediction.reasoning = """Step 1: Assume 6 + 7
Step 2: Calculate: 6 + 7 = 13"""
        generator_prediction.answer = "13"  # Wrong (should be 42)
        generator_prediction.confidence = 0.7

        generator_predictor.return_value = generator_prediction
        mock_generator_cot.return_value = generator_predictor

        # Mock Reflector predictor
        reflector_predictor = Mock()
        reflector_prediction = Mock()
        reflector_prediction.analysis = "Incorrect answer. Used addition instead of multiplication."
        reflector_prediction.helpful_insights = ""
        reflector_prediction.harmful_insights = """Misinterpreted operation (used addition instead of multiplication)
Did not verify operation type before calculating"""
        reflector_prediction.confidence = 0.85

        reflector_predictor.return_value = reflector_prediction
        mock_reflector_cot.return_value = reflector_predictor

        # Create modules
        generator = CoTGenerator(model="gpt-4-turbo")
        generator.predictor = generator_predictor

        reflector = GroundedReflector(model="gpt-4o-mini")
        reflector.predictor = reflector_predictor

        # Execute Generator
        task = TaskInput(
            task_id="task-002",
            description="Calculate 6 * 7",
            domain="arithmetic"
        )

        generator_output = generator(task)

        # Execute Reflector with ground truth
        reflector_input = ReflectorInput(
            task_id=generator_output.task_id,
            reasoning_trace=generator_output.reasoning_trace,
            answer=generator_output.answer,
            confidence=generator_output.confidence,
            bullets_referenced=generator_output.bullets_referenced,
            ground_truth="42",  # Correct answer (doesn't match "13")
            domain="arithmetic"
        )

        reflector_output = reflector(reflector_input)

        # Assertions
        assert reflector_output.task_id == "task-002"
        assert len(reflector_output.insights) == 2
        assert all(i.section == InsightSection.HARMFUL for i in reflector_output.insights)
        assert "addition instead of multiplication" in reflector_output.insights[0].content.lower()

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    @patch('ace.reflector.grounded_reflector.dspy.ChainOfThought')
    def test_test_pass_generates_helpful_insights(
        self,
        mock_reflector_cot,
        mock_generator_cot
    ):
        """
        T051: Test that passing tests generate Helpful insights.

        Workflow:
        1. Generator produces solution
        2. Tests pass
        3. Reflector extracts Helpful strategies from passing tests
        """
        # Mock Generator predictor
        generator_predictor = Mock()
        generator_prediction = Mock()
        generator_prediction.reasoning = """Step 1: Define function with input validation
Step 2: Implement core logic with edge case handling
Step 3: Return result"""
        generator_prediction.answer = "def add(a, b): return a + b"
        generator_prediction.confidence = 0.9

        generator_predictor.return_value = generator_prediction
        mock_generator_cot.return_value = generator_predictor

        # Mock Reflector predictor
        reflector_predictor = Mock()
        reflector_prediction = Mock()
        reflector_prediction.analysis = "All tests passed. Input validation was effective."
        reflector_prediction.helpful_insights = """Adding input validation upfront
Handling edge cases explicitly"""
        reflector_prediction.harmful_insights = ""
        reflector_prediction.confidence = 0.88

        reflector_predictor.return_value = reflector_prediction
        mock_reflector_cot.return_value = reflector_predictor

        # Create modules
        generator = CoTGenerator(model="gpt-4-turbo")
        generator.predictor = generator_predictor

        reflector = GroundedReflector(model="gpt-4o-mini")
        reflector.predictor = reflector_predictor

        # Execute Generator
        task = TaskInput(
            task_id="task-003",
            description="Write an addition function",
            domain="code_gen"
        )

        generator_output = generator(task)

        # Execute Reflector with test results
        test_results = json.dumps({
            "test_basic_addition": True,
            "test_negative_numbers": True,
            "test_zero": True
        })

        reflector_input = ReflectorInput(
            task_id=generator_output.task_id,
            reasoning_trace=generator_output.reasoning_trace,
            answer=generator_output.answer,
            confidence=generator_output.confidence,
            test_results=test_results,  # All tests passed
            domain="code_gen"
        )

        reflector_output = reflector(reflector_input)

        # Assertions
        assert len(reflector_output.insights) == 2
        assert all(i.section == InsightSection.HELPFUL for i in reflector_output.insights)
        assert FeedbackType.TEST_RESULT in reflector_output.feedback_types_used

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    @patch('ace.reflector.grounded_reflector.dspy.ChainOfThought')
    def test_test_fail_generates_harmful_insights(
        self,
        mock_reflector_cot,
        mock_generator_cot
    ):
        """
        T051: Test that failing tests generate Harmful insights.

        Workflow:
        1. Generator produces solution
        2. Some tests fail
        3. Reflector extracts Harmful strategies from failures
        """
        # Mock Generator predictor
        generator_predictor = Mock()
        generator_prediction = Mock()
        generator_prediction.reasoning = """Step 1: Define function
Step 2: Implement basic logic (no validation)"""
        generator_prediction.answer = "def add(a, b): return a + b"
        generator_prediction.confidence = 0.75

        generator_predictor.return_value = generator_prediction
        mock_generator_cot.return_value = generator_predictor

        # Mock Reflector predictor
        reflector_predictor = Mock()
        reflector_prediction = Mock()
        reflector_prediction.analysis = "Tests failed due to missing input validation."
        reflector_prediction.helpful_insights = ""
        reflector_prediction.harmful_insights = """Skipping input type validation
Not handling None inputs"""
        reflector_prediction.confidence = 0.82

        reflector_predictor.return_value = reflector_prediction
        mock_reflector_cot.return_value = reflector_predictor

        # Create modules
        generator = CoTGenerator(model="gpt-4-turbo")
        generator.predictor = generator_predictor

        reflector = GroundedReflector(model="gpt-4o-mini")
        reflector.predictor = reflector_predictor

        # Execute Generator
        task = TaskInput(
            task_id="task-004",
            description="Write an addition function",
            domain="code_gen"
        )

        generator_output = generator(task)

        # Execute Reflector with test failures
        test_results = json.dumps({
            "test_basic_addition": True,
            "test_type_validation": False,  # Failed
            "test_none_input": False  # Failed
        })

        reflector_input = ReflectorInput(
            task_id=generator_output.task_id,
            reasoning_trace=generator_output.reasoning_trace,
            answer=generator_output.answer,
            confidence=generator_output.confidence,
            test_results=test_results,
            domain="code_gen"
        )

        reflector_output = reflector(reflector_input)

        # Assertions
        assert len(reflector_output.insights) == 2
        assert all(i.section == InsightSection.HARMFUL for i in reflector_output.insights)
        assert "validation" in reflector_output.insights[0].content.lower()

    @patch('ace.generator.cot_generator.dspy.ChainOfThought')
    @patch('ace.reflector.grounded_reflector.dspy.ChainOfThought')
    def test_multiple_feedback_signals(
        self,
        mock_reflector_cot,
        mock_generator_cot
    ):
        """
        T051: Test reflection with multiple feedback signals.

        Combines ground truth, test results, and errors.
        """
        # Mock Generator predictor
        generator_predictor = Mock()
        generator_prediction = Mock()
        generator_prediction.reasoning = """Step 1: Parse input
Step 2: Process data
Step 3: Return output"""
        generator_prediction.answer = "Result"
        generator_prediction.confidence = 0.8

        generator_predictor.return_value = generator_prediction
        mock_generator_cot.return_value = generator_predictor

        # Mock Reflector predictor
        reflector_predictor = Mock()
        reflector_prediction = Mock()
        reflector_prediction.analysis = "Mixed outcome: correct answer but runtime errors."
        reflector_prediction.helpful_insights = "Correct parsing logic"
        reflector_prediction.harmful_insights = "Missing error handling for edge cases"
        reflector_prediction.confidence = 0.75

        reflector_predictor.return_value = reflector_prediction
        mock_reflector_cot.return_value = reflector_predictor

        # Create modules
        generator = CoTGenerator(model="gpt-4-turbo")
        generator.predictor = generator_predictor

        reflector = GroundedReflector(model="gpt-4o-mini")
        reflector.predictor = reflector_predictor

        # Execute Generator
        task = TaskInput(
            task_id="task-005",
            description="Process data pipeline",
            domain="agent_workflow"
        )

        generator_output = generator(task)

        # Execute Reflector with multiple signals
        reflector_input = ReflectorInput(
            task_id=generator_output.task_id,
            reasoning_trace=generator_output.reasoning_trace,
            answer=generator_output.answer,
            confidence=generator_output.confidence,
            ground_truth="Result",  # Matches
            test_results='{"test_main": true, "test_edge": false}',
            error_messages=["IndexError: list index out of range"],
            domain="agent_workflow"
        )

        reflector_output = reflector(reflector_input)

        # Assertions
        assert FeedbackType.GROUND_TRUTH in reflector_output.feedback_types_used
        assert FeedbackType.TEST_RESULT in reflector_output.feedback_types_used
        assert FeedbackType.ERROR in reflector_output.feedback_types_used
        assert len(reflector_output.insights) == 2
        helpful = [i for i in reflector_output.insights if i.section == InsightSection.HELPFUL]
        harmful = [i for i in reflector_output.insights if i.section == InsightSection.HARMFUL]
        assert len(helpful) == 1
        assert len(harmful) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
