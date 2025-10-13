"""
Unit Tests for Reflector Module

Tests for GroundedReflector, insight labeling, ground-truth comparison,
and test result analysis.

Coverage:
- T050: Unit tests for insight labeling logic
- Ground-truth comparison (T048)
- Test result analysis (T049)
- Insight parsing and confidence thresholds
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import List

from ace.reflector import (
    GroundedReflector,
    ReflectorInput,
    ReflectorOutput,
    InsightCandidate,
    FeedbackType,
    InsightSection,
    create_grounded_reflector
)


class TestGroundedReflector:
    """Test suite for GroundedReflector class."""

    def test_initialization(self):
        """Test GroundedReflector can be initialized with default params."""
        reflector = GroundedReflector()

        assert reflector.model == "gpt-4o-mini"
        assert reflector.temperature == 0.3
        assert reflector.max_insights == 10

    def test_initialization_with_custom_params(self):
        """Test GroundedReflector initialization with custom parameters."""
        reflector = GroundedReflector(
            model="claude-3-haiku",
            temperature=0.5,
            max_insights=5
        )

        assert reflector.model == "claude-3-haiku"
        assert reflector.temperature == 0.5
        assert reflector.max_insights == 5

    def test_factory_function(self):
        """Test create_grounded_reflector factory function."""
        reflector = create_grounded_reflector(model="gpt-4o-mini", temperature=0.2)

        assert isinstance(reflector, GroundedReflector)
        assert reflector.model == "gpt-4o-mini"
        assert reflector.temperature == 0.2


class TestGroundTruthComparison:
    """Test suite for ground-truth comparison (T048)."""

    def test_compare_correct_answer(self):
        """Test comparison when answer matches ground truth."""
        reflector = GroundedReflector()
        answer = "42"
        ground_truth = "42"

        is_correct, rationale = reflector.compare_with_ground_truth(answer, ground_truth)

        assert is_correct is True
        assert "matches ground truth" in rationale
        assert "42" in rationale

    def test_compare_incorrect_answer(self):
        """Test comparison when answer does not match ground truth."""
        reflector = GroundedReflector()
        answer = "41"
        ground_truth = "42"

        is_correct, rationale = reflector.compare_with_ground_truth(answer, ground_truth)

        assert is_correct is False
        assert "does not match" in rationale
        assert "41" in rationale
        assert "42" in rationale

    def test_compare_case_insensitive(self):
        """Test comparison is case-insensitive."""
        reflector = GroundedReflector()
        answer = "Paris"
        ground_truth = "paris"

        is_correct, rationale = reflector.compare_with_ground_truth(answer, ground_truth)

        assert is_correct is True

    def test_compare_whitespace_normalization(self):
        """Test comparison normalizes whitespace."""
        reflector = GroundedReflector()
        answer = "  42  "
        ground_truth = "42"

        is_correct, rationale = reflector.compare_with_ground_truth(answer, ground_truth)

        assert is_correct is True

    def test_compare_no_ground_truth(self):
        """Test comparison when ground truth is missing."""
        reflector = GroundedReflector()
        answer = "42"
        ground_truth = ""

        is_correct, rationale = reflector.compare_with_ground_truth(answer, ground_truth)

        assert is_correct is False
        assert "No ground truth available" in rationale


class TestTestResultAnalysis:
    """Test suite for test result analysis (T049)."""

    def test_analyze_all_tests_passed(self):
        """Test analysis when all tests pass."""
        reflector = GroundedReflector()
        test_results = json.dumps({
            "test_addition": True,
            "test_multiplication": True,
            "test_subtraction": True
        })

        all_passed, results, rationale = reflector.analyze_test_results(test_results)

        assert all_passed is True
        assert len(results) == 3
        assert "All 3 tests passed" in rationale

    def test_analyze_some_tests_failed(self):
        """Test analysis when some tests fail."""
        reflector = GroundedReflector()
        test_results = json.dumps({
            "test_addition": True,
            "test_multiplication": False,
            "test_subtraction": False
        })

        all_passed, results, rationale = reflector.analyze_test_results(test_results)

        assert all_passed is False
        assert len(results) == 3
        assert "2 tests failed" in rationale
        assert "test_multiplication" in rationale
        assert "test_subtraction" in rationale

    def test_analyze_empty_test_results(self):
        """Test analysis when test results are empty."""
        reflector = GroundedReflector()
        test_results = ""

        all_passed, results, rationale = reflector.analyze_test_results(test_results)

        assert all_passed is False
        assert len(results) == 0
        assert "No test results available" in rationale

    def test_analyze_invalid_json(self):
        """Test analysis when test results JSON is invalid."""
        reflector = GroundedReflector()
        test_results = "not valid json"

        all_passed, results, rationale = reflector.analyze_test_results(test_results)

        assert all_passed is False
        assert len(results) == 0
        assert "Invalid test results JSON" in rationale

    def test_analyze_mixed_test_results(self):
        """Test analysis with complex test outcomes."""
        reflector = GroundedReflector()
        test_results = json.dumps({
            "test_edge_case_1": True,
            "test_edge_case_2": True,
            "test_performance": False,
            "test_integration": True
        })

        all_passed, results, rationale = reflector.analyze_test_results(test_results)

        assert all_passed is False
        assert results["test_edge_case_1"] is True
        assert results["test_performance"] is False
        assert "1 tests failed" in rationale


class TestFeedbackTypeDetection:
    """Test suite for feedback signal detection."""

    def test_determine_all_feedback_types(self):
        """Test detection when all feedback signals present."""
        reflector = GroundedReflector()
        ground_truth = "42"
        test_results = '{"test": true}'
        error_messages = ["ValueError: invalid input"]
        performance_metrics = '{"latency_ms": 150}'

        types = reflector.determine_feedback_types(
            ground_truth, test_results, error_messages, performance_metrics
        )

        assert FeedbackType.GROUND_TRUTH in types
        assert FeedbackType.TEST_RESULT in types
        assert FeedbackType.ERROR in types
        assert FeedbackType.PERFORMANCE in types

    def test_determine_only_ground_truth(self):
        """Test detection when only ground truth available."""
        reflector = GroundedReflector()
        ground_truth = "42"
        test_results = ""
        error_messages = []
        performance_metrics = ""

        types = reflector.determine_feedback_types(
            ground_truth, test_results, error_messages, performance_metrics
        )

        assert FeedbackType.GROUND_TRUTH in types
        assert FeedbackType.TEST_RESULT not in types
        assert FeedbackType.ERROR not in types
        assert FeedbackType.PERFORMANCE not in types

    def test_determine_no_feedback(self):
        """Test detection when no feedback signals present."""
        reflector = GroundedReflector()
        ground_truth = ""
        test_results = ""
        error_messages = []
        performance_metrics = ""

        types = reflector.determine_feedback_types(
            ground_truth, test_results, error_messages, performance_metrics
        )

        assert len(types) == 0


class TestInsightParsing:
    """Test suite for insight parsing from LLM analysis."""

    def test_parse_helpful_insights(self):
        """Test parsing helpful insights from analysis."""
        reflector = GroundedReflector()
        helpful_text = """Break problems into smaller steps
Use explicit variable names
Verify intermediate results"""
        harmful_text = ""
        confidence = 0.85
        domain = "arithmetic"

        insights = reflector.parse_insights_from_analysis(
            helpful_text, harmful_text, confidence, domain
        )

        assert len(insights) == 3
        assert all(i.section == InsightSection.HELPFUL for i in insights)
        assert insights[0].content == "Break problems into smaller steps"
        assert insights[1].content == "Use explicit variable names"
        assert insights[2].content == "Verify intermediate results"
        assert all(i.confidence <= 0.95 for i in insights)  # Capped at 0.95
        assert all("arithmetic" in i.tags for i in insights)

    def test_parse_harmful_insights(self):
        """Test parsing harmful insights from analysis."""
        reflector = GroundedReflector()
        helpful_text = ""
        harmful_text = """Skipping input validation
Ignoring edge cases
Assuming sorted input"""
        confidence = 0.9
        domain = "code_gen"

        insights = reflector.parse_insights_from_analysis(
            helpful_text, harmful_text, confidence, domain
        )

        assert len(insights) == 3
        assert all(i.section == InsightSection.HARMFUL for i in insights)
        assert insights[0].content == "Skipping input validation"
        assert insights[1].content == "Ignoring edge cases"
        assert all("code_gen" in i.tags for i in insights)

    def test_parse_mixed_insights(self):
        """Test parsing both helpful and harmful insights."""
        reflector = GroundedReflector()
        helpful_text = """Strategy A worked well
Strategy B was effective"""
        harmful_text = """Strategy C caused errors
Strategy D was inefficient"""
        confidence = 0.8
        domain = "general"

        insights = reflector.parse_insights_from_analysis(
            helpful_text, harmful_text, confidence, domain
        )

        assert len(insights) == 4
        helpful_insights = [i for i in insights if i.section == InsightSection.HELPFUL]
        harmful_insights = [i for i in insights if i.section == InsightSection.HARMFUL]
        assert len(helpful_insights) == 2
        assert len(harmful_insights) == 2

    def test_parse_truncates_at_max_insights(self):
        """Test parsing respects max_insights limit."""
        reflector = GroundedReflector(max_insights=5)

        # Generate 10 helpful and 10 harmful insights
        helpful_text = "\n".join([f"Helpful strategy {i}" for i in range(10)])
        harmful_text = "\n".join([f"Harmful strategy {i}" for i in range(10)])
        confidence = 0.8
        domain = "general"

        insights = reflector.parse_insights_from_analysis(
            helpful_text, harmful_text, confidence, domain
        )

        # Should be truncated to max_insights (5 total)
        assert len(insights) <= 5

    def test_parse_confidence_capped(self):
        """Test that insight confidence is capped at 0.95."""
        reflector = GroundedReflector()
        helpful_text = "Strategy A"
        harmful_text = ""
        confidence = 0.99  # Very high confidence
        domain = "general"

        insights = reflector.parse_insights_from_analysis(
            helpful_text, harmful_text, confidence, domain
        )

        assert len(insights) == 1
        assert insights[0].confidence == 0.95  # Capped


class TestReflectorInputValidation:
    """Test suite for ReflectorInput validation."""

    def test_validate_valid_reflector_input(self):
        """Test validation passes for valid ReflectorInput."""
        reflector = GroundedReflector()
        reflector_input = ReflectorInput(
            task_id="task-001",
            reasoning_trace=["Step 1: Analyze", "Step 2: Calculate"],
            answer="42",
            confidence=0.95,
            ground_truth="42"
        )

        # Should not raise
        reflector.validate_reflector_input(reflector_input)

    def test_validate_missing_task_id(self):
        """Test validation fails when task_id is missing."""
        reflector = GroundedReflector()
        reflector_input = ReflectorInput(
            task_id="",
            reasoning_trace=["Step 1"],
            answer="42",
            confidence=0.95
        )

        with pytest.raises(ValueError, match="task_id is required"):
            reflector.validate_reflector_input(reflector_input)

    def test_validate_empty_reasoning_trace(self):
        """Test validation fails when reasoning_trace is empty."""
        reflector = GroundedReflector()
        reflector_input = ReflectorInput(
            task_id="task-001",
            reasoning_trace=[],
            answer="42",
            confidence=0.95
        )

        with pytest.raises(ValueError, match="reasoning_trace cannot be empty"):
            reflector.validate_reflector_input(reflector_input)

    def test_validate_missing_answer(self):
        """Test validation fails when answer is missing."""
        reflector = GroundedReflector()
        reflector_input = ReflectorInput(
            task_id="task-001",
            reasoning_trace=["Step 1"],
            answer="",
            confidence=0.95
        )

        with pytest.raises(ValueError, match="answer is required"):
            reflector.validate_reflector_input(reflector_input)

    def test_validate_invalid_confidence_too_low(self):
        """Test validation fails when confidence < 0.0."""
        reflector = GroundedReflector()
        reflector_input = ReflectorInput(
            task_id="task-001",
            reasoning_trace=["Step 1"],
            answer="42",
            confidence=-0.1
        )

        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            reflector.validate_reflector_input(reflector_input)

    def test_validate_invalid_confidence_too_high(self):
        """Test validation fails when confidence > 1.0."""
        reflector = GroundedReflector()
        reflector_input = ReflectorInput(
            task_id="task-001",
            reasoning_trace=["Step 1"],
            answer="42",
            confidence=1.5
        )

        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            reflector.validate_reflector_input(reflector_input)


class TestContextFormatting:
    """Test suite for task and feedback context formatting."""

    def test_format_task_context(self):
        """Test task context formatting."""
        reflector = GroundedReflector()
        context = reflector.format_task_context(
            task_id="task-001",
            reasoning_trace=["Step 1: Analyze", "Step 2: Calculate"],
            answer="42",
            confidence=0.95,
            bullets_referenced=["bullet-abc", "bullet-def"]
        )

        assert "Task ID: task-001" in context
        assert "1. Step 1: Analyze" in context
        assert "2. Step 2: Calculate" in context
        assert "Final Answer: 42" in context
        assert "Confidence: 0.95" in context
        assert "bullet-abc, bullet-def" in context

    def test_format_task_context_no_bullets(self):
        """Test task context formatting with no bullets referenced."""
        reflector = GroundedReflector()
        context = reflector.format_task_context(
            task_id="task-002",
            reasoning_trace=["Step 1"],
            answer="Result",
            confidence=0.8,
            bullets_referenced=[]
        )

        assert "Bullets Referenced: None" in context

    def test_format_feedback_context_all_signals(self):
        """Test feedback context formatting with all signals."""
        reflector = GroundedReflector()
        context = reflector.format_feedback_context(
            ground_truth="42",
            test_results='{"test": true}',
            error_messages=["Error 1", "Error 2"],
            performance_metrics='{"latency_ms": 100}'
        )

        assert "Ground Truth: 42" in context
        assert "Test Results:" in context
        assert "Errors:" in context
        assert "- Error 1" in context
        assert "- Error 2" in context
        assert "Performance:" in context

    def test_format_feedback_context_no_signals(self):
        """Test feedback context formatting with no signals."""
        reflector = GroundedReflector()
        context = reflector.format_feedback_context(
            ground_truth="",
            test_results="",
            error_messages=[],
            performance_metrics=""
        )

        assert "No execution feedback available" in context


class TestReflectorExecution:
    """Test suite for end-to-end Reflector execution (mocked)."""

    @patch('ace.reflector.grounded_reflector.dspy.ChainOfThought')
    def test_execute_with_correct_answer(self, mock_cot_class):
        """Test reflection when answer is correct (Helpful insights)."""
        # Mock the predictor response
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.analysis = "Task completed successfully with correct answer."
        mock_prediction.helpful_insights = "Breaking problem into steps\nVerifying intermediate results"
        mock_prediction.harmful_insights = ""
        mock_prediction.confidence = 0.9

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor

        # Create reflector and execute
        reflector = GroundedReflector()
        reflector.predictor = mock_predictor

        reflector_input = ReflectorInput(
            task_id="task-001",
            reasoning_trace=["Step 1: Identify problem", "Step 2: Calculate"],
            answer="42",
            confidence=0.95,
            ground_truth="42",
            domain="arithmetic"
        )

        output = reflector(reflector_input)

        # Assertions
        assert output.task_id == "task-001"
        assert len(output.insights) == 2
        assert all(i.section == InsightSection.HELPFUL for i in output.insights)
        assert output.confidence_score == 0.9
        assert FeedbackType.GROUND_TRUTH in output.feedback_types_used
        assert output.requires_human_review is False

    @patch('ace.reflector.grounded_reflector.dspy.ChainOfThought')
    def test_execute_with_incorrect_answer(self, mock_cot_class):
        """Test reflection when answer is incorrect (Harmful insights)."""
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.analysis = "Task failed due to incorrect reasoning."
        mock_prediction.helpful_insights = ""
        mock_prediction.harmful_insights = "Skipped validation step\nAssumed wrong input format"
        mock_prediction.confidence = 0.85

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor

        reflector = GroundedReflector()
        reflector.predictor = mock_predictor

        reflector_input = ReflectorInput(
            task_id="task-002",
            reasoning_trace=["Step 1: Assume input", "Step 2: Calculate"],
            answer="41",
            confidence=0.6,
            ground_truth="42",
            domain="arithmetic"
        )

        output = reflector(reflector_input)

        assert output.task_id == "task-002"
        assert len(output.insights) == 2
        assert all(i.section == InsightSection.HARMFUL for i in output.insights)
        assert output.requires_human_review is False

    @patch('ace.reflector.grounded_reflector.dspy.ChainOfThought')
    def test_execute_requires_human_review(self, mock_cot_class):
        """Test that low confidence triggers human review flag."""
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.analysis = "Uncertain analysis."
        mock_prediction.helpful_insights = "Strategy A"
        mock_prediction.harmful_insights = ""
        mock_prediction.confidence = 0.6  # Low confidence

        mock_predictor.return_value = mock_prediction
        mock_cot_class.return_value = mock_predictor

        reflector = GroundedReflector()
        reflector.predictor = mock_predictor

        reflector_input = ReflectorInput(
            task_id="task-003",
            reasoning_trace=["Step 1"],
            answer="Result",
            confidence=0.5,
            domain="general"
        )

        output = reflector(reflector_input)

        assert output.requires_human_review is True  # confidence < 0.7

    def test_execute_with_invalid_input_raises(self):
        """Test that invalid ReflectorInput raises ValueError."""
        reflector = GroundedReflector()

        invalid_input = ReflectorInput(
            task_id="",  # Empty task_id
            reasoning_trace=["Step 1"],
            answer="42",
            confidence=0.95
        )

        with pytest.raises(ValueError):
            reflector(invalid_input)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
