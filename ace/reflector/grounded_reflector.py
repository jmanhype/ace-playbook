"""
GroundedReflector Implementation

Analyzes task outcomes using ground-truth and execution feedback to extract
labeled insights (Helpful/Harmful/Neutral) without manual annotation.

Based on contracts/reflector.py and tasks.md T047-T049.
"""

import dspy
import json
import time
import math
import re
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from dspy.utils.exceptions import AdapterParseError

from ace.reflector.signatures import (
    ReflectorInput,
    AnalysisSignature,
    FeedbackType,
    InsightSection
)
from ace.utils.finance_guardrails import get_guardrail
from ace.utils.logging_config import get_logger
from ace.utils.llm_circuit_breaker import protected_predict

logger = get_logger(__name__, component="reflector")


class InsightCandidate(BaseModel):
    """
    Single insight extracted from task analysis.

    T045: Pydantic model for type safety and validation.
    """
    content: str = Field(..., description="Strategy or observation text")
    section: InsightSection = Field(..., description="Helpful/Harmful/Neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    rationale: str = Field(..., description="Explanation linking to feedback")
    tags: List[str] = Field(default_factory=list, description="Domain/category tags")
    referenced_steps: List[int] = Field(default_factory=list, description="Trace step indices")


class ReflectorOutput(BaseModel):
    """
    Output from Reflector analysis.

    T046: Pydantic model with all reflection metadata.
    """
    task_id: str = Field(..., description="Task identifier")
    insights: List[InsightCandidate] = Field(..., description="Extracted insights")
    analysis_summary: str = Field(..., description="High-level outcome summary")

    # Metadata
    referenced_steps: List[int] = Field(default_factory=list, description="Analyzed step indices")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    feedback_types_used: List[FeedbackType] = Field(default_factory=list, description="Signals used")

    # Quality flags
    requires_human_review: bool = Field(default=False, description="High impact + low confidence")
    contradicts_existing: List[str] = Field(default_factory=list, description="Conflicting bullet IDs")


class GroundedReflector:
    """
    Reflector implementation using ground-truth and execution feedback signals.

    T047: Main implementation with DSPy ChainOfThought analysis.
    T048: Ground-truth comparison logic.
    T049: Test result analysis logic.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_insights: int = 10
    ):
        """
        Initialize GroundedReflector.

        Args:
            model: Cost-optimized LLM model (gpt-4o-mini, claude-3-haiku)
            temperature: Sampling temperature (lower = more consistent)
            max_insights: Maximum insights to extract per task (default: 10)
        """
        self.model = model
        self.temperature = temperature
        self.max_insights = max_insights

        # Initialize DSPy ChainOfThought predictor
        self.predictor = dspy.ChainOfThought(AnalysisSignature)

        logger.info(
            "grounded_reflector_initialized",
            model=model,
            temperature=temperature,
            max_insights=max_insights
        )

    def format_task_context(
        self,
        task_id: str,
        reasoning_trace: List[str],
        answer: str,
        confidence: float,
        bullets_referenced: List[str]
    ) -> str:
        """
        Format Generator output into readable context.

        Args:
            task_id: Task identifier
            reasoning_trace: Step-by-step reasoning
            answer: Final answer
            confidence: Generator confidence
            bullets_referenced: Referenced bullet IDs

        Returns:
            Formatted context string
        """
        trace_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(reasoning_trace)])
        bullets_text = ", ".join(bullets_referenced) if bullets_referenced else "None"

        context = f"""Task ID: {task_id}
Reasoning Trace:
{trace_text}

Final Answer: {answer}
Confidence: {confidence:.2f}
Bullets Referenced: {bullets_text}
"""
        return context

    def format_feedback_context(
        self,
        ground_truth: str,
        test_results: str,
        error_messages: List[str],
        performance_metrics: str
    ) -> str:
        """
        Format execution feedback into readable context.

        Args:
            ground_truth: Correct answer (if available)
            test_results: Test outcomes JSON string
            error_messages: Runtime errors
            performance_metrics: Performance data JSON string

        Returns:
            Formatted feedback string
        """
        parts = []

        if ground_truth:
            parts.append(f"Ground Truth: {ground_truth}")

        if test_results:
            parts.append(f"Test Results: {test_results}")

        if error_messages:
            errors_text = "\n".join([f"- {err}" for err in error_messages])
            parts.append(f"Errors:\n{errors_text}")

        if performance_metrics:
            parts.append(f"Performance: {performance_metrics}")

        if not parts:
            parts.append("No execution feedback available")

        return "\n\n".join(parts)

    @staticmethod
    def _answers_match(predicted: str, ground_truth: str) -> bool:
        if ground_truth is None:
            return False

        raw_pred = (predicted or "").strip()
        raw_gt = (ground_truth or "").strip()

        if not raw_pred or not raw_gt:
            return False

        pred_lower = raw_pred.lower()
        gt_lower = raw_gt.lower()

        if gt_lower in {"success", "failure"}:
            success_tokens = {"success", "successful", "done", "completed", "complete", "resolved", "yes"}
            failure_tokens = {"fail", "failed", "failure", "unable", "cannot", "can't", "won't", "error", "issue", "unsuccessful", "no"}

            contains_success = any(token in pred_lower for token in success_tokens)
            contains_failure = any(token in pred_lower for token in failure_tokens)

            if gt_lower == "success":
                return not contains_failure and (contains_success or bool(pred_lower))

            if gt_lower == "failure":
                return contains_failure and not contains_success

        pred_trimmed = pred_lower.rstrip(" .")
        gt_trimmed = gt_lower.rstrip(" .")

        if pred_trimmed == gt_trimmed:
            return True

        if gt_trimmed and pred_trimmed.endswith(gt_trimmed):
            return True

        if gt_trimmed and " " not in gt_trimmed:
            token_pattern = rf"(?<![\w\d]){re.escape(gt_trimmed)}(?![\w\d])"
            if re.search(token_pattern, pred_trimmed):
                return True

        gt_numbers = GroundedReflector._extract_numeric_candidates(raw_gt)
        pred_numbers = GroundedReflector._extract_numeric_candidates(raw_pred)

        if gt_numbers and pred_numbers:
            for target in gt_numbers:
                for candidate in pred_numbers:
                    if GroundedReflector._numbers_match(candidate, target):
                        return True

        return False

    @staticmethod
    def _extract_numeric_candidates(value: str) -> List[float]:
        if not value:
            return []

        cleaned = value.replace(",", "")
        candidates: List[float] = []

        for match in re.finditer(r"-?\d+(?:\.\d+)?", cleaned):
            number_text = match.group()
            try:
                parsed = float(number_text)
            except ValueError:
                continue

            candidates.append(parsed)

            suffix_index = match.end()
            if suffix_index < len(cleaned) and cleaned[suffix_index] == '%':
                candidates.append(parsed / 100.0)

        return candidates

    @staticmethod
    def _numbers_match(candidate: float, target: float) -> bool:
        tolerance = max(1e-6, 0.01 * abs(target))
        return math.isclose(candidate, target, rel_tol=0.01, abs_tol=tolerance)

    def compare_with_ground_truth(
        self,
        task_id: str,
        answer: str,
        ground_truth: str
    ) -> tuple[bool, str]:
        """
        T048: Compare answer with ground truth.

        Args:
            answer: Generator's answer
            ground_truth: Known correct answer

        Returns:
            Tuple of (is_correct, rationale)
        """
        if not ground_truth:
            return False, "No ground truth available for comparison"

        guardrail = get_guardrail(task_id)
        if guardrail:
            is_correct = guardrail.validate(answer, ground_truth)
            if is_correct:
                rationale = "Finance guardrail passed exact-match validation"
            else:
                rationale = (
                    "Finance guardrail mismatch: expected "
                    f"'{ground_truth}', received '{answer.strip() or '<empty>'}'"
                )
            logger.debug(
                "finance_guardrail_validation",
                task_id=task_id,
                is_correct=is_correct
            )
            return is_correct, rationale

        is_correct = self._answers_match(answer, ground_truth)

        if is_correct:
            rationale = f"Answer matches ground truth: '{ground_truth}'"
        else:
            rationale = f"Answer '{answer}' does not match ground truth '{ground_truth}'"

        logger.debug(
            "ground_truth_comparison",
            answer=answer,
            ground_truth=ground_truth,
            is_correct=is_correct
        )

        return is_correct, rationale

    def analyze_test_results(
        self,
        test_results_json: str
    ) -> tuple[bool, Dict[str, bool], str]:
        """
        T049: Analyze test pass/fail outcomes.

        Args:
            test_results_json: JSON string with test outcomes

        Returns:
            Tuple of (all_passed, results_dict, rationale)
        """
        if not test_results_json:
            return False, {}, "No test results available"

        try:
            results = json.loads(test_results_json)
            all_passed = all(results.values())

            passed_tests = [name for name, passed in results.items() if passed]
            failed_tests = [name for name, passed in results.items() if not passed]

            if all_passed:
                rationale = f"All {len(results)} tests passed"
            else:
                rationale = f"{len(failed_tests)} tests failed: {', '.join(failed_tests)}"

            logger.debug(
                "test_results_analysis",
                total_tests=len(results),
                passed=len(passed_tests),
                failed=len(failed_tests),
                all_passed=all_passed
            )

            return all_passed, results, rationale

        except json.JSONDecodeError as e:
            logger.warning("test_results_parse_error", error=str(e))
            return False, {}, f"Invalid test results JSON: {str(e)}"

    def determine_feedback_types(
        self,
        ground_truth: str,
        test_results: str,
        error_messages: List[str],
        performance_metrics: str
    ) -> List[FeedbackType]:
        """
        Identify which feedback signals are available.

        Args:
            ground_truth: Ground truth string
            test_results: Test results JSON
            error_messages: Error messages list
            performance_metrics: Performance JSON

        Returns:
            List of available FeedbackType enum values
        """
        types = []

        if ground_truth:
            types.append(FeedbackType.GROUND_TRUTH)
        if test_results:
            types.append(FeedbackType.TEST_RESULT)
        if error_messages:
            types.append(FeedbackType.ERROR)
        if performance_metrics:
            types.append(FeedbackType.PERFORMANCE)

        return types

    def parse_insights_from_analysis(
        self,
        helpful_text: str,
        harmful_text: str,
        analysis_confidence: float,
        domain: str
    ) -> List[InsightCandidate]:
        """
        Parse LLM analysis output into InsightCandidate objects.

        Args:
            helpful_text: Helpful strategies (one per line)
            harmful_text: Harmful strategies (one per line)
            analysis_confidence: Overall confidence
            domain: Task domain

        Returns:
            List of InsightCandidate objects
        """
        insights = []

        # Parse helpful insights
        helpful_lines = [line.strip() for line in helpful_text.split("\n") if line.strip()]
        for line in helpful_lines[:self.max_insights]:
            insight = InsightCandidate(
                content=line,
                section=InsightSection.HELPFUL,
                confidence=min(analysis_confidence, 0.95),  # Cap at 0.95
                rationale="Strategy contributed to successful task outcome",
                tags=[domain] if domain else [],
                referenced_steps=[]
            )
            insights.append(insight)

        # Parse harmful insights
        harmful_lines = [line.strip() for line in harmful_text.split("\n") if line.strip()]
        for line in harmful_lines[:self.max_insights]:
            insight = InsightCandidate(
                content=line,
                section=InsightSection.HARMFUL,
                confidence=min(analysis_confidence, 0.95),
                rationale="Strategy led to errors or incorrect outcomes",
                tags=[domain] if domain else [],
                referenced_steps=[]
            )
            insights.append(insight)

        logger.debug(
            "insights_parsed",
            num_helpful=len(helpful_lines),
            num_harmful=len(harmful_lines),
            total_extracted=len(insights)
        )

        return insights[:self.max_insights]

    def validate_reflector_input(self, reflector_input: ReflectorInput) -> None:
        """
        Validate ReflectorInput before analysis.

        Args:
            reflector_input: Input to validate

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not reflector_input.task_id:
            raise ValueError("task_id is required")

        if not reflector_input.reasoning_trace:
            raise ValueError("reasoning_trace cannot be empty")

        if not reflector_input.answer:
            raise ValueError("answer is required")

        if not (0.0 <= reflector_input.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

        logger.debug(
            "reflector_input_validated",
            task_id=reflector_input.task_id,
            num_steps=len(reflector_input.reasoning_trace),
            has_ground_truth=bool(reflector_input.ground_truth)
        )

    def __call__(self, reflector_input: ReflectorInput) -> ReflectorOutput:
        """
        Analyze task outcome and extract labeled insights.

        Args:
            reflector_input: ReflectorInput with Generator output + feedback

        Returns:
            ReflectorOutput with labeled insights

        Raises:
            ValueError: If input is invalid
            RuntimeError: If analysis fails
        """
        start_time = time.time()

        # Validate input
        self.validate_reflector_input(reflector_input)

        logger.info(
            "reflection_start",
            task_id=reflector_input.task_id,
            domain=reflector_input.domain,
            num_trace_steps=len(reflector_input.reasoning_trace)
        )

        try:
            # Format contexts
            task_context = self.format_task_context(
                reflector_input.task_id,
                reflector_input.reasoning_trace,
                reflector_input.answer,
                reflector_input.confidence,
                reflector_input.bullets_referenced
            )

            feedback_context = self.format_feedback_context(
                reflector_input.ground_truth,
                reflector_input.test_results,
                reflector_input.error_messages,
                reflector_input.performance_metrics
            )

            # T048: Compare with ground truth if available
            is_correct = False
            correctness_rationale = ""
            if reflector_input.ground_truth:
                is_correct, correctness_rationale = self.compare_with_ground_truth(
                    reflector_input.task_id,
                    reflector_input.answer,
                    reflector_input.ground_truth
                )

            # T049: Analyze test results if available
            tests_passed = False
            test_rationale = ""
            if reflector_input.test_results:
                tests_passed, _, test_rationale = self.analyze_test_results(
                    reflector_input.test_results
                )

            # T071: Call DSPy predictor for analysis with circuit breaker protection
            try:
                prediction = protected_predict(
                    self.predictor,
                    circuit_name="reflector",
                    failure_threshold=5,
                    recovery_timeout=60,
                    task_context=task_context,
                    feedback_context=feedback_context,
                    domain=reflector_input.domain
                )
            except AdapterParseError as parse_error:
                return self._handle_adapter_parse_failure(
                    reflector_input=reflector_input,
                    start_time=start_time,
                    error=parse_error
                )

            # Parse insights from analysis
            insights = self.parse_insights_from_analysis(
                prediction.helpful_insights,
                prediction.harmful_insights,
                float(prediction.confidence),
                reflector_input.domain
            )

            if reflector_input.ground_truth and not is_correct:
                suppressed_count = sum(1 for ins in insights if ins.section == InsightSection.HELPFUL)
                if suppressed_count:
                    insights = [
                        ins for ins in insights if ins.section != InsightSection.HELPFUL
                    ]
                    logger.info(
                        "helpful_insights_suppressed_due_to_ground_truth_failure",
                        task_id=reflector_input.task_id,
                        suppressed=suppressed_count
                    )

            # Determine feedback types used
            feedback_types = self.determine_feedback_types(
                reflector_input.ground_truth,
                reflector_input.test_results,
                reflector_input.error_messages,
                reflector_input.performance_metrics
            )

            # Determine if human review is required
            requires_review = float(prediction.confidence) < 0.7 and len(insights) > 0

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Construct ReflectorOutput
            output = ReflectorOutput(
                task_id=reflector_input.task_id,
                insights=insights,
                analysis_summary=prediction.analysis,
                referenced_steps=list(range(len(reflector_input.reasoning_trace))),
                confidence_score=float(prediction.confidence),
                feedback_types_used=feedback_types,
                requires_human_review=requires_review,
                contradicts_existing=[]
            )

            logger.info(
                "reflection_complete",
                task_id=reflector_input.task_id,
                num_insights=len(insights),
                confidence=output.confidence_score,
                requires_review=requires_review,
                latency_ms=latency_ms
            )

            return output

        except Exception as e:
            logger.error(
                "reflection_failed",
                task_id=reflector_input.task_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise RuntimeError(f"Reflection failed: {str(e)}") from e

    def _handle_adapter_parse_failure(
        self,
        reflector_input: ReflectorInput,
        start_time: float,
        error: Exception
    ) -> ReflectorOutput:
        """Produce a safe fallback output when DSPy adapter parsing fails."""

        latency_ms = int((time.time() - start_time) * 1000)

        feedback_types = self.determine_feedback_types(
            reflector_input.ground_truth,
            reflector_input.test_results,
            reflector_input.error_messages,
            reflector_input.performance_metrics
        )

        summary = (
            "Reflection parsing failed; returning empty insights. "
            f"Original error: {error}"
        )

        output = ReflectorOutput(
            task_id=reflector_input.task_id,
            insights=[],
            analysis_summary=summary,
            referenced_steps=list(range(len(reflector_input.reasoning_trace))),
            confidence_score=0.0,
            feedback_types_used=feedback_types,
            requires_human_review=True,
            contradicts_existing=[]
        )

        logger.warning(
            "reflection_parse_failure_recovered",
            task_id=reflector_input.task_id,
            latency_ms=latency_ms,
            error=str(error)
        )

        return output


def create_grounded_reflector(
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_insights: int = 10
) -> GroundedReflector:
    """
    Factory function to create GroundedReflector instance.

    Args:
        model: LLM model name
        temperature: Sampling temperature
        max_insights: Maximum insights per task

    Returns:
        Initialized GroundedReflector instance
    """
    return GroundedReflector(
        model=model,
        temperature=temperature,
        max_insights=max_insights
    )


__version__ = "v1.0.0"
