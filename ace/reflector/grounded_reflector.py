"""
GroundedReflector Implementation

Analyzes task outcomes using ground-truth and execution feedback to extract
labeled insights (Helpful/Harmful/Neutral) without manual annotation.

Based on contracts/reflector.py and tasks.md T047-T049.
"""

import dspy
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ace.reflector.signatures import (
    ReflectorInput,
    AnalysisSignature,
    FeedbackType,
    InsightSection
)
from ace.utils.logging_config import get_logger
from ace.utils.llm_circuit_breaker import protected_predict

logger = get_logger(__name__, component="reflector")


class InsightCandidate(BaseModel):
    """
    Single insight extracted from task analysis.

    T045: Pydantic model for type safety and validation.
    T022: Extended with tool-calling metadata for ReAct agents.
    """
    content: str = Field(..., description="Strategy or observation text")
    section: InsightSection = Field(..., description="Helpful/Harmful/Neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    rationale: str = Field(..., description="Explanation linking to feedback")
    tags: List[str] = Field(default_factory=list, description="Domain/category tags")
    referenced_steps: List[int] = Field(default_factory=list, description="Trace step indices")

    # T022: Tool-calling metadata (optional, only for ReAct tasks)
    tool_sequence: Optional[List[str]] = Field(None, description="Ordered tool names used")
    tool_success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Success rate for this sequence")
    avg_iterations: Optional[int] = Field(None, ge=0, description="Average iterations for this pattern")
    # T031: Tool reliability metrics
    avg_execution_time_ms: Optional[float] = Field(None, ge=0.0, description="Average execution time in milliseconds")


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

    def compare_with_ground_truth(
        self,
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

        # Normalize for comparison
        answer_norm = answer.strip().lower()
        truth_norm = ground_truth.strip().lower()

        is_correct = answer_norm == truth_norm

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

    def analyze_tool_usage(
        self,
        task_output: Any,
        task_successful: bool,
        domain: str
    ) -> List[InsightCandidate]:
        """
        T022/T030: Extract tool usage patterns from ReAct task execution.

        Analyzes structured_trace to identify:
        - Successful tool sequences (T022)
        - Tool failure patterns (T030)
        - Tool adaptations after errors (T030)

        Args:
            task_output: TaskOutput from ReActGenerator (must have structured_trace)
            task_successful: Whether task completed successfully
            domain: Task domain for tagging

        Returns:
            List of InsightCandidate objects with tool metadata (empty if no tools used)
        """
        # Check if this is a ReAct task output (has structured_trace)
        if not hasattr(task_output, 'structured_trace'):
            logger.debug("analyze_tool_usage_skipped", reason="Not a ReAct task output")
            return []

        structured_trace = task_output.structured_trace
        if not structured_trace:
            logger.debug("analyze_tool_usage_skipped", reason="Empty structured trace")
            return []

        insights = []

        # Extract tool sequence and identify failures/adaptations
        tool_sequence = []
        tool_steps = []
        failed_tools = []  # T030: Track failed tool attempts
        adaptations = []  # T030: Track tool switches after errors
        tool_execution_times = []  # T031: Track execution durations

        for i, step in enumerate(structured_trace):
            if step.action == "call_tool" and step.tool_name:
                tool_sequence.append(step.tool_name)
                tool_steps.append(step.iteration)

                # T031: Collect execution time
                duration_ms = getattr(step, 'duration_ms', 0.0)
                if duration_ms > 0:
                    tool_execution_times.append(duration_ms)

                # T030: Check if tool call resulted in error
                observation = getattr(step, 'observation', '')
                if observation and ('error' in observation.lower() or 'failed' in observation.lower()):
                    failed_tools.append(step.tool_name)

                    # Check if next step switches to different tool (adaptation)
                    if i + 1 < len(structured_trace):
                        next_step = structured_trace[i + 1]
                        if (next_step.action == "call_tool" and
                            next_step.tool_name and
                            next_step.tool_name != step.tool_name):
                            adaptations.append({
                                'from_tool': step.tool_name,
                                'to_tool': next_step.tool_name,
                                'iteration': step.iteration
                            })

        if not tool_sequence:
            logger.debug("analyze_tool_usage_skipped", reason="No tools used in trace")
            return []

        # Calculate metadata
        total_iterations = getattr(task_output, 'total_iterations', 0)
        success_rate = 1.0 if task_successful else 0.0

        # T031: Calculate average execution time
        avg_exec_time = sum(tool_execution_times) / len(tool_execution_times) if tool_execution_times else None

        # Create main tool sequence insight
        if len(tool_sequence) == 1:
            content = f"Used '{tool_sequence[0]}' tool to solve the task"
        else:
            tool_chain = " → ".join(tool_sequence)
            content = f"Tool sequence: {tool_chain}"

        # Determine section based on success
        section = InsightSection.HELPFUL if task_successful else InsightSection.HARMFUL

        # Create rationale
        outcome = "successful" if task_successful else "unsuccessful"
        rationale = f"Tool sequence led to {outcome} task completion in {total_iterations} iterations"

        # Create main InsightCandidate with tool metadata
        main_insight = InsightCandidate(
            content=content,
            section=section,
            confidence=0.9 if task_successful else 0.7,
            rationale=rationale,
            tags=[domain, "tool-calling"] if domain else ["tool-calling"],
            referenced_steps=tool_steps,
            tool_sequence=tool_sequence,
            tool_success_rate=success_rate,
            avg_iterations=total_iterations,
            avg_execution_time_ms=avg_exec_time  # T031
        )
        insights.append(main_insight)

        # T030: Create insights for tool failures (Harmful patterns)
        if failed_tools:
            unique_failures = list(set(failed_tools))
            failure_content = f"Tool failures: {', '.join(unique_failures)} caused errors or timeouts"
            failure_insight = InsightCandidate(
                content=failure_content,
                section=InsightSection.HARMFUL,
                confidence=0.8,
                rationale=f"Tools {', '.join(unique_failures)} encountered errors during execution",
                tags=[domain, "tool-calling", "tool-failure"] if domain else ["tool-calling", "tool-failure"],
                referenced_steps=tool_steps,
                tool_sequence=unique_failures,
                tool_success_rate=0.0,
                avg_iterations=total_iterations,
                avg_execution_time_ms=avg_exec_time  # T031
            )
            insights.append(failure_insight)

        # T030: Create insights for tool adaptations (learning pattern)
        if adaptations:
            for adaptation in adaptations:
                adapt_content = f"After '{adaptation['from_tool']}' failed, switched to '{adaptation['to_tool']}'"
                adapt_insight = InsightCandidate(
                    content=adapt_content,
                    section=InsightSection.HELPFUL if task_successful else InsightSection.HARMFUL,
                    confidence=0.75,
                    rationale=f"Tool adaptation at iteration {adaptation['iteration']}",
                    tags=[domain, "tool-calling", "adaptation"] if domain else ["tool-calling", "adaptation"],
                    referenced_steps=[adaptation['iteration']],
                    tool_sequence=[adaptation['from_tool'], adaptation['to_tool']],
                    tool_success_rate=success_rate,
                    avg_iterations=total_iterations,
                    avg_execution_time_ms=avg_exec_time  # T031
                )
                insights.append(adapt_insight)

        logger.info(
            "tool_usage_analyzed",
            task_id=getattr(task_output, 'task_id', 'unknown'),
            tool_sequence=tool_sequence,
            num_tools=len(tool_sequence),
            success=task_successful,
            iterations=total_iterations,
            failures=len(failed_tools),
            adaptations=len(adaptations)
        )

        return insights

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
            prediction = protected_predict(
                self.predictor,
                circuit_name="reflector",
                failure_threshold=5,
                recovery_timeout=60,
                task_context=task_context,
                feedback_context=feedback_context,
                domain=reflector_input.domain
            )

            # Parse insights from analysis
            insights = self.parse_insights_from_analysis(
                prediction.helpful_insights,
                prediction.harmful_insights,
                float(prediction.confidence),
                reflector_input.domain
            )

            # T022: Analyze tool usage for ReAct tasks
            if reflector_input.structured_trace:
                # Determine task success from feedback signals
                task_successful = is_correct or tests_passed

                # Create a mock task output object with ReAct fields
                class TaskOutputMock:
                    def __init__(self):
                        self.task_id = reflector_input.task_id
                        self.structured_trace = reflector_input.structured_trace
                        self.tools_used = reflector_input.tools_used
                        self.total_iterations = reflector_input.total_iterations
                        self.iteration_limit_reached = reflector_input.iteration_limit_reached

                mock_output = TaskOutputMock()
                tool_insights = self.analyze_tool_usage(
                    mock_output,
                    task_successful,
                    reflector_input.domain
                )

                # Add tool insights to overall insights
                insights.extend(tool_insights)

                logger.debug(
                    "tool_insights_added",
                    task_id=reflector_input.task_id,
                    num_tool_insights=len(tool_insights)
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
