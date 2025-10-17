"""
Chain-of-Thought Generator Implementation

Implements CoTGenerator using DSPy ChainOfThought for sequential reasoning tasks.
Injects playbook bullets into context and produces structured reasoning traces.

Based on contracts/generator.py and tasks.md T039-T041.
"""

import dspy
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ace.generator.signatures import TaskInput, ChainOfThoughtSignature
from ace.utils.logging_config import get_logger
from ace.utils.llm_circuit_breaker import protected_predict
from pydantic import BaseModel, Field

logger = get_logger(__name__, component="generator")


class TaskOutput(BaseModel):
    """
    Task execution output with reasoning trace and metadata.

    Pydantic model (not DSPy Signature) to support optional fields
    that DSPy doesn't handle well.

    Backward Compatibility (T052):
    - Core fields (task_id, reasoning_trace, answer, confidence) are required
    - ReAct-specific fields (structured_trace, tools_used, etc.) are optional
    - CoTGenerator leaves ReAct fields empty/default, ReActGenerator populates them
    """
    task_id: str = Field(..., description="Task identifier")
    reasoning_trace: List[str] = Field(..., description="Step-by-step reasoning")
    answer: str = Field(..., description="Final answer")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    bullets_referenced: List[str] = Field(default_factory=list, description="Referenced bullet IDs")
    latency_ms: Optional[int] = Field(None, description="Execution latency")
    model_name: Optional[str] = Field(None, description="LLM model used")
    prompt_tokens: Optional[int] = Field(None, description="Input tokens")
    completion_tokens: Optional[int] = Field(None, description="Output tokens")

    # ReAct-specific optional fields (T052 - backward compatibility)
    structured_trace: List[Any] = Field(default_factory=list, description="Structured reasoning steps with metadata")
    tools_used: List[str] = Field(default_factory=list, description="Tools used during execution")
    total_iterations: int = Field(default=0, description="Number of ReAct iterations")
    iteration_limit_reached: bool = Field(default=False, description="Whether max iterations was hit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CoTGenerator:
    """
    Chain-of-Thought Generator using DSPy for pure reasoning tasks.

    Suitable for arithmetic, logic puzzles, QA tasks without tool use.
    Produces structured reasoning traces with explicit playbook references.

    Implementation details:
    - T039: Uses dspy.ChainOfThought predictor
    - T040: Formats reasoning traces into List[str] with step labels
    - T041: Injects top-K playbook bullets into prompt context
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize CoTGenerator.

        Args:
            model: LLM model name (gpt-4-turbo, gpt-3.5-turbo, etc.)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens for generation
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize DSPy ChainOfThought predictor
        self.predictor = dspy.ChainOfThought(ChainOfThoughtSignature)

        logger.info(
            "cot_generator_initialized",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def format_playbook_context(self, bullets: List[str], bullet_ids: Optional[List[str]] = None) -> str:
        """
        Format playbook bullets as numbered list with IDs.

        T041: Inject top-K bullets into Generator prompt context.

        Args:
            bullets: List of strategy text content
            bullet_ids: Optional list of bullet IDs (if None, generates placeholder IDs)

        Returns:
            Formatted string with numbered bullets and IDs

        Example:
            "Strategy 1 [bullet-abc123]: Break problems into smaller steps
             Strategy 2 [bullet-def456]: Verify intermediate results"
        """
        if not bullets:
            return "No playbook strategies available."

        # Generate placeholder IDs if not provided
        if bullet_ids is None:
            bullet_ids = [f"bullet-{i:03d}" for i in range(len(bullets))]

        formatted_lines = []
        for idx, (bullet, bullet_id) in enumerate(zip(bullets, bullet_ids), start=1):
            formatted_lines.append(f"Strategy {idx} [{bullet_id}]: {bullet}")

        context = "\n".join(formatted_lines)

        logger.debug(
            "playbook_context_formatted",
            num_bullets=len(bullets),
            context_length=len(context)
        )

        return context

    def parse_reasoning_trace(self, reasoning_text: str, max_steps: int = 20) -> List[str]:
        """
        Parse LLM reasoning output into structured List[str] trace.

        T040: Extract reasoning steps and ensure 1-20 step limit.

        Args:
            reasoning_text: Raw reasoning text from LLM
            max_steps: Maximum number of steps (default: 20)

        Returns:
            List of reasoning step strings

        Example:
            Input: "Step 1: Identify the problem\\nStep 2: Break into parts\\n..."
            Output: ["Step 1: Identify the problem", "Step 2: Break into parts", ...]
        """
        # Split by common delimiters (newlines, numbered steps)
        lines = reasoning_text.strip().split("\n")

        # Filter out empty lines and clean whitespace
        steps = [line.strip() for line in lines if line.strip()]

        # Enforce max_steps limit
        if len(steps) > max_steps:
            logger.warning(
                "reasoning_trace_truncated",
                original_steps=len(steps),
                truncated_to=max_steps
            )
            steps = steps[:max_steps]

        # Ensure at least 1 step
        if not steps:
            steps = ["No explicit reasoning steps provided"]

        logger.debug(
            "reasoning_trace_parsed",
            num_steps=len(steps)
        )

        return steps

    def extract_bullet_references(self, reasoning_text: str) -> List[str]:
        """
        Extract playbook bullet IDs referenced in reasoning trace.

        T040: Detect strategy references and populate bullets_referenced list.

        Args:
            reasoning_text: Full reasoning text

        Returns:
            List of bullet IDs mentioned in reasoning

        Example:
            Input: "Using strategy [bullet-abc123], I will break this down..."
            Output: ["bullet-abc123"]
        """
        # Regex pattern to match bullet IDs in brackets: [bullet-xxx] or [bullet-xxx-xxx]
        pattern = r'\[bullet-[a-zA-Z0-9\-]+\]'
        matches = re.findall(pattern, reasoning_text)

        # Extract IDs without brackets
        bullet_ids = [match.strip('[]') for match in matches]

        # Remove duplicates while preserving order
        unique_ids = []
        seen = set()
        for bullet_id in bullet_ids:
            if bullet_id not in seen:
                unique_ids.append(bullet_id)
                seen.add(bullet_id)

        logger.debug(
            "bullet_references_extracted",
            num_references=len(unique_ids),
            bullet_ids=unique_ids
        )

        return unique_ids

    def validate_task_input(self, task_input: TaskInput) -> None:
        """
        Validate TaskInput before execution.

        Args:
            task_input: TaskInput object to validate

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not task_input.task_id:
            raise ValueError("task_id is required")

        if not task_input.description:
            raise ValueError("task description is required")

        if task_input.max_reasoning_steps < 1 or task_input.max_reasoning_steps > 20:
            raise ValueError("max_reasoning_steps must be between 1 and 20")

        logger.debug(
            "task_input_validated",
            task_id=task_input.task_id,
            domain=task_input.domain,
            num_bullets=len(task_input.playbook_bullets)
        )

    def __call__(self, task_input: TaskInput) -> TaskOutput:
        """
        Execute task with CoT reasoning and playbook context.

        Args:
            task_input: TaskInput with task description and playbook bullets

        Returns:
            TaskOutput with reasoning trace, answer, and metadata

        Raises:
            ValueError: If task_input is invalid
            RuntimeError: If generation fails
        """
        start_time = time.time()

        # Validate input
        self.validate_task_input(task_input)

        logger.info(
            "cot_generation_start",
            task_id=task_input.task_id,
            domain=task_input.domain,
            num_bullets=len(task_input.playbook_bullets)
        )

        try:
            # T041: Format playbook bullets into context
            playbook_context = self.format_playbook_context(task_input.playbook_bullets)

            # T071: Call DSPy ChainOfThought predictor with circuit breaker protection
            prediction = protected_predict(
                self.predictor,
                circuit_name="generator",
                failure_threshold=5,
                recovery_timeout=60,
                task_description=task_input.description,
                playbook_context=playbook_context,
                domain=task_input.domain
            )

            # T040: Parse reasoning into structured trace
            reasoning_trace = self.parse_reasoning_trace(
                prediction.reasoning,
                max_steps=task_input.max_reasoning_steps
            )

            # T040: Extract bullet references
            bullets_referenced = self.extract_bullet_references(prediction.reasoning)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Construct TaskOutput
            output = TaskOutput(
                task_id=task_input.task_id,
                reasoning_trace=reasoning_trace,
                answer=prediction.answer,
                confidence=float(prediction.confidence),
                bullets_referenced=bullets_referenced,
                latency_ms=latency_ms,
                model_name=self.model,
                prompt_tokens=None,  # Populated by runner if available
                completion_tokens=None
            )

            logger.info(
                "cot_generation_complete",
                task_id=task_input.task_id,
                num_steps=len(reasoning_trace),
                num_bullets_referenced=len(bullets_referenced),
                latency_ms=latency_ms,
                confidence=output.confidence
            )

            return output

        except Exception as e:
            logger.error(
                "cot_generation_failed",
                task_id=task_input.task_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise RuntimeError(f"CoT generation failed: {str(e)}") from e


def create_cot_generator(
    model: str = "gpt-4-turbo",
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> CoTGenerator:
    """
    Factory function to create CoTGenerator instance.

    Args:
        model: LLM model name
        temperature: Sampling temperature
        max_tokens: Maximum generation tokens

    Returns:
        Initialized CoTGenerator instance
    """
    return CoTGenerator(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )


__version__ = "v1.0.0"
