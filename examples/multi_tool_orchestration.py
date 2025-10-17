"""
Multi-Tool Workflow Orchestration Example

Demonstrates:
- ReActGenerator with API, calculator, and formatter tools
- Tool failure handling and recovery
- Tool adaptation patterns (switching tools after errors)
- Learning from failures to avoid problematic tool sequences

Based on User Story 2 (US2) from 001-tool-calling-agent spec.

Usage:
    # Mock mode (no LLM required)
    python examples/multi_tool_orchestration.py --mock

    # Real mode (requires OpenAI API key)
    export OPENAI_API_KEY=your-api-key
    python examples/multi_tool_orchestration.py

Expected Output:
    - Demonstration of tool failures and adaptations
    - Harmful bullets for failed tool patterns
    - Helpful bullets for successful recovery strategies
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import dspy
from ace.generator.react_generator import ReActGenerator
from ace.generator.signatures import TaskInput, ReasoningStep
from ace.reflector.grounded_reflector import GroundedReflector, InsightCandidate, ReflectorOutput
from ace.reflector.signatures import ReflectorInput, InsightSection
from ace.curator.semantic_curator import SemanticCurator
from ace.curator.curator_models import CuratorInput
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="example")


# ============================================================================
# MOCK TOOLS WITH FAILURE MODES
# ============================================================================


def fetch_api_data(endpoint: str, retries: int = 1) -> Dict[str, Any]:
    """
    Fetch data from API endpoint with retry logic.

    Args:
        endpoint: API endpoint URL
        retries: Number of retry attempts

    Returns:
        Dict with API response data

    Failure Mode: Fails with timeout error 30% of the time
    """
    # Simulate API failure
    if random.random() < 0.3:
        raise TimeoutError(f"API timeout for endpoint: {endpoint}")

    # Success
    mock_data = {
        "endpoint": endpoint,
        "data": {"value": random.randint(100, 1000)},
        "status": "success"
    }
    logger.debug("api_fetch_success", endpoint=endpoint)
    return mock_data


def calculate_result(expression: str) -> float:
    """
    Calculate mathematical expression.

    Args:
        expression: Math expression string

    Returns:
        Calculation result

    Failure Mode: Fails with division by zero 20% of the time
    """
    # Simulate calculation error
    if random.random() < 0.2:
        raise ZeroDivisionError(f"Division by zero in expression: {expression}")

    # Mock calculation
    result = random.uniform(10.0, 100.0)
    logger.debug("calculation_success", expression=expression, result=result)
    return result


def format_output(data: Any, format_type: str = "json") -> str:
    """
    Format data for output.

    Args:
        data: Data to format
        format_type: Output format (json, csv, xml)

    Returns:
        Formatted string

    Failure Mode: Very reliable, rarely fails
    """
    # Simulate formatting error (rare)
    if random.random() < 0.05:
        raise ValueError(f"Unsupported format type: {format_type}")

    # Success
    formatted = f"[{format_type.upper()}] {str(data)}"
    logger.debug("format_success", format_type=format_type)
    return formatted


def backup_calculator(value: float) -> float:
    """
    Backup calculation method (more reliable).

    Args:
        value: Input value

    Returns:
        Processed value

    Failure Mode: Highly reliable, 5% failure rate
    """
    # Simulate rare failure
    if random.random() < 0.05:
        raise RuntimeError("Backup calculator unavailable")

    # Success
    result = value * 1.1  # Add 10% markup
    logger.debug("backup_calc_success", value=value, result=result)
    return result


def backup_api_fetch(endpoint: str) -> Dict[str, Any]:
    """
    Backup API fetch method (more reliable).

    Args:
        endpoint: API endpoint

    Returns:
        API response data

    Failure Mode: Very reliable, 10% failure rate
    """
    # Simulate rare failure
    if random.random() < 0.1:
        raise ConnectionError("Backup API unavailable")

    # Success
    mock_data = {
        "endpoint": endpoint,
        "data": {"value": random.randint(50, 500)},
        "status": "success_backup"
    }
    logger.debug("backup_api_success", endpoint=endpoint)
    return mock_data


# ============================================================================
# MOCK REFLECTOR WITH TOOL FAILURE DETECTION
# ============================================================================


class MockReflectorWithFailures:
    """Mock reflector that simulates tool failure detection."""

    def __call__(self, reflector_input: ReflectorInput):
        """Return mock insights with tool failure patterns."""
        insights = []

        # Simulate successful tool sequence
        if not reflector_input.iteration_limit_reached and reflector_input.total_iterations < 10:
            # Success pattern
            if reflector_input.tools_used:
                tool_chain = " → ".join(reflector_input.tools_used)
                insights.append(
                    InsightCandidate(
                        content=f"Successful workflow: {tool_chain}",
                        section=InsightSection.HELPFUL,
                        confidence=0.9,
                        rationale="Tool sequence completed task successfully",
                        tags=["multi-tool", "tool-calling"],
                        referenced_steps=[0, 1],
                        tool_sequence=reflector_input.tools_used,
                        tool_success_rate=1.0,
                        avg_iterations=reflector_input.total_iterations,
                        avg_execution_time_ms=150.5,
                    )
                )

            # Simulate adaptation pattern (30% chance)
            if random.random() < 0.3 and len(reflector_input.tools_used) > 1:
                insights.append(
                    InsightCandidate(
                        content=f"After '{reflector_input.tools_used[0]}' failed, switched to '{reflector_input.tools_used[1]}'",
                        section=InsightSection.HELPFUL,
                        confidence=0.8,
                        rationale="Tool adaptation led to success",
                        tags=["multi-tool", "adaptation"],
                        referenced_steps=[0, 1],
                        tool_sequence=reflector_input.tools_used[:2],
                        tool_success_rate=1.0,
                        avg_iterations=reflector_input.total_iterations,
                        avg_execution_time_ms=200.0,
                    )
                )

        else:
            # Failure pattern
            if reflector_input.tools_used:
                insights.append(
                    InsightCandidate(
                        content=f"Tool failures: {', '.join(reflector_input.tools_used)} caused timeouts or errors",
                        section=InsightSection.HARMFUL,
                        confidence=0.85,
                        rationale="Tools encountered multiple errors",
                        tags=["multi-tool", "tool-failure"],
                        referenced_steps=[0, 1],
                        tool_sequence=reflector_input.tools_used,
                        tool_success_rate=0.0,
                        avg_iterations=reflector_input.total_iterations,
                        avg_execution_time_ms=500.0,
                    )
                )

        return ReflectorOutput(
            task_id=reflector_input.task_id,
            insights=insights,
            analysis_summary="Mock analysis with tool failure detection",
            referenced_steps=list(range(len(reflector_input.reasoning_trace))),
            confidence_score=0.85,
            feedback_types_used=[],
            requires_human_review=False,
            contradicts_existing=[],
        )


# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================


def create_workflow_tasks(num_tasks: int) -> List[TaskInput]:
    """Create workflow tasks that exercise different tool combinations."""
    workflows = [
        "Fetch data from API and calculate the result",
        "Calculate values and format as JSON",
        "Fetch API data and format output",
        "Calculate result with backup if primary fails",
        "Fetch data with retry logic and backup",
    ]

    tasks = []
    for i in range(num_tasks):
        workflow = workflows[i % len(workflows)]
        task = TaskInput(
            task_id=f"workflow-{i:03d}",
            description=workflow,
            domain="multi-tool-orchestration",
            playbook_bullets=[],
        )
        tasks.append(task)

    return tasks


def run_multi_tool_workflow(use_mock: bool = True):
    """
    Demonstrate multi-tool orchestration with failure handling.

    Args:
        use_mock: Use mock reflector (no LLM) if True
    """
    print("=" * 80)
    print("MULTI-TOOL WORKFLOW ORCHESTRATION - ERROR HANDLING DEMO")
    print(f"Mode: {'MOCK' if use_mock else 'REAL (with LLM)'}")
    print("=" * 80)
    print()

    # Configure DSPy if not mock mode
    if not use_mock:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Use --mock flag for demonstration.")
            sys.exit(1)
        lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
        dspy.configure(lm=lm)

    # Initialize components
    print("Initializing ACE components...")
    agent = ReActGenerator(
        tools=[
            fetch_api_data,
            calculate_result,
            format_output,
            backup_calculator,
            backup_api_fetch,
        ],
        model="gpt-4",
        max_iters=10
    )
    reflector = MockReflectorWithFailures() if use_mock else GroundedReflector()
    curator = SemanticCurator()

    # Create tasks
    tasks = create_workflow_tasks(15)
    current_playbook: List[PlaybookBullet] = []
    domain_id = "multi-tool-orchestration"

    # Track statistics
    total_failures = 0
    total_adaptations = 0
    total_successes = 0

    print("\n" + "=" * 80)
    print("EXECUTING WORKFLOWS WITH TOOL FAILURE SIMULATION")
    print("=" * 80)

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/15] {task.description}")

        # Simulate task execution
        tools_used = []
        total_iterations = random.randint(2, 8)
        iteration_limit_reached = total_iterations > 8

        # Simulate tool selection and potential failures
        if "API" in task.description.upper():
            if random.random() < 0.3:  # 30% primary API failure
                tools_used = ["fetch_api_data", "backup_api_fetch"]  # Adaptation
                total_adaptations += 1
                print(f"  → Tools: {' → '.join(tools_used)} (adapted after failure)")
            else:
                tools_used = ["fetch_api_data"]
                total_successes += 1
                print(f"  → Tools: {', '.join(tools_used)} (success)")

        elif "calculate" in task.description.lower():
            if random.random() < 0.2:  # 20% calculator failure
                tools_used = ["calculate_result", "backup_calculator"]  # Adaptation
                total_adaptations += 1
                print(f"  → Tools: {' → '.join(tools_used)} (adapted after failure)")
            else:
                tools_used = ["calculate_result"]
                total_successes += 1
                print(f"  → Tools: {', '.join(tools_used)} (success)")

        else:
            # Multi-tool workflow
            tools_used = ["fetch_api_data", "calculate_result", "format_output"]
            if random.random() < 0.1:  # 10% failure
                total_failures += 1
                iteration_limit_reached = True
                print(f"  → Tools: {', '.join(tools_used)} (FAILED - hit iteration limit)")
            else:
                total_successes += 1
                print(f"  → Tools: {', '.join(tools_used)} (success)")

        print(f"  → Iterations: {total_iterations}")

        # Reflection
        reflector_input = ReflectorInput(
            task_id=task.task_id,
            reasoning_trace=[f"Step {i+1}" for i in range(3)],
            answer="Mock workflow result",
            confidence=0.8 if not iteration_limit_reached else 0.5,
            bullets_referenced=[],
            ground_truth="",
            test_results="",
            error_messages=[],
            performance_metrics="",
            domain=task.domain,
            structured_trace=[],
            tools_used=tools_used,
            total_iterations=total_iterations,
            iteration_limit_reached=iteration_limit_reached,
        )

        reflector_output = reflector(reflector_input)
        print(f"  → Insights: {len(reflector_output.insights)}")

        # Curator merging
        if reflector_output.insights:
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

            curator_input = CuratorInput(
                task_id=task.task_id,
                domain_id=domain_id,
                insights=insights_dicts,
                current_playbook=current_playbook,
                target_stage=PlaybookStage.SHADOW,
            )

            curator_output = curator.apply_delta(curator_input)
            current_playbook = curator_output.updated_playbook

    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal workflows: 15")
    print(f"  - Successes: {total_successes}")
    print(f"  - Failures: {total_failures}")
    print(f"  - Adaptations: {total_adaptations}")
    print(f"\nPlaybook size: {len(current_playbook)} bullets")

    helpful_bullets = [b for b in current_playbook if b.section == "Helpful"]
    harmful_bullets = [b for b in current_playbook if b.section == "Harmful"]

    print(f"  - Helpful: {len(helpful_bullets)}")
    print(f"  - Harmful: {len(harmful_bullets)}")

    # Show learned patterns
    print("\nLearned Patterns:")
    print("\nSuccessful Strategies:")
    for i, bullet in enumerate(helpful_bullets[:3], 1):
        tools = f" (Tools: {', '.join(bullet.tool_sequence)})" if bullet.tool_sequence else ""
        print(f"  {i}. {bullet.content}{tools}")

    if harmful_bullets:
        print("\nFailed Patterns to Avoid:")
        for i, bullet in enumerate(harmful_bullets[:3], 1):
            tools = f" (Tools: {', '.join(bullet.tool_sequence)})" if bullet.tool_sequence else ""
            print(f"  {i}. {bullet.content}{tools}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Tool Workflow Orchestration Demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock reflector (no LLM required)",
    )
    args = parser.parse_args()

    run_multi_tool_workflow(use_mock=args.mock or not os.environ.get("OPENAI_API_KEY"))
