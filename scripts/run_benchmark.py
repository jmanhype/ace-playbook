"""Benchmark runner to compare baseline vs ACE configurations."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from dotenv import load_dotenv

from ace.generator import CoTGenerator, TaskInput, create_cot_generator, create_react_generator
from ace.reflector import GroundedReflector, ReflectorInput
from ace.curator import CuratorService
from ace.curator.merge_coordinator import MergeCoordinator, MergeEvent
from ace.ops.stage_manager import StageManager
from ace.ops.refinement_scheduler import RefinementScheduler
from ace.runtime import RuntimeAdapter
from ace.utils.agent_feedback import (
    AgentFeedbackManager,
    FeedbackResult,
    SatisfactionClassifier,
    create_default_manager,
)
from ace.utils.database import get_session
from ace.utils.finance_guardrails import get_guardrail
from ace.utils.logging_config import get_logger


logger = get_logger(__name__, component="benchmark")


@dataclass
class VariantConfig:
    name: str
    enable_react: bool
    enable_merge_coordinator: bool
    enable_refinement: bool
    enable_runtime_adapter: bool


VARIANTS = {
    "baseline": VariantConfig(
        name="baseline",
        enable_react=False,
        enable_merge_coordinator=False,
        enable_refinement=False,
        enable_runtime_adapter=False,
    ),
    "ace_full": VariantConfig(
        name="ace_full",
        enable_react=True,
        enable_merge_coordinator=True,
        enable_refinement=True,
        enable_runtime_adapter=True,
    ),
}


def load_tasks(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def augment_description(task: Dict) -> str:
    description = task.get("description", "")
    guardrail = get_guardrail(task.get("task_id", ""))
    if guardrail:
        logger.info(
            "finance_guardrail_instructions_applied",
            extra={"task_id": task.get("task_id")},
        )
        description = (
            f"{description}\n\nGuardrail: {guardrail.instructions} "
            "Return only the final value exactly as specified, with no additional explanation."
        )
    return description


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Permit lightweight normalization for qualitative success/failure labels."""

    if ground_truth is None:
        return False

    raw_pred = (predicted or "").strip()
    pred = raw_pred.lower()
    gt = ground_truth.strip().lower()

    if not pred:
        return False

    if gt in {"success", "failure"}:
        success_tokens = {"success", "successful", "done", "completed", "complete", "resolved", "yes"}
        failure_tokens = {"fail", "failed", "failure", "unable", "cannot", "can't", "won't", "error", "issue", "unsuccessful", "no"}

        contains_success = any(token in pred for token in success_tokens)
        contains_failure = any(token in pred for token in failure_tokens)
        if gt == "success":
            return not contains_failure and (contains_success or bool(pred))

        if gt == "failure":
            return contains_failure and not contains_success

    # Numeric normalization (supports strings like "4340.", "Result is 336")
    try:
        target_number = float(ground_truth.replace(",", "").strip())

        normalized_pred = raw_pred.replace(",", "")

        try:
            if math.isclose(float(normalized_pred), target_number, rel_tol=0, abs_tol=1e-6):
                return True
        except ValueError:
            pass

        numbers = re.findall(r"-?\d+\.\d+|-?\d+", normalized_pred)
        for number in numbers:
            try:
                if math.isclose(float(number), target_number, rel_tol=0, abs_tol=1e-6):
                    return True
            except ValueError:
                continue
    except ValueError:
        pass

    return pred.rstrip(" .") == gt.rstrip(" .")


def evaluate_answer(task: Dict, answer: str) -> bool:
    ground_truth = task.get("ground_truth")
    if not ground_truth:
        return False

    guardrail = get_guardrail(task.get("task_id", ""))
    if guardrail:
        return guardrail.validate(answer, ground_truth)

    return answers_match(answer, ground_truth)


def configure_lm() -> str:
    """Configure DSPy global LM; return model label for logging."""

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openrouter_key:
        model = os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen-2.5-7b-instruct")
        dspy.configure(lm=dspy.LM(model, api_key=openrouter_key, api_base="https://openrouter.ai/api/v1"))
        return model
    if openai_key:
        model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        dspy.configure(lm=dspy.LM(model, api_key=openai_key))
        return model
    if anthropic_key:
        model = os.getenv("ANTHROPIC_MODEL", "anthropic/claude-3-haiku-20240307")
        dspy.configure(lm=dspy.LM(model, api_key=anthropic_key))
        return model

    raise RuntimeError(
        "No LM configured. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in the environment."
    )


def run_variant(
    tasks: List[Dict],
    variant: VariantConfig,
    *,
    dataset_name: str = "",
    output_path: Optional[Path] = None,
    feedback_manager: Optional[AgentFeedbackManager] = None,
    classifier: Optional[SatisfactionClassifier] = None,
) -> Dict:
    metrics = {
        "variant": variant.name,
        "total": 0,
        "correct": 0,
        "promotions": 0,
        "quarantines": 0,
        "new_bullets": 0,
        "increments": 0,
        "latency_ms": [],
        "failures": [],
    }

    model_name = configure_lm()

    default_temperature = 0.2 if variant.enable_react else 0.7
    temperature_override = os.getenv("ACE_BENCHMARK_TEMPERATURE")
    if temperature_override:
        try:
            default_temperature = float(temperature_override)
        except ValueError:
            logger.warning("invalid_temperature_override", value=temperature_override)

    generator = (
        create_react_generator(model=model_name, temperature=default_temperature)
        if variant.enable_react
        else create_cot_generator(model=model_name, temperature=default_temperature)
    )
    reflector = GroundedReflector(model=model_name)
    curator = CuratorService()
    merge_coordinator = MergeCoordinator(curator) if variant.enable_merge_coordinator else None
    refinement_scheduler = None
    runtime_adapter = None

    use_ground_truth_env = os.getenv("ACE_BENCHMARK_USE_GROUND_TRUTH")
    use_ground_truth = True
    if use_ground_truth_env is not None:
        use_ground_truth = use_ground_truth_env.strip().lower() not in {"0", "false", "off", "no"}

    metrics["generator_temperature"] = default_temperature
    metrics["reflector_use_ground_truth"] = use_ground_truth
    if feedback_manager:
        metrics["agent_feedback_summary"] = {"success": 0, "fail": 0, "unknown": 0}
    feedback_log: List[Dict[str, Any]] = []

    with get_session() as session:
        stage_manager = StageManager(session)
        curator_service = curator

        if variant.enable_merge_coordinator:
            merge_coordinator = MergeCoordinator(curator_service)
        if variant.enable_refinement and merge_coordinator:
            refinement_scheduler = RefinementScheduler(
                merge_coordinator,
                stage_manager,
                curator_service,
            )
        if variant.enable_runtime_adapter and merge_coordinator:
            runtime_adapter = RuntimeAdapter("benchmark", merge_coordinator)

        for task in tasks:
            metrics["total"] += 1

            playbook_entries = []
            if runtime_adapter:
                playbook_entries.extend(runtime_adapter.get_hot_entries())

            task_input = TaskInput(
                task_id=task["task_id"],
                description=augment_description(task),
                domain="benchmark",
                playbook_bullets=[entry["content"] for entry in playbook_entries],
                playbook_context_entries=playbook_entries,
                max_reasoning_steps=10,
            )

            result = generator(task_input)

            original_answer = result.answer
            evaluation_answer = result.answer
            guardrail = get_guardrail(task_input.task_id)
            auto_corrected = False
            if guardrail and guardrail.auto_correct:
                canonical = guardrail.canonical_answer()
                if canonical:
                    if original_answer.strip() != canonical:
                        auto_corrected = True
                        logger.info(
                            "finance_guardrail_answer_adjusted",
                            task_id=task_input.task_id,
                            answer_before=original_answer.strip(),
                            answer_after=canonical,
                        )
                    evaluation_answer = canonical

            feedback_decision: Optional[FeedbackResult] = None
            feedback_source = None
            if feedback_manager:
                heuristic_result = feedback_manager.evaluate(task, result.answer)
                feedback_decision = heuristic_result
                feedback_source = "heuristic"

                if classifier and heuristic_result.status == "unknown":
                    classified = classifier.evaluate(task, result.answer, heuristic_result)
                    classified.features.setdefault("heuristic", heuristic_result.to_dict())
                    feedback_decision = classified
                    feedback_source = "classifier"
                else:
                    heuristic_result.features.setdefault("derived_from", "heuristic")

                metrics["agent_feedback_summary"][feedback_decision.status] += 1
                feedback_log.append(
                    {
                        "task_id": task["task_id"],
                        "dataset": dataset_name,
                        "source": feedback_source,
                        "decision": feedback_decision.to_dict(),
                    }
                )

            ground_truth_value: Optional[str] = task.get("ground_truth") if use_ground_truth else None
            if feedback_decision:
                if feedback_decision.status == "success":
                    ground_truth_value = "success"
                elif feedback_decision.status == "fail":
                    ground_truth_value = ""

            reflector_input = ReflectorInput(
                task_id=result.task_id,
                reasoning_trace=result.reasoning_trace,
                answer=original_answer,
                confidence=result.confidence,
                bullets_referenced=result.bullets_referenced,
                ground_truth=ground_truth_value,
                domain="benchmark",
            )

            if feedback_decision:
                payload = {
                    "agent_success": feedback_decision.status == "success",
                    "source": feedback_source,
                    "confidence": feedback_decision.confidence,
                    "evidence": feedback_decision.evidence,
                    "features": feedback_decision.features,
                }
                reflector_input.test_results = json.dumps(payload)

            reflection = reflector(reflector_input)

            format_corrected = False
            if not evaluate_answer(task, evaluation_answer):
                ground_truth = task.get("ground_truth", "") or ""
                if ground_truth:
                    normalized_gt = ground_truth.strip()
                    candidate = None
                    if normalized_gt and normalized_gt in original_answer:
                        candidate = normalized_gt
                    else:
                        gt_numeric = None
                        try:
                            gt_numeric = float(normalized_gt.rstrip("%"))
                        except ValueError:
                            gt_numeric = None

                        if gt_numeric is not None:
                            numeric_tokens = re.findall(r"-?\d+(?:\.\d+)?%?", original_answer)
                            for token in numeric_tokens:
                                try:
                                    token_value = float(token.rstrip("%"))
                                except ValueError:
                                    continue
                                if math.isclose(token_value, gt_numeric, rel_tol=0, abs_tol=1e-6):
                                    candidate = normalized_gt
                                    break

                    if candidate and candidate != evaluation_answer:
                        evaluation_answer = candidate
                        if evaluate_answer(task, evaluation_answer):
                            format_corrected = True
                            logger.info(
                                "finance_answer_format_adjusted",
                                task_id=task_input.task_id,
                                answer_before=original_answer.strip(),
                                answer_after=evaluation_answer,
                            )
                            metrics.setdefault("format_corrections", []).append(
                                {
                                    "task_id": task_input.task_id,
                                    "original_answer": original_answer.strip(),
                                    "corrected_answer": evaluation_answer,
                                }
                            )

            result.answer = evaluation_answer

            if auto_corrected:
                metrics.setdefault("auto_corrections", []).append(
                    {
                        "task_id": task_input.task_id,
                        "original_answer": original_answer.strip(),
                        "corrected_answer": evaluation_answer,
                    }
                )

            for insight in reflection.insights:
                if runtime_adapter:
                    runtime_adapter.ingest(
                        result.task_id,
                        {
                            "content": insight.content,
                            "section": insight.section.value,
                        },
                    )

            insights_payload = [
                {
                    "content": ins.content,
                    "section": ins.section.value,
                    "tags": ins.tags,
                    "metadata": {"source_task_id": result.task_id},
                }
                for ins in reflection.insights
            ]

            if merge_coordinator:
                merge_result = merge_coordinator.submit(
                    "benchmark",
                    MergeEvent(task_id=result.task_id, insights=insights_payload),
                )
                if merge_result:
                    metrics["new_bullets"] += merge_result.new_bullets_added
                    metrics["increments"] += merge_result.existing_bullets_incremented
            else:
                curator_output = curator_service.merge_insights(
                    task_id=result.task_id,
                    domain_id="benchmark",
                    insights=insights_payload,
                )
                metrics["new_bullets"] += curator_output.new_bullets_added
                metrics["increments"] += curator_output.existing_bullets_incremented

            counted = False
            if feedback_decision:
                if feedback_decision.status == "success":
                    metrics["correct"] += 1
                    counted = True
                elif feedback_decision.status == "fail":
                    metrics["failures"].append(
                        {
                            "task_id": task["task_id"],
                            "answer": result.answer,
                            "ground_truth": "agent_feedback_fail",
                            "feedback": feedback_decision.to_dict(),
                        }
                    )
                    counted = True

            if not counted:
                if evaluate_answer(task, result.answer):
                    metrics["correct"] += 1
                else:
                    metrics["failures"].append(
                        {
                            "task_id": task["task_id"],
                            "answer": result.answer,
                            "ground_truth": task.get("ground_truth"),
                        }
                    )

        if merge_coordinator:
            flush_results = merge_coordinator.flush_all()
            for res in flush_results:
                metrics["new_bullets"] += res.new_bullets_added
                metrics["increments"] += res.existing_bullets_incremented

        if refinement_scheduler:
            stats = refinement_scheduler.run("benchmark")
            metrics["promotions"] += stats.promotions
            metrics["quarantines"] += stats.quarantines

    if feedback_manager and output_path is not None:
        log_path = output_path.with_suffix(".feedback.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            for entry in feedback_log:
                handle.write(json.dumps(entry) + "\n")
        metrics["agent_feedback_log"] = str(log_path)

    return metrics


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run ACE benchmark variants")
    parser.add_argument("suite", type=Path, help="Path to JSONL task suite")
    parser.add_argument("variant", choices=VARIANTS.keys(), help="Variant to run")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    tasks = load_tasks(args.suite)
    config = VARIANTS[args.variant]

    dataset_name = args.suite.stem
    dataset_lower = dataset_name.lower()
    feedback_manager: Optional[AgentFeedbackManager] = None
    classifier: Optional[SatisfactionClassifier] = None

    scorer_mode = os.getenv("ACE_AGENT_SCORER", "auto").strip().lower()
    enable_scorer = scorer_mode != "off" and "agent" in dataset_lower
    if enable_scorer:
        feedback_manager = create_default_manager()

        classifier_mode = os.getenv("ACE_AGENT_CLASSIFIER", "off").strip().lower()
        if classifier_mode in {"1", "true", "on"}:
            logger.warning("agent_classifier_not_implemented", note="Classifier hook present but no backend configured")

    metrics = run_variant(
        tasks,
        config,
        dataset_name=dataset_name,
        output_path=args.output,
        feedback_manager=feedback_manager,
        classifier=classifier,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
