"""Minimal ACE + Agent Learning demo loop using configurable LLM clients."""

from __future__ import annotations

import argparse
import itertools
from typing import Literal

from ace.agent_learning import (
    LiveLoop,
    PredictionGuardrails,
    ReflectionEngine,
    RuntimeReflector,
    StaticTaskDataset,
    TaskSpecification,
    WorldModel,
    create_in_memory_runtime,
)
from ace.agent_learning.policy import EpsilonGreedyPolicy
from ace.agent_learning.utils import prepare_default_metrics
from ace.llm_client import DSPyLLMClient, DummyLLMClient


BackendChoice = Literal["dummy", "dspy"]


def build_dummy_client() -> DummyLLMClient:
    """Return a dummy client producing deterministic responses."""

    world_predictions = itertools.cycle(
        [
            {
                "answer": "4",
                "reasoning": ["Add the integers"],
                "confidence": 0.9,
                "raw_response": {"episode": 1},
            },
            {
                "answer": "5",
                "reasoning": ["Forgot to carry the one"],
                "confidence": 0.2,
                "raw_response": {"episode": 2},
            },
        ]
    )
    insight_cycle = itertools.cycle(
        [
            {
                "insights": [
                    {
                        "content": "Double-check arithmetic against ground truth",
                        "section": "Helpful",
                        "tags": ["math"],
                    }
                ]
            }
        ]
    )

    client = DummyLLMClient()
    client.register("WorldModelPrediction", lambda: next(world_predictions))
    client.register("ReflectionResponse", lambda: next(insight_cycle))
    return client


def build_llm_client(backend: BackendChoice) -> DSPyLLMClient | DummyLLMClient:
    if backend == "dummy":
        return build_dummy_client()
    if backend == "dspy":
        return DSPyLLMClient()
    raise ValueError(f"Unsupported backend '{backend}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ACE + Agent Learning quick-start demo.")
    parser.add_argument(
        "--backend",
        choices=("dummy", "dspy"),
        default="dummy",
        help="LLM backend to use. 'dummy' uses canned responses; 'dspy' calls the configured dspy.LM.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of live-loop episodes to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model identifier passed to the world model / reflection engine.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the world model when using a real backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = build_llm_client(args.backend)

    world_model = WorldModel(client, model=args.model, temperature=args.temperature)
    reflection_engine = ReflectionEngine(client, model=args.model)
    reflector = RuntimeReflector(reflection_engine)
    runtime_client = create_in_memory_runtime(reflector=reflector)
    policy = EpsilonGreedyPolicy(epsilon=0.0, threshold=0.6)
    metric_registry = prepare_default_metrics()

    dataset = StaticTaskDataset(
        tasks=[
            TaskSpecification(
                task_id="math-1",
                domain_id="demo",
                description="What is 2 + 2?",
                ground_truth="4",
            ),
            TaskSpecification(
                task_id="math-2",
                domain_id="demo",
                description="What is 3 + 3?",
                ground_truth="6",
            ),
        ]
    )

    loop = LiveLoop(
        runtime_client=runtime_client,
        world_model=world_model,
        policy=policy,
        metric_registry=metric_registry,
        tasks=dataset.tasks,
        guardrails=PredictionGuardrails(),
        reflector=reflector,
    )

    results = loop.run(episodes=args.episodes)
    for episode in results:
        accuracy = episode.metrics.get("accuracy", 0.0)
        print(
            f"Episode {episode.task_id}: accuracy={accuracy:.2f} "
            f"operations={len(episode.curator_operations)}"
        )


if __name__ == "__main__":
    main()
