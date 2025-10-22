"""Minimal ACE + Agent Learning demo loop using the dummy LLM client."""

from __future__ import annotations

import itertools

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
from ace.llm_client import DummyLLMClient


def main() -> None:
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

    world_model = WorldModel(client)
    reflection_engine = ReflectionEngine(client)
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

    results = loop.run(episodes=2)
    for episode in results:
        accuracy = episode.metrics.get("accuracy", 0.0)
        print(
            f"Episode {episode.task_id}: accuracy={accuracy:.2f} "
            f"operations={len(episode.curator_operations)}"
        )


if __name__ == "__main__":
    main()
