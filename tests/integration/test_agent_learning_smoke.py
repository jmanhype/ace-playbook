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


def test_agent_learning_smoke_run():
    world_predictions = itertools.cycle(
        [
            {
                "answer": "4",
                "reasoning": ["Add numbers"],
                "confidence": 0.9,
                "raw_response": {"call": 1},
            },
            {
                "answer": "5",
                "reasoning": ["Arithmetic slip"],
                "confidence": 0.3,
                "raw_response": {"call": 2},
            },
        ]
    )
    insight_cycle = itertools.cycle(
        [
            {
                "insights": [
                    {
                        "content": "Double-check arithmetic when confidence is low",
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
    assert len(results) == 2
    assert results[0].metrics["accuracy"] == 1.0
    assert not results[0].curator_operations
    assert results[1].metrics["accuracy"] == 0.0
    assert results[1].curator_operations
