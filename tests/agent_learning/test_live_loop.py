import itertools

from ace.agent_learning import (
    ExperienceBuffer,
    LiveLoop,
    PredictionGuardrails,
    ReflectionEngine,
    RuntimeReflector,
    TaskSpecification,
    WorldModel,
    create_in_memory_runtime,
)
from ace.agent_learning.policy import EpsilonGreedyPolicy
from ace.agent_learning.types import EpisodeResult
from ace.agent_learning.utils import prepare_default_metrics
from ace.llm_client import DummyLLMClient


def test_live_loop_executes_and_records_experience():
    dummy = DummyLLMClient(
        responses={
            "WorldModelPrediction": {
                "answer": "4",
                "reasoning": ["Add the integers"],
                "confidence": 0.9,
                "raw_response": {"source": "dummy"},
            },
            "ReflectionResponse": {
                "insights": [
                    {
                        "content": "Arithmetic tasks benefit from explicit step-by-step addition",
                        "section": "Helpful",
                        "tags": ["math"],
                    }
                ]
            },
        }
    )
    world_model = WorldModel(dummy)
    reflection_engine = ReflectionEngine(dummy)
    reflector = RuntimeReflector(reflection_engine)
    runtime_client = create_in_memory_runtime(reflector=reflector)
    policy = EpsilonGreedyPolicy(epsilon=1.0)
    metric_registry = prepare_default_metrics()
    buffer = ExperienceBuffer(maxlen=5)

    tasks = [
        TaskSpecification(
            task_id="task-1",
            domain_id="demo",
            description="What is 2 + 2?",
            ground_truth="4",
        )
    ]

    loop = LiveLoop(
        runtime_client=runtime_client,
        world_model=world_model,
        policy=policy,
        metric_registry=metric_registry,
        tasks=tasks,
        experience_buffer=buffer,
        guardrails=PredictionGuardrails(),
        reflector=reflector,
    )

    results = loop.run(episodes=1)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EpisodeResult)
    assert buffer.to_dict()[0]["task_id"] == "task-1"
    assert result.metrics["accuracy"] == 1.0
    assert result.curator_operations, "Expected curator operations to be recorded"


def test_policy_skips_high_confidence_updates():
    prediction_cycle = itertools.cycle(
        [
            {
                "answer": "4",
                "reasoning": ["Step reasoning"],
                "confidence": 0.9,
                "raw_response": {},
            },
            {
                "answer": "5",
                "reasoning": ["Incorrect reasoning"],
                "confidence": 0.2,
                "raw_response": {},
            },
        ]
    )
    insight_cycle = itertools.cycle(
        [
            {
                "insights": [
                    {
                        "content": "Emphasise verifying arithmetic",
                        "section": "Helpful",
                        "tags": ["math"],
                    }
                ]
            }
        ]
    )

    dummy = DummyLLMClient()
    dummy.register("WorldModelPrediction", lambda: next(prediction_cycle))
    dummy.register("ReflectionResponse", lambda: next(insight_cycle))

    world_model = WorldModel(dummy)
    reflection_engine = ReflectionEngine(dummy)
    reflector = RuntimeReflector(reflection_engine)
    runtime_client = create_in_memory_runtime(reflector=reflector)
    policy = EpsilonGreedyPolicy(epsilon=0.0, threshold=0.5)
    metric_registry = prepare_default_metrics()

    tasks = [
        TaskSpecification(
            task_id="task-1",
            domain_id="demo",
            description="What is 2 + 2?",
            ground_truth="4",
        )
    ]

    loop = LiveLoop(
        runtime_client=runtime_client,
        world_model=world_model,
        policy=policy,
        metric_registry=metric_registry,
        tasks=tasks,
        guardrails=PredictionGuardrails(),
        reflector=reflector,
    )

    results = loop.run(episodes=2)
    assert len(results) == 2
    first, second = results
    assert first.metrics["accuracy"] == 1.0
    assert not first.curator_operations
    assert second.metrics["accuracy"] == 0.0
    assert second.curator_operations
