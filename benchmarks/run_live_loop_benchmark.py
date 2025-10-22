"""Benchmark runner comparing ACE-enabled and baseline live loops."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, List, Literal

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from ace.agent_learning.policy import BasePolicy, EpsilonGreedyPolicy
from ace.agent_learning.types import EpisodeResult
from ace.agent_learning.utils import prepare_default_metrics
from ace.llm_client import DSPyLLMClient, DummyLLMClient

BackendChoice = Literal["dummy", "dspy"]


@dataclass
class RunSummary:
    run_type: str
    episodes: int
    mean_accuracy: float
    total_operations: int
    helpful_operations: int
    timestamp_utc: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_tasks(path: Path) -> List[TaskSpecification]:
    data = json.loads(path.read_text())
    tasks = []
    for entry in data:
        tasks.append(
            TaskSpecification(
                task_id=entry["task_id"],
                domain_id=entry["domain_id"],
                description=entry["description"],
                ground_truth=entry.get("ground_truth"),
            )
        )
    return tasks


def build_dummy_client() -> DummyLLMClient:
    """Return a dummy client that alternates between correct and incorrect answers."""

    responses = {
        "WorldModelPrediction": [
            {
                "answer": "4",
                "reasoning": ["Add two numbers."],
                "confidence": 0.9,
                "raw_response": {},
            },
            {
                "answer": "10",
                "reasoning": ["Incorrect addition to trigger updates."],
                "confidence": 0.3,
                "raw_response": {},
            },
        ],
        "ReflectionResponse": {
            "insights": [
                {
                    "content": "Verify arithmetic against ground truth before finalizing.",
                    "section": "Helpful",
                    "tags": ["math"],
                }
            ]
        },
    }

    client = DummyLLMClient()
    for key, items in responses.items():
        if isinstance(items, list):
            iterator = iter(items)

            def make_iter(it: Iterable[dict[str, Any]]):
                cycle = list(it)
                while True:
                    for item in cycle:
                        yield item

            cyclic = make_iter(items)
            client.register(key, lambda cyc=cyclic: next(cyc))
        else:
            client.register(key, items)
    return client


def build_llm_client(backend: BackendChoice) -> DSPyLLMClient | DummyLLMClient:
    if backend == "dummy":
        return build_dummy_client()
    if backend == "dspy":
        return DSPyLLMClient()
    raise ValueError(f"Unsupported backend '{backend}'")


class BaselinePolicy(BasePolicy):
    """Wrapper policy that never triggers curator updates."""

    def should_update(self, **_: Any) -> bool:
        return False


def summarise_results(run_type: str, episodes: List[EpisodeResult]) -> RunSummary:
    accuracies = [episode.metrics.get("accuracy", 0.0) for episode in episodes]
    operations = [len(episode.curator_operations) for episode in episodes]
    helpful = [
        sum(1 for op in episode.curator_operations if op.section.lower() == "helpful")
        for episode in episodes
    ]
    ts = datetime.now(tz=timezone.utc).isoformat()
    return RunSummary(
        run_type=run_type,
        episodes=len(episodes),
        mean_accuracy=mean(accuracies) if accuracies else 0.0,
        total_operations=sum(operations),
        helpful_operations=sum(helpful),
        timestamp_utc=ts,
    )


def run_live_loop(
    *,
    backend: BackendChoice,
    episodes: int,
    use_ace: bool,
    tasks: List[TaskSpecification],
    model: str | None,
    temperature: float,
) -> List[EpisodeResult]:
    client = build_llm_client(backend)
    world_model = WorldModel(client, model=model, temperature=temperature)
    metric_registry = prepare_default_metrics()

    reflector = None
    guardrails = PredictionGuardrails()
    if use_ace:
        reflection_engine = ReflectionEngine(client, model=model)
        reflector = RuntimeReflector(reflection_engine)
    runtime_client = create_in_memory_runtime(reflector=reflector)

    policy: BasePolicy
    if use_ace:
        policy = EpsilonGreedyPolicy(epsilon=0.0, threshold=0.6)
    else:
        policy = BaselinePolicy()

    loop = LiveLoop(
        runtime_client=runtime_client,
        world_model=world_model,
        policy=policy,
        metric_registry=metric_registry,
        tasks=tasks,
        guardrails=guardrails,
        reflector=reflector,
    )
    return loop.run(episodes=episodes)


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACE vs baseline live loop benchmark.")
    parser.add_argument(
        "--backend",
        choices=("dummy", "dspy"),
        default="dummy",
        help="LLM backend to use for both runs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Number of episodes per run.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/data/sample_tasks.json"),
        help="Path to dataset describing benchmark tasks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model identifier passed through when using the DSPy backend.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the world model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark/live_loop_benchmark.json"),
        help="File path to save benchmark summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.dataset)
    dataset = StaticTaskDataset(tasks=tasks)

    baseline_results = run_live_loop(
        backend=args.backend,
        episodes=args.episodes,
        use_ace=False,
        tasks=dataset.tasks,
        model=args.model,
        temperature=args.temperature,
    )
    ace_results = run_live_loop(
        backend=args.backend,
        episodes=args.episodes,
        use_ace=True,
        tasks=dataset.tasks,
        model=args.model,
        temperature=args.temperature,
    )

    baseline_summary = summarise_results("baseline", baseline_results)
    ace_summary = summarise_results("ace", ace_results)

    payload = {
        "backend": args.backend,
        "episodes_per_run": args.episodes,
        "dataset": str(args.dataset),
        "model": args.model,
        "temperature": args.temperature,
        "runs": [
            baseline_summary.as_dict(),
            ace_summary.as_dict(),
        ],
    }
    write_output(args.output, payload)
    print(f"Benchmark results saved to {args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
