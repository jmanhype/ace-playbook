#!/usr/bin/env python3
"""Audit tool for agent feedback heuristics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ace.utils.agent_feedback import create_default_manager


def load_tasks(path: Path) -> List[Dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported dataset format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit agent scoring heuristics.")
    parser.add_argument("dataset", type=Path, help="Path to dataset JSONL/JSON")
    parser.add_argument("--sample", type=int, default=20, help="Number of tasks to sample")
    parser.add_argument("--task-id", type=str, help="Evaluate a single task id")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    tasks = load_tasks(args.dataset)
    if args.task_id:
        tasks = [t for t in tasks if t.get("task_id") == args.task_id]
    if args.sample and args.sample < len(tasks):
        tasks = random.sample(tasks, args.sample)

    manager = create_default_manager()
    for task in tasks:
        answer = task.get("ground_truth_sample", "") or "N/A"
        result = manager.evaluate(task, answer)
        print(json.dumps({
            "task_id": task.get("task_id"),
            "description": task.get("description"),
            "heuristic": result.to_dict(),
        }, indent=2))


if __name__ == "__main__":
    main()
