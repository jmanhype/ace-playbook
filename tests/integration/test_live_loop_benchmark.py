"""Integration tests for the live loop benchmark runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_live_loop_benchmark_dummy_backend(tmp_path: Path) -> None:
    """Benchmark runner should execute end-to-end with the dummy backend."""

    output_path = tmp_path / "benchmark.json"
    script = Path("benchmarks/run_live_loop_benchmark.py")
    result = subprocess.run(
        [sys.executable, str(script), "--backend", "dummy", "--episodes", "4", "--output", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output_path.exists(), result.stdout
    payload = json.loads(output_path.read_text())
    assert payload["backend"] == "dummy"
    assert len(payload["runs"]) == 2
    baseline, ace = payload["runs"]
    assert baseline["run_type"] == "baseline"
    assert ace["run_type"] == "ace"
    # ACE run should record at least as many operations as baseline.
    assert ace["total_operations"] >= baseline["total_operations"]
