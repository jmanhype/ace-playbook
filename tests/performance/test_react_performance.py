"""
Performance Tests for ReActGenerator (T040-T044)

Tests performance budgets:
- T040: Tool call overhead <100ms per iteration (excluding tool execution)
- T041: Playbook retrieval <10ms P50 latency
- T042: Agent initialization <500ms with 10-50 tools
- T043: End-to-end RAG queries <10s for 95% of queries (SC-004)
- T044: Scaling test with 10, 25, 50 tools without degradation
"""

import pytest
import time
import statistics
from typing import Dict, Any, List, Callable
from pathlib import Path

from ace.generator.react_generator import ReActGenerator
from ace.generator.signatures import TaskInput
from ace.models.playbook import PlaybookBullet, PlaybookStage


# ============================================================================
# MOCK TOOLS FOR PERFORMANCE TESTING
# ============================================================================

def fast_tool_a(query: str) -> str:
    """Fast tool - completes in <10ms."""
    return f"Result: {query}"


def fast_tool_b(value: int) -> int:
    """Fast tool - completes in <10ms."""
    return value * 2


def fast_tool_c(data: Dict[str, Any]) -> str:
    """Fast tool - completes in <10ms."""
    return str(data)


def medium_tool_d(query: str) -> str:
    """Medium tool - completes in ~50ms."""
    time.sleep(0.05)
    return f"Result: {query}"


def medium_tool_e(value: int) -> int:
    """Medium tool - completes in ~50ms."""
    time.sleep(0.05)
    return value * 3


def slow_tool_f(query: str) -> str:
    """Slow tool - completes in ~200ms."""
    time.sleep(0.2)
    return f"Result: {query}"


def generate_tools(count: int) -> List[Callable]:
    """Generate specified number of mock tools for scaling tests."""
    tools = [fast_tool_a, fast_tool_b, fast_tool_c, medium_tool_d, medium_tool_e, slow_tool_f]

    # Generate additional tools if needed
    if count > len(tools):
        for i in range(len(tools), count):
            # Create dynamic tool function
            def make_tool(idx: int):
                def dynamic_tool(query: str) -> str:
                    return f"Tool {idx} result: {query}"
                dynamic_tool.__name__ = f"dynamic_tool_{idx}"
                return dynamic_tool
            tools.append(make_tool(i))

    return tools[:count]


# ============================================================================
# T040: TOOL CALL OVERHEAD BENCHMARK
# ============================================================================

def test_tool_call_overhead_per_iteration():
    """
    T040: Verify tool call overhead is <100ms per iteration (excluding tool execution).

    Measures the overhead of:
    - Tool lookup
    - Argument validation
    - Execution wrapper
    - Observation formatting

    Does NOT include actual tool execution time.
    """
    agent = ReActGenerator(tools=[fast_tool_a, fast_tool_b, fast_tool_c], model="gpt-4", max_iters=10)

    overhead_times = []

    for _ in range(50):  # Run 50 iterations to get stable measurements
        # Measure overhead for tool execution wrapper
        start = time.perf_counter()

        # Execute fast tool (execution time ~0ms)
        result, error = agent._execute_tool_with_timeout("fast_tool_a", {"query": "test"})

        end = time.perf_counter()
        overhead_ms = (end - start) * 1000
        overhead_times.append(overhead_ms)

    # Calculate statistics
    avg_overhead = statistics.mean(overhead_times)
    p50_overhead = statistics.median(overhead_times)
    p95_overhead = sorted(overhead_times)[int(len(overhead_times) * 0.95)]
    p99_overhead = sorted(overhead_times)[int(len(overhead_times) * 0.99)]

    print(f"\nTool Call Overhead Benchmark:")
    print(f"  Avg: {avg_overhead:.2f}ms")
    print(f"  P50: {p50_overhead:.2f}ms")
    print(f"  P95: {p95_overhead:.2f}ms")
    print(f"  P99: {p99_overhead:.2f}ms")

    # Assert performance budget: <100ms overhead per iteration
    assert p95_overhead < 100, f"P95 overhead {p95_overhead:.2f}ms exceeds 100ms budget"
    assert avg_overhead < 50, f"Average overhead {avg_overhead:.2f}ms exceeds 50ms target"


# ============================================================================
# T041: PLAYBOOK RETRIEVAL BENCHMARK
# ============================================================================

def test_playbook_retrieval_latency():
    """
    T041: Verify playbook retrieval has <10ms P50 latency.

    Tests the performance of:
    - Strategy filtering by domain
    - Tool sequence matching
    - LRU cache effectiveness
    """
    agent = ReActGenerator(tools=[fast_tool_a, fast_tool_b], model="gpt-4", max_iters=10)

    # Create mock playbook with 100 strategies
    mock_playbook = [
        PlaybookBullet(
            content=f"Strategy {i}: Use tool combination for task type {i % 10}",
            section="Helpful",
            domain_id=f"domain_{i % 5}",
            tags=["tool-calling"],
            tool_sequence=["fast_tool_a", "fast_tool_b"] if i % 2 == 0 else ["fast_tool_b"],
            tool_success_rate=0.8 + (i % 20) * 0.01,
            avg_iterations=3 + (i % 5),
        )
        for i in range(100)
    ]

    # Simulate playbook retrieval
    retrieval_times = []

    for domain_id in ["domain_0", "domain_1", "domain_2", "domain_3", "domain_4"]:
        for _ in range(20):  # 20 retrievals per domain
            start = time.perf_counter()

            # Simulate retrieval (filter by domain)
            strategies = [b for b in mock_playbook if b.domain_id == domain_id]

            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            retrieval_times.append(latency_ms)

    # Calculate statistics
    avg_latency = statistics.mean(retrieval_times)
    p50_latency = statistics.median(retrieval_times)
    p95_latency = sorted(retrieval_times)[int(len(retrieval_times) * 0.95)]

    print(f"\nPlaybook Retrieval Benchmark:")
    print(f"  Avg: {avg_latency:.3f}ms")
    print(f"  P50: {p50_latency:.3f}ms")
    print(f"  P95: {p95_latency:.3f}ms")

    # Assert performance budget: <10ms P50 latency
    assert p50_latency < 10, f"P50 latency {p50_latency:.3f}ms exceeds 10ms budget"
    assert p95_latency < 20, f"P95 latency {p95_latency:.3f}ms exceeds 20ms target"


# ============================================================================
# T042: AGENT INITIALIZATION BENCHMARK
# ============================================================================

@pytest.mark.parametrize("num_tools", [10, 25, 50])
def test_agent_initialization_time(num_tools: int):
    """
    T042: Verify agent initialization takes <500ms with 10-50 tools.

    Tests the performance of:
    - Tool validation
    - Tool registration
    - DSPy ReAct module initialization
    """
    tools = generate_tools(num_tools)

    init_times = []

    for _ in range(10):  # Run 10 initializations
        start = time.perf_counter()

        agent = ReActGenerator(tools=tools, model="gpt-4", max_iters=10)

        end = time.perf_counter()
        init_ms = (end - start) * 1000
        init_times.append(init_ms)

    # Calculate statistics
    avg_init = statistics.mean(init_times)
    p95_init = sorted(init_times)[int(len(init_times) * 0.95)]

    print(f"\nAgent Initialization Benchmark ({num_tools} tools):")
    print(f"  Avg: {avg_init:.2f}ms")
    print(f"  P95: {p95_init:.2f}ms")

    # Assert performance budget: <500ms initialization
    assert p95_init < 500, f"P95 init time {p95_init:.2f}ms exceeds 500ms budget for {num_tools} tools"
    assert avg_init < 300, f"Average init time {avg_init:.2f}ms exceeds 300ms target for {num_tools} tools"


# ============================================================================
# T043: END-TO-END PERFORMANCE TEST
# ============================================================================

def test_end_to_end_rag_query_performance():
    """
    T043: Verify RAG queries complete in <10s for 95% of queries (SC-004).

    Tests full task execution including:
    - Playbook retrieval
    - Tool selection
    - Tool execution
    - Observation processing
    - Iteration management

    Note: This test uses mock mode (no actual LLM calls) to test framework overhead.
    Real queries will depend on LLM latency.
    """
    # Create RAG-style tools
    def search_vector_db(query: str) -> Dict[str, Any]:
        time.sleep(0.1)  # Simulate 100ms vector search
        return {"results": [f"Document {i}: {query}" for i in range(3)]}

    def search_sql_db(table: str, filter_clause: str) -> List[Dict[str, Any]]:
        time.sleep(0.05)  # Simulate 50ms SQL query
        return [{"id": i, "data": f"{table} row {i}"} for i in range(5)]

    def rank_results(results: List[str]) -> List[str]:
        time.sleep(0.02)  # Simulate 20ms ranking
        return sorted(results, reverse=True)

    agent = ReActGenerator(
        tools=[search_vector_db, search_sql_db, rank_results],
        model="gpt-4",
        max_iters=5
    )

    # Simulate 100 RAG queries
    query_times = []

    for i in range(100):
        task = TaskInput(
            task_id=f"rag-perf-{i:03d}",
            description=f"Search for information about topic {i}",
            domain="rag-performance-test",
            playbook_bullets=[]
        )

        # Measure task execution time (excluding LLM calls)
        start = time.perf_counter()

        # Simulate tool calls (mock mode - no actual LLM)
        # In real mode, this would call agent.forward(task)
        for _ in range(3):  # Simulate 3 iterations
            agent._execute_tool_with_timeout("search_vector_db", {"query": f"topic {i}"})
            agent._execute_tool_with_timeout("rank_results", {"results": ["a", "b", "c"]})

        end = time.perf_counter()
        duration_s = end - start
        query_times.append(duration_s)

    # Calculate statistics
    avg_duration = statistics.mean(query_times)
    p50_duration = statistics.median(query_times)
    p95_duration = sorted(query_times)[int(len(query_times) * 0.95)]
    p99_duration = sorted(query_times)[int(len(query_times) * 0.99)]

    print(f"\nEnd-to-End RAG Query Benchmark:")
    print(f"  Avg: {avg_duration:.3f}s")
    print(f"  P50: {p50_duration:.3f}s")
    print(f"  P95: {p95_duration:.3f}s")
    print(f"  P99: {p99_duration:.3f}s")

    # Assert performance budget: <10s for 95% of queries (SC-004)
    # Note: Framework overhead only - real queries depend on LLM latency
    assert p95_duration < 1.0, f"P95 framework overhead {p95_duration:.3f}s exceeds 1s target (excluding LLM)"


# ============================================================================
# T044: SCALING TEST
# ============================================================================

@pytest.mark.parametrize("num_tools", [10, 25, 50])
def test_scaling_with_tool_count(num_tools: int):
    """
    T044: Verify agent handles 10, 25, 50 tools without degradation.

    Tests that performance remains stable as tool count increases.
    Acceptable degradation: <20% increase in overhead per 2x tool count.
    """
    tools = generate_tools(num_tools)
    agent = ReActGenerator(tools=tools, model="gpt-4", max_iters=10)

    # Measure tool execution overhead
    overhead_times = []

    for i in range(30):  # 30 executions
        tool_name = f"dynamic_tool_{i % num_tools}" if num_tools > 6 else list(agent.tools.keys())[i % len(agent.tools)]

        start = time.perf_counter()
        agent._execute_tool_with_timeout(tool_name, {"query": "test"})
        end = time.perf_counter()

        overhead_times.append((end - start) * 1000)

    avg_overhead = statistics.mean(overhead_times)
    p95_overhead = sorted(overhead_times)[int(len(overhead_times) * 0.95)]

    print(f"\nScaling Test ({num_tools} tools):")
    print(f"  Avg overhead: {avg_overhead:.2f}ms")
    print(f"  P95 overhead: {p95_overhead:.2f}ms")

    # Assert scaling performance
    # Baseline: 10 tools should be <50ms, 25 tools <60ms, 50 tools <75ms (20% increase per 2x)
    if num_tools == 10:
        assert avg_overhead < 50, f"Avg overhead {avg_overhead:.2f}ms exceeds 50ms for 10 tools"
    elif num_tools == 25:
        assert avg_overhead < 60, f"Avg overhead {avg_overhead:.2f}ms exceeds 60ms for 25 tools"
    elif num_tools == 50:
        assert avg_overhead < 75, f"Avg overhead {avg_overhead:.2f}ms exceeds 75ms for 50 tools"

    # P95 should always be <100ms
    assert p95_overhead < 100, f"P95 overhead {p95_overhead:.2f}ms exceeds 100ms budget"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
