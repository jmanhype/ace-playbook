"""
Batch Tool Learning - Strategy Evolution Demo

Demonstrates:
- Processing 100+ tasks to show learning at scale
- Strategy evolution tracking (playbook growth over time)
- Iteration reduction visualization
- Playbook analytics (success rates, tool patterns, domain insights)

Based on User Story 3 (US3) from 001-tool-calling-agent spec.

Usage:
    # Mock mode (no LLM required)
    python examples/batch_tool_learning.py --mock

    # Real mode (requires OpenAI API key)
    export OPENAI_API_KEY=your-api-key
    python examples/batch_tool_learning.py

Expected Output:
    - Batch processing of 100+ tasks
    - Strategy evolution metrics over time
    - Iteration reduction trends
    - Playbook analytics dashboard
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import random
from collections import defaultdict

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
# MOCK TOOLS FOR BATCH PROCESSING
# ============================================================================

def search_documents(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search document database for relevant results."""
    if random.random() < 0.15:  # 15% failure rate
        raise TimeoutError(f"Document search timeout for: {query}")

    results = [
        {"id": i, "title": f"Doc {i}", "score": random.uniform(0.5, 1.0)}
        for i in range(random.randint(1, limit))
    ]
    return {"query": query, "results": results, "count": len(results)}


def fetch_metadata(doc_id: int) -> Dict[str, Any]:
    """Fetch metadata for a specific document."""
    if random.random() < 0.1:  # 10% failure rate
        raise ValueError(f"Invalid document ID: {doc_id}")

    return {
        "id": doc_id,
        "author": f"Author-{doc_id}",
        "date": "2024-01-01",
        "tags": ["tag1", "tag2"]
    }


def rank_results(results: List[Dict], method: str = "relevance") -> List[Dict]:
    """Rank search results by specified method."""
    if random.random() < 0.05:  # 5% failure rate
        raise ValueError(f"Unsupported ranking method: {method}")

    if method == "relevance":
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    elif method == "recency":
        random.shuffle(results)
        return results
    else:
        return results


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text."""
    if random.random() < 0.08:  # 8% failure rate
        raise RuntimeError("Keyword extraction service unavailable")

    keywords = [f"keyword-{i}" for i in range(random.randint(1, max_keywords))]
    return keywords


def summarize_text(text: str, max_length: int = 100) -> str:
    """Generate text summary."""
    if random.random() < 0.12:  # 12% failure rate
        raise TimeoutError("Summarization timeout")

    return f"Summary of text (max {max_length} chars): {text[:50]}..."


# ============================================================================
# MOCK REFLECTOR FOR BATCH LEARNING
# ============================================================================

class BatchMockReflector:
    """Mock reflector that simulates realistic tool learning patterns."""

    def __call__(self, reflector_input: ReflectorInput):
        """Generate insights based on task success and tool usage."""
        insights = []

        # Success pattern - generate helpful insights
        if not reflector_input.iteration_limit_reached and reflector_input.total_iterations < 8:
            if reflector_input.tools_used:
                tool_chain = " → ".join(reflector_input.tools_used)

                # Primary success pattern
                insights.append(
                    InsightCandidate(
                        content=f"Efficient workflow: {tool_chain}",
                        section=InsightSection.HELPFUL,
                        confidence=0.85,
                        rationale="Tool sequence completed task successfully",
                        tags=["batch-processing", "tool-calling"],
                        referenced_steps=[0, 1],
                        tool_sequence=reflector_input.tools_used,
                        tool_success_rate=1.0,
                        avg_iterations=reflector_input.total_iterations,
                        avg_execution_time_ms=random.uniform(50.0, 200.0),
                    )
                )

                # Domain-specific pattern (30% chance)
                if random.random() < 0.3 and len(reflector_input.tools_used) >= 2:
                    insights.append(
                        InsightCandidate(
                            content=f"For {reflector_input.domain} tasks, start with {reflector_input.tools_used[0]}",
                            section=InsightSection.HELPFUL,
                            confidence=0.75,
                            rationale="First tool provides good initial context",
                            tags=[reflector_input.domain, "optimization"],
                            referenced_steps=[0],
                            tool_sequence=reflector_input.tools_used[:1],
                            tool_success_rate=1.0,
                            avg_iterations=reflector_input.total_iterations,
                            avg_execution_time_ms=random.uniform(30.0, 100.0),
                        )
                    )

        # Failure pattern - generate harmful insights
        else:
            if reflector_input.tools_used:
                insights.append(
                    InsightCandidate(
                        content=f"Avoid pattern: {', '.join(reflector_input.tools_used)} - causes excessive iterations",
                        section=InsightSection.HARMFUL,
                        confidence=0.80,
                        rationale="Tool sequence led to iteration limit",
                        tags=["batch-processing", "tool-failure"],
                        referenced_steps=[0, 1],
                        tool_sequence=reflector_input.tools_used,
                        tool_success_rate=0.0,
                        avg_iterations=reflector_input.total_iterations,
                        avg_execution_time_ms=random.uniform(300.0, 500.0),
                    )
                )

        return ReflectorOutput(
            task_id=reflector_input.task_id,
            insights=insights,
            analysis_summary="Batch processing analysis",
            referenced_steps=list(range(len(reflector_input.reasoning_trace))),
            confidence_score=0.85,
            feedback_types_used=[],
            requires_human_review=False,
            contradicts_existing=[],
        )


# ============================================================================
# BATCH TASK GENERATION
# ============================================================================

def create_batch_tasks(num_tasks: int = 100) -> List[TaskInput]:
    """Create diverse batch of tasks for learning demonstration."""

    task_templates = [
        ("document-search", "Search for documents about {topic}"),
        ("document-search", "Find relevant papers on {topic}"),
        ("metadata-extraction", "Get metadata for document {doc_id}"),
        ("content-ranking", "Rank search results by relevance"),
        ("keyword-extraction", "Extract keywords from {topic} documents"),
        ("text-summarization", "Summarize findings about {topic}"),
        ("multi-tool", "Search {topic}, extract keywords, and summarize"),
        ("multi-tool", "Find documents about {topic} and rank by relevance"),
    ]

    topics = ["AI", "machine learning", "databases", "cloud computing", "security",
              "networking", "algorithms", "data structures", "web development", "testing"]

    tasks = []
    for i in range(num_tasks):
        domain, template = random.choice(task_templates)

        # Fill template
        if "{topic}" in template:
            description = template.format(topic=random.choice(topics))
        elif "{doc_id}" in template:
            description = template.format(doc_id=random.randint(1, 100))
        else:
            description = template

        task = TaskInput(
            task_id=f"batch-{i:03d}",
            description=description,
            domain=domain,
            playbook_bullets=[],
        )
        tasks.append(task)

    return tasks


# ============================================================================
# ANALYTICS AND VISUALIZATION
# ============================================================================

class BatchAnalytics:
    """Track and analyze batch processing metrics."""

    def __init__(self):
        self.task_metrics: List[Dict[str, Any]] = []
        self.playbook_snapshots: List[Dict[str, Any]] = []
        self.domain_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "total_iterations": 0,
            "tools_used": set(),
        })

    def record_task(
        self,
        task_id: str,
        domain: str,
        tools_used: List[str],
        iterations: int,
        success: bool,
        playbook_size: int,
    ):
        """Record metrics for a single task."""
        self.task_metrics.append({
            "task_id": task_id,
            "domain": domain,
            "tools_used": tools_used,
            "iterations": iterations,
            "success": success,
            "playbook_size": playbook_size,
        })

        # Update domain stats
        stats = self.domain_stats[domain]
        stats["total"] += 1
        if success:
            stats["successful"] += 1
        stats["total_iterations"] += iterations
        for tool in tools_used:
            stats["tools_used"].add(tool)

    def record_playbook_snapshot(self, task_idx: int, playbook: List[PlaybookBullet]):
        """Record playbook state at specific intervals."""
        self.playbook_snapshots.append({
            "task_idx": task_idx,
            "total_bullets": len(playbook),
            "helpful": sum(1 for b in playbook if b.section == "Helpful"),
            "harmful": sum(1 for b in playbook if b.section == "Harmful"),
            "domains": len(set(b.domain_id for b in playbook)),
        })

    def print_summary(self):
        """Print comprehensive analytics summary."""
        print("\n" + "=" * 80)
        print("BATCH PROCESSING ANALYTICS")
        print("=" * 80)

        # Overall metrics
        total_tasks = len(self.task_metrics)
        successful_tasks = sum(1 for m in self.task_metrics if m["success"])
        total_iterations = sum(m["iterations"] for m in self.task_metrics)
        avg_iterations = total_iterations / total_tasks if total_tasks > 0 else 0

        print(f"\n📊 Overall Performance:")
        print(f"  Total tasks processed: {total_tasks}")
        print(f"  Successful tasks: {successful_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
        print(f"  Average iterations per task: {avg_iterations:.2f}")

        # Iteration reduction trend
        print(f"\n📈 Iteration Reduction Trend:")
        if len(self.task_metrics) >= 20:
            first_20 = self.task_metrics[:20]
            last_20 = self.task_metrics[-20:]

            avg_first_20 = sum(m["iterations"] for m in first_20) / 20
            avg_last_20 = sum(m["iterations"] for m in last_20) / 20

            reduction = ((avg_first_20 - avg_last_20) / avg_first_20 * 100) if avg_first_20 > 0 else 0

            print(f"  First 20 tasks avg: {avg_first_20:.2f} iterations")
            print(f"  Last 20 tasks avg: {avg_last_20:.2f} iterations")
            print(f"  Reduction: {reduction:.1f}% {'✓' if reduction >= 30 else '⚠️'}")

        # Playbook growth
        print(f"\n📚 Playbook Evolution:")
        if self.playbook_snapshots:
            for snapshot in self.playbook_snapshots:
                print(f"  After {snapshot['task_idx']:3d} tasks: "
                      f"{snapshot['total_bullets']:3d} bullets "
                      f"({snapshot['helpful']} helpful, {snapshot['harmful']} harmful, "
                      f"{snapshot['domains']} domains)")

        # Domain analysis
        print(f"\n🎯 Domain Performance:")
        for domain, stats in sorted(self.domain_stats.items()):
            success_rate = stats["successful"] / stats["total"] * 100 if stats["total"] > 0 else 0
            avg_iters = stats["total_iterations"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {domain:20s}: {stats['total']:3d} tasks, "
                  f"{success_rate:5.1f}% success, {avg_iters:5.2f} avg iterations, "
                  f"{len(stats['tools_used'])} unique tools")

        # Tool usage patterns
        print(f"\n🔧 Tool Usage Patterns:")
        tool_frequency: Dict[str, int] = defaultdict(int)
        for metric in self.task_metrics:
            for tool in metric["tools_used"]:
                tool_frequency[tool] += 1

        for tool, count in sorted(tool_frequency.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool:30s}: {count:3d} uses ({count/total_tasks*100:.1f}%)")


# ============================================================================
# BATCH PROCESSING WORKFLOW
# ============================================================================

def run_batch_learning(use_mock: bool = True, num_tasks: int = 100):
    """
    Demonstrate batch processing with strategy evolution tracking.

    Args:
        use_mock: Use mock reflector (no LLM) if True
        num_tasks: Number of tasks to process
    """
    print("=" * 80)
    print(f"BATCH TOOL LEARNING - PROCESSING {num_tasks} TASKS")
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
            search_documents,
            fetch_metadata,
            rank_results,
            extract_keywords,
            summarize_text,
        ],
        model="gpt-4",
        max_iters=10
    )
    reflector = BatchMockReflector() if use_mock else GroundedReflector()
    curator = SemanticCurator()
    analytics = BatchAnalytics()

    # Create batch tasks
    tasks = create_batch_tasks(num_tasks)

    # Maintain separate playbooks per domain (enforced by domain isolation)
    domain_playbooks: Dict[str, List[PlaybookBullet]] = defaultdict(list)

    print(f"Created {len(tasks)} tasks across multiple domains")
    print(f"Available tools: {len(agent.tools)}")
    print("\nProcessing tasks...\n")

    # Process tasks with periodic snapshots
    snapshot_interval = 10

    for i, task in enumerate(tasks):
        # Simulate task execution
        tools_used = []
        total_iterations = random.randint(2, 10)
        iteration_limit_reached = total_iterations > 8

        # Simulate tool selection based on task domain
        if task.domain == "document-search":
            tools_used = ["search_documents"]
            if random.random() < 0.4:  # 40% use ranking
                tools_used.append("rank_results")
        elif task.domain == "metadata-extraction":
            tools_used = ["fetch_metadata"]
        elif task.domain == "content-ranking":
            tools_used = ["search_documents", "rank_results"]
        elif task.domain == "keyword-extraction":
            tools_used = ["extract_keywords"]
        elif task.domain == "text-summarization":
            tools_used = ["summarize_text"]
        else:  # multi-tool
            tools_used = random.sample(
                ["search_documents", "extract_keywords", "rank_results", "summarize_text"],
                k=random.randint(2, 3)
            )

        success = not iteration_limit_reached

        # Reflection
        reflector_input = ReflectorInput(
            task_id=task.task_id,
            reasoning_trace=[f"Step {j+1}" for j in range(total_iterations)],
            answer="Mock result",
            confidence=0.8 if success else 0.5,
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

            # Use domain-specific playbook
            current_domain_playbook = domain_playbooks[task.domain]

            curator_input = CuratorInput(
                task_id=task.task_id,
                domain_id=task.domain,
                insights=insights_dicts,
                current_playbook=current_domain_playbook,
                target_stage=PlaybookStage.SHADOW,
            )

            curator_output = curator.apply_delta(curator_input)
            domain_playbooks[task.domain] = curator_output.updated_playbook

        # Get combined playbook size across all domains
        combined_playbook = [bullet for bullets in domain_playbooks.values() for bullet in bullets]

        # Record metrics
        analytics.record_task(
            task_id=task.task_id,
            domain=task.domain,
            tools_used=tools_used,
            iterations=total_iterations,
            success=success,
            playbook_size=len(combined_playbook),
        )

        # Periodic snapshot
        if (i + 1) % snapshot_interval == 0:
            analytics.record_playbook_snapshot(i + 1, combined_playbook)
            print(f"  [{i+1:3d}/{num_tasks}] Playbook: {len(combined_playbook)} bullets "
                  f"(+{len(reflector_output.insights)} new insights)")

    # Final snapshot
    combined_playbook = [bullet for bullets in domain_playbooks.values() for bullet in bullets]
    analytics.record_playbook_snapshot(num_tasks, combined_playbook)

    # Print analytics
    analytics.print_summary()

    # Demonstrate high-success strategy retrieval (T037)
    print("\n" + "=" * 80)
    print("HIGH-SUCCESS STRATEGIES (T037)")
    print("=" * 80)

    # Get document-search playbook if available
    doc_search_playbook = domain_playbooks.get("document-search", [])

    if doc_search_playbook:
        high_success = curator.get_high_success_strategies(
            playbook=doc_search_playbook,
            domain_id="document-search",
            min_success_rate=0.7,
            max_results=5,
        )

        print(f"\nTop strategies for 'document-search' domain (success rate ≥ 70%):")
        for i, bullet in enumerate(high_success, 1):
            tools = f" [{', '.join(bullet.tool_sequence)}]" if bullet.tool_sequence else ""
            rate = f"{bullet.tool_success_rate*100:.1f}%" if bullet.tool_success_rate else "N/A"
            print(f"  {i}. {bullet.content}{tools} (success: {rate})")

        if not high_success:
            print("  (No strategies meeting criteria)")
    else:
        print("\n  (No document-search tasks processed)")

    # Demonstrate cross-domain pattern transfer (T038)
    print("\n" + "=" * 80)
    print("CROSS-DOMAIN PATTERN TRANSFER (T038)")
    print("=" * 80)

    # Find patterns from related domains using combined playbook
    if combined_playbook and doc_search_playbook:
        # Use first document-search bullet as query
        query_bullet = doc_search_playbook[0]

        similar_patterns = curator.find_cross_domain_patterns(
            playbook=combined_playbook,
            source_domain="document-search",
            related_domains=["keyword-extraction", "content-ranking", "multi-tool"],
            query_embedding=query_bullet.embedding,
            similarity_threshold=0.75,
            max_results=3,
        )

        print(f"\nPatterns from related domains similar to '{query_bullet.content[:60]}...':")
        for i, bullet in enumerate(similar_patterns, 1):
            tools = f" [{', '.join(bullet.tool_sequence)}]" if bullet.tool_sequence else ""
            print(f"  {i}. [{bullet.domain_id}] {bullet.content}{tools}")

        if not similar_patterns:
            print("  (No similar patterns found in related domains)")
    else:
        print("\n  (Insufficient data for cross-domain analysis)")

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Tool Learning Demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock reflector (no LLM required)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=100,
        help="Number of tasks to process (default: 100)",
    )
    args = parser.parse_args()

    run_batch_learning(
        use_mock=args.mock or not os.environ.get("OPENAI_API_KEY"),
        num_tasks=args.tasks
    )
