# Quickstart: Tool-Calling Agent with ReAct Reasoning

**Feature**: 001-tool-calling-agent
**Target Audience**: Developers integrating ReAct agents into their applications
**Time to First Success**: 30 minutes

---

## What You'll Build

By the end of this guide, you'll have a working ReAct agent that:
- Queries multiple databases iteratively to find information
- Learns which tool sequences work best over time
- Reduces query iterations by 30-50% after 20+ examples

**Use Case**: RAG application that searches across vector and SQL databases

---

## Prerequisites

- Python 3.11+
- ACE Playbook installed (`pip install ace-playbook`)
- Basic understanding of DSPy (helpful but not required)
- 5-10 example tasks for initial learning

---

## Step 1: Define Your Tools (5 minutes)

Tools are simple Python functions with type annotations. The agent will learn which tools to use and when.

```python
# tools.py
from typing import List, Dict

def search_vector_db(query: str, k: int = 5) -> List[Dict]:
    """
    Search vector database for semantically similar documents.

    Args:
        query: Search query text
        k: Number of results to return (default: 5)

    Returns:
        List of documents with similarity scores
    """
    # Your vector DB implementation (FAISS, Pinecone, Qdrant, etc.)
    results = vector_db.similarity_search(query, k=k)
    return [{"text": r.text, "score": r.score} for r in results]


def search_sql_db(table: str, filters: Dict) -> List[Dict]:
    """
    Search SQL database with filters.

    Args:
        table: Table name to query
        filters: Dictionary of column: value filters

    Returns:
        List of matching rows
    """
    # Your SQL DB implementation (PostgreSQL, MySQL, etc.)
    query = f"SELECT * FROM {table} WHERE " + " AND ".join(
        f"{k}='{v}'" for k, v in filters.items()
    )
    return db.execute(query).fetchall()


def rank_results(results: List[Dict], criteria: str) -> List[Dict]:
    """
    Rank and filter results based on criteria.

    Args:
        results: List of results to rank
        criteria: Ranking criteria (e.g., "relevance", "recency")

    Returns:
        Ranked and filtered results
    """
    if criteria == "relevance":
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:3]
    elif criteria == "recency":
        return sorted(results, key=lambda x: x.get("timestamp", 0), reverse=True)[:3]
    return results[:3]
```

**Tool Best Practices**:
- ✅ Use type annotations for all parameters
- ✅ Include docstrings (used in LLM prompts)
- ✅ Keep tools focused (single responsibility)
- ✅ Use descriptive verb names (`search_`, `filter_`, `rank_`)
- ❌ Avoid tools that depend on other tools (agent will chain them)

---

## Step 2: Create Your ReAct Agent (5 minutes)

```python
# agent.py
from ace.generator import ReActGenerator
from tools import search_vector_db, search_sql_db, rank_results

# Initialize agent with tools
agent = ReActGenerator(
    tools=[search_vector_db, search_sql_db, rank_results],
    model="gpt-4",  # or "claude-3-opus", "gemini-pro"
    max_iters=15    # Agent-level default (can override per task)
)

# Verify tools are valid
errors = agent.validate_tools()
if errors:
    print("Tool validation errors:", errors)
    exit(1)

print("✅ Agent initialized with 3 tools")
print(f"   Tools: {', '.join([t.__name__ for t in agent.tools])}")
```

---

## Step 3: Execute Your First Task (5 minutes)

```python
# main.py
from ace.generator import TaskInput
from agent import agent

# Create task
task = TaskInput(
    task_id="query-001",
    description="Find the top 3 most relevant documents about 'machine learning optimization' published in the last 6 months",
    playbook_bullets=[],  # Empty on first run (will learn strategies)
    domain="ml-research"
)

# Execute
output = agent.forward(task)

# View results
print(f"\n📊 Answer: {output.answer}")
print(f"\n🔧 Tools used: {' → '.join(output.tools_used)}")
print(f"\n🔄 Iterations: {output.total_iterations}")
print(f"\n✅ Confidence: {output.confidence:.2f}")

# Inspect reasoning (optional)
print("\n🧠 Reasoning Trace:")
for step in output.structured_trace:
    print(f"  [{step.iteration}] {step.thought}")
    if step.action == "call_tool":
        print(f"      → {step.tool_name}({step.tool_args})")
        print(f"      → {step.observation[:100]}...")  # First 100 chars
```

**Expected Output** (first run, no learning yet):
```
📊 Answer: Found 3 documents: "Gradient Descent Optimization" (0.89), "Neural Network Training" (0.85), "Efficient ML Models" (0.82)

🔧 Tools used: search_sql_db → search_vector_db → rank_results

🔄 Iterations: 4

✅ Confidence: 0.85

🧠 Reasoning Trace:
  [1] I need to find recent documents about ML optimization
      → search_sql_db({'table': 'documents', 'filters': {'topic': 'machine learning', 'date_range': '6_months'}})
      → Found 1,200 documents matching filters
  [2] Too many results, need semantic search to narrow down
      → search_vector_db({'query': 'machine learning optimization', 'k': 10})
      → Found 10 semantically relevant documents
  [3] Now rank by relevance to get top 3
      → rank_results({'results': [...], 'criteria': 'relevance'})
      → Ranked and filtered to top 3 documents
  [4] Have final answer with high confidence
```

---

## Step 4: Enable Learning (5 minutes)

Connect the Reflector to analyze tool usage and build the playbook.

```python
# learning.py
from ace.reflector import GroundedReflector
from ace.curator import SemanticCurator
from ace.playbook import Playbook

# Initialize ACE components
reflector = GroundedReflector()
curator = SemanticCurator(domain="ml-research")
playbook = Playbook(domain="ml-research")

# After task execution
insights = reflector.analyze(output, ground_truth="Expected answer")  # ground_truth optional

# Curator merges insights into playbook
curator.merge(insights)

# Playbook now contains tool-calling strategies
strategies = playbook.retrieve(
    query="database search",
    filters={"has_tool_sequence": True},
    k=3
)

print("\n📚 Learned Strategies:")
for strategy in strategies:
    print(f"  • {strategy.content}")
    if strategy.tool_sequence:
        print(f"    Pattern: {' → '.join(strategy.tool_sequence)}")
    print(f"    Success rate: {strategy.tool_success_rate:.0%} ({strategy.helpful_count} successful uses)")
```

**Expected Output** (after 5-10 tasks):
```
📚 Learned Strategies:
  • For ML research queries, filter SQL first by date/topic, then semantic search, then rank
    Pattern: search_sql_db → search_vector_db → rank_results
    Success rate: 87% (13 successful uses)

  • When SQL returns <100 results, skip to vector search directly
    Pattern: search_vector_db → rank_results
    Success rate: 92% (11 successful uses)

  • For broad queries, always apply temporal filter to reduce result set
    Pattern: search_sql_db → search_vector_db
    Success rate: 78% (7 successful uses)
```

---

## Step 5: Run with Playbook Context (5 minutes)

Now the agent uses learned strategies to reduce iterations.

```python
# run_with_learning.py
from agent import agent
from learning import playbook

# Retrieve learned strategies
task = TaskInput(
    task_id="query-002",
    description="Find recent papers on 'neural network pruning'",  # Similar to previous tasks
    playbook_bullets=playbook.retrieve(
        query="Find recent papers on neural network pruning",
        domain="ml-research",
        filters={"has_tool_sequence": True},
        k=3
    ),
    domain="ml-research"
)

# Execute with playbook context
output = agent.forward(task)

print(f"\n📊 Answer: {output.answer}")
print(f"\n🔧 Tools used: {' → '.join(output.tools_used)}")
print(f"\n🔄 Iterations: {output.total_iterations}  (vs 4 on first run)")
print(f"\n📈 Improvement: {((4 - output.total_iterations) / 4) * 100:.0f}% fewer iterations")
```

**Expected Output** (with learning):
```
📊 Answer: Found 3 papers: "Efficient Neural Network Pruning" (0.91), "Sparse Networks" (0.88), "Structured Pruning Methods" (0.85)

🔧 Tools used: search_sql_db → search_vector_db → rank_results

🔄 Iterations: 2  (vs 4 on first run)

📈 Improvement: 50% fewer iterations
```

**What Happened**:
- Agent saw playbook strategy: "For ML research, filter SQL → semantic search → rank"
- Applied this pattern directly instead of exploring alternatives
- Completed task in 2 iterations instead of 4
- **Achieved SC-002 success criteria** (30-50% reduction) ✅

---

## Step 6: Production Integration (5 minutes)

### Batch Processing with Learning

```python
# batch_processor.py
import json
from pathlib import Path

# Load tasks
tasks_file = Path("tasks.json")
tasks = json.loads(tasks_file.read_text())

results = []
for task_data in tasks:
    # Retrieve playbook strategies
    strategies = playbook.retrieve(
        query=task_data["description"],
        domain="ml-research",
        k=3
    )

    # Create task with playbook context
    task = TaskInput(
        task_id=task_data["id"],
        description=task_data["description"],
        playbook_bullets=[s.content for s in strategies],
        domain="ml-research"
    )

    # Execute
    output = agent.forward(task)

    # Reflect and learn
    insights = reflector.analyze(output, ground_truth=task_data.get("expected_answer"))
    curator.merge(insights)

    # Save result
    results.append({
        "task_id": task.task_id,
        "answer": output.answer,
        "tools_used": output.tools_used,
        "iterations": output.total_iterations,
        "confidence": output.confidence
    })

# Save results
Path("results.json").write_text(json.dumps(results, indent=2))

# Print learning summary
print(f"\n📊 Batch Summary:")
print(f"   Tasks processed: {len(results)}")
print(f"   Avg iterations: {sum(r['iterations'] for r in results) / len(results):.1f}")
print(f"   Avg confidence: {sum(r['confidence'] for r in results) / len(results):.2f}")
print(f"\n📚 Playbook now has {len(playbook.get_all())} strategies")
```

### Error Handling

```python
# error_handling.py
from ace.generator import MaxIterationsExceededError, ToolExecutionError

try:
    output = agent.forward(task, max_iters=5)  # Strict limit
except MaxIterationsExceededError as e:
    print(f"⚠️ Agent hit iteration limit without completing task")
    print(f"   Partial result: {e.partial_output.answer}")
    print(f"   Tools attempted: {e.partial_output.tools_used}")
    # Reflect on failure to learn anti-patterns
    insights = reflector.analyze(e.partial_output, success=False)
    curator.merge(insights)  # Creates "Harmful" bullet for this tool sequence

except ToolExecutionError as e:
    print(f"❌ Tool execution failed: {e}")
    print(f"   Tool: {e.tool_name}")
    print(f"   Error: {e.message}")
    # Agent continues with other tools (graceful degradation)
```

---

## Next Steps

### 1. Monitor Performance

```python
# metrics.py
import time

start = time.time()
output = agent.forward(task)
duration = time.time() - start

print(f"\n⏱️ Performance Metrics:")
print(f"   Total time: {duration:.2f}s")
print(f"   Time per iteration: {duration / output.total_iterations:.2f}s")
print(f"   Tool call overhead: {output.metadata.get('tool_overhead_ms', 0):.1f}ms")

# Success criteria check
assert duration < 10.0, "❌ Failed SC-004: Query should complete in <10s"
assert output.total_iterations < 10, "❌ Too many iterations"
print("✅ All performance criteria met")
```

### 2. Advanced Tool Patterns

```python
# advanced_tools.py

# Tool with error handling
def safe_search_db(query: str) -> str:
    """Search with automatic retry and fallback."""
    try:
        return search_vector_db(query)
    except Exception as e:
        # Fallback to keyword search
        return search_sql_db("documents", {"keywords": query})

# Tool with caching
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_search(query: str) -> str:
    """Search with result caching for repeated queries."""
    return search_vector_db(query)

# Tool with validation
def validated_rank(results: List[Dict], criteria: str) -> List[Dict]:
    """Rank results with input validation."""
    if not results:
        raise ValueError("No results to rank")
    if criteria not in ["relevance", "recency", "popularity"]:
        raise ValueError(f"Invalid criteria: {criteria}")
    return rank_results(results, criteria)
```

### 3. Multi-Agent Workflows

```python
# multi_agent.py
from ace.generator import ReActGenerator

# Specialized agents
research_agent = ReActGenerator(
    tools=[search_vector_db, search_sql_db, rank_results],
    model="gpt-4",
    max_iters=15
)

analysis_agent = ReActGenerator(
    tools=[analyze_text, summarize_document, extract_entities],
    model="claude-3-opus",
    max_iters=10
)

# Workflow
search_output = research_agent.forward(search_task)
analysis_output = analysis_agent.forward(TaskInput(
    task_id="analysis-001",
    description=f"Analyze these search results: {search_output.answer}",
    domain="analysis"
))

print(f"Final answer: {analysis_output.answer}")
```

---

## Troubleshooting

### Issue: Agent uses wrong tools

**Cause**: Playbook doesn't have relevant strategies yet

**Solution**: Run 10-20 example tasks with ground truth to build playbook

```python
# bootstrap_playbook.py
training_tasks = [
    {"description": "Find ML papers", "expected_tools": ["search_sql_db", "search_vector_db"]},
    {"description": "Get recent research", "expected_tools": ["search_sql_db", "rank_results"]},
    # ... 10-20 examples
]

for task_data in training_tasks:
    task = TaskInput(task_id=f"train-{i}", description=task_data["description"], domain="ml-research")
    output = agent.forward(task)

    # Provide explicit feedback
    success = set(output.tools_used) == set(task_data["expected_tools"])
    insights = reflector.analyze(output, success=success)
    curator.merge(insights)
```

### Issue: Too many iterations

**Cause**: Agent exploring tools unnecessarily

**Solution**: Lower max_iters and provide more specific playbook context

```python
# Retrieve more specific strategies
strategies = playbook.retrieve(
    query=task.description,
    domain="ml-research",
    filters={
        "has_tool_sequence": True,
        "tool_success_rate": {"$gt": 0.8}  # Only high-success strategies
    },
    k=5  # More strategies for better guidance
)
```

### Issue: Tool validation errors

**Cause**: Tool signature missing type annotations

**Solution**: Add annotations and docstrings

```python
# ❌ Bad
def search(query):
    return db.search(query)

# ✅ Good
def search(query: str, limit: int = 10) -> List[Dict]:
    """Search database for query.

    Args:
        query: Search query text
        limit: Max results to return

    Returns:
        List of matching documents
    """
    return db.search(query, limit=limit)
```

---

## Resources

- **Full API Reference**: See `contracts/react_generator_api.py`
- **Data Model**: See `data-model.md`
- **Examples**: Check `examples/react_rag_agent.py`
- **Constitution**: Read `.specify/memory/constitution.md` for ACE principles

---

## Success Criteria Checklist

After completing this guide, verify you've achieved:

- ✅ **SC-006**: Integrated tools and ran first task in <30 minutes
- ✅ **SC-001**: Agent completes 90% of 2-5 tool tasks within iteration limit
- ✅ **SC-002**: Iterations reduced 30-50% after 20+ learning examples
- ✅ **SC-003**: Tool strategies captured in playbook within 3 similar tasks
- ✅ **SC-004**: RAG queries complete in <10s (95% of queries)
- ✅ **SC-005**: Agent finds alternatives when primary tools fail (95% success)

**Congratulations!** You've successfully built a learning tool-calling agent. 🎉

---

**Next**: Run `/speckit.tasks` to generate implementation tasks and start building.
