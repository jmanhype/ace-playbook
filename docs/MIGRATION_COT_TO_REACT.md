# Migration Guide: CoTGenerator to ReActGenerator

This guide helps you migrate from `CoTGenerator` to `ReActGenerator` to enable tool-calling capabilities while maintaining backward compatibility.

## Table of Contents

- [Why Migrate?](#why-migrate)
- [Backward Compatibility](#backward-compatibility)
- [Migration Strategies](#migration-strategies)
- [Step-by-Step Migration](#step-by-step-migration)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

---

## Why Migrate?

**ReActGenerator** extends CoTGenerator with tool-calling capabilities:

| Feature | CoTGenerator | ReActGenerator |
|---------|-------------|----------------|
| Chain-of-Thought reasoning | ✅ | ✅ |
| Tool calling | ❌ | ✅ |
| Multi-step workflows | ❌ | ✅ |
| Tool failure handling | ❌ | ✅ |
| Performance tracking | ❌ | ✅ |
| Graceful degradation | ❌ | ✅ |

**When to migrate:**
- You need agents to use external tools (APIs, databases, calculators)
- You want multi-step workflows with tool orchestration
- You need automatic failure recovery and tool adaptation
- You want to track tool usage patterns and learn strategies

**When to stay with CoTGenerator:**
- Pure reasoning tasks without external tool calls
- Simple Q&A that doesn't require data retrieval
- Existing implementations that don't need tool support

---

## Backward Compatibility

**ReActGenerator is a drop-in replacement for CoTGenerator when used without tools.**

### ✅ What's Compatible

```python
# Old code (CoTGenerator)
from ace.generator.cot_generator import CoTGenerator

agent = CoTGenerator(model="gpt-4")
output = agent.forward(task)

# New code (ReActGenerator) - WORKS THE SAME
from ace.generator.react_generator import ReActGenerator

agent = ReActGenerator(model="gpt-4")  # No tools = CoT mode
output = agent.forward(task)
```

Both return the same `TaskOutput` structure with:
- `task_id`
- `answer`
- `confidence`
- `reasoning_trace`
- `bullets_referenced`
- `latency_ms`
- `model_name`

### 🆕 What's New (Optional)

ReActGenerator adds optional fields when tools are used:
- `structured_trace`: List[ReasoningStep] with timing metadata
- `tools_used`: List of tool names used
- `total_iterations`: Number of ReAct iterations
- `iteration_limit_reached`: Boolean flag
- `metadata`: Performance metrics and failure tracking

These fields are **empty/default when no tools are registered**, ensuring backward compatibility.

---

## Migration Strategies

### Strategy 1: Direct Replacement (No Tools)

**Best for:** Maintaining existing behavior while preparing for future tool additions.

```python
# Before
from ace.generator.cot_generator import CoTGenerator
agent = CoTGenerator(model="gpt-4")

# After (simple find-and-replace)
from ace.generator.react_generator import ReActGenerator
agent = ReActGenerator(model="gpt-4")
```

**No code changes needed.** Works identically to CoTGenerator.

### Strategy 2: Gradual Tool Addition

**Best for:** Incrementally adding tool capabilities to existing agents.

```python
from ace.generator.react_generator import ReActGenerator

# Step 1: Start without tools (backward compatible)
agent = ReActGenerator(model="gpt-4")

# Step 2: Add tools as needed
def search_tool(query: str) -> list[str]:
    """Search database."""
    return ["result1", "result2"]

agent.register_tool(search_tool)

# Step 3: Gradually add more tools
def calculator(expression: str) -> float:
    """Calculate expression."""
    return eval(expression)

agent.register_tool(calculator)
```

### Strategy 3: Full Migration with Tools

**Best for:** New implementations or complete rewrites.

```python
from ace.generator.react_generator import ReActGenerator

# Define all tools upfront
def search_db(query: str, limit: int = 5) -> list[str]:
    """Search vector database."""
    # Implementation
    return results

def rank_results(results: list[str], criteria: str = "relevance") -> list[str]:
    """Rank results."""
    # Implementation
    return sorted_results

# Initialize with tools
agent = ReActGenerator(
    tools=[search_db, rank_results],
    model="gpt-4o-mini",
    max_iters=10
)
```

---

## Step-by-Step Migration

### Step 1: Update Imports

```python
# Old
from ace.generator.cot_generator import CoTGenerator

# New
from ace.generator.react_generator import ReActGenerator
```

### Step 2: Replace Class Name

```python
# Old
agent = CoTGenerator(model="gpt-4")

# New
agent = ReActGenerator(model="gpt-4")
```

### Step 3: (Optional) Add Tools

Only if you want tool-calling capabilities:

```python
# Define tools with type annotations
def my_tool(param: str) -> str:
    """Tool description."""
    return "result"

# Register during init
agent = ReActGenerator(
    tools=[my_tool],
    model="gpt-4"
)

# Or register after init
agent.register_tool(my_tool)
```

### Step 4: Update Task Execution (Optional)

Leverage new tool-specific features:

```python
from ace.generator.signatures import TaskInput

task = TaskInput(
    task_id="task-001",
    description="Find and rank ML papers",
    domain="ml-research",
    playbook_bullets=[],
    available_tools=["search_db", "rank_results"],  # NEW: Restrict tools
    max_iterations=15  # NEW: Task-level max iters
)

output = agent.forward(task)

# Access new fields
print(f"Tools used: {output.tools_used}")
print(f"Iterations: {output.total_iterations}")
print(f"Performance: {output.metadata['performance']}")
```

### Step 5: Update Reflector/Curator (Optional)

If using ACE cycle, update to handle tool metadata:

```python
from ace.reflector.signatures import ReflectorInput

reflector_input = ReflectorInput(
    task_id=output.task_id,
    reasoning_trace=output.reasoning_trace,
    answer=output.answer,
    confidence=output.confidence,
    # NEW: Tool-specific fields
    tools_used=output.tools_used,
    total_iterations=output.total_iterations,
    structured_trace=output.structured_trace,
    # ... other fields
)
```

---

## Common Patterns

### Pattern 1: RAG Agent with Database Tools

```python
from ace.generator.react_generator import ReActGenerator

def search_vector_db(query: str, k: int = 5) -> list[str]:
    """Search vector database."""
    # Your vector search implementation
    return documents

def search_sql_db(table: str, filters: dict) -> list[dict]:
    """Search SQL database."""
    # Your SQL query implementation
    return rows

def rank_results(results: list, criteria: str = "relevance") -> list:
    """Rank results by criteria."""
    # Your ranking implementation
    return sorted_results

# Create RAG agent
rag_agent = ReActGenerator(
    tools=[search_vector_db, search_sql_db, rank_results],
    model="gpt-4o-mini",
    max_iters=10
)

# Use with playbook strategies
task = TaskInput(
    task_id="rag-001",
    description="Find top papers about transformers",
    domain="ml-research",
    playbook_bullets=[
        "Use vector search for semantic similarity",
        "Filter SQL by date range",
        "Rank by citation count"
    ]
)

output = rag_agent.forward(task)
```

### Pattern 2: Multi-Tool Workflow with Error Handling

```python
from ace.generator.react_generator import ReActGenerator

def api_fetch(endpoint: str) -> dict:
    """Fetch from API."""
    # May fail with timeout
    response = requests.get(endpoint, timeout=5)
    return response.json()

def backup_fetch(endpoint: str) -> dict:
    """Backup API fetch."""
    # More reliable alternative
    response = requests.get(backup_url + endpoint, timeout=10)
    return response.json()

def process_data(data: dict) -> str:
    """Process API response."""
    return str(data)

agent = ReActGenerator(
    tools=[api_fetch, backup_fetch, process_data],
    model="gpt-4"
)

# Agent automatically:
# - Tries api_fetch first
# - Falls back to backup_fetch on failure
# - Excludes api_fetch after 3+ failures
# - Provides error context with suggestions
```

### Pattern 3: Learning from Tool Usage

```python
from ace.generator.react_generator import ReActGenerator
from ace.reflector.grounded_reflector import GroundedReflector
from ace.curator.semantic_curator import SemanticCurator

agent = ReActGenerator(tools=[tool1, tool2, tool3])
reflector = GroundedReflector()
curator = SemanticCurator()

playbook = []

for task in tasks:
    # Add learned strategies to task
    task.playbook_bullets = [b.content for b in playbook]

    # Execute
    output = agent.forward(task)

    # Reflect on tool usage
    reflector_input = ReflectorInput(
        task_id=output.task_id,
        reasoning_trace=output.reasoning_trace,
        answer=output.answer,
        confidence=output.confidence,
        tools_used=output.tools_used,  # Tool usage pattern
        total_iterations=output.total_iterations,
        # ... other fields
    )
    reflector_output = reflector(reflector_input)

    # Curate insights
    curator_input = CuratorInput(
        task_id=output.task_id,
        domain_id=task.domain,
        insights=[{
            "content": i.content,
            "section": i.section.value,
            "tags": i.tags,
            "tool_sequence": i.tool_sequence,  # NEW
            "tool_success_rate": i.tool_success_rate,  # NEW
        } for i in reflector_output.insights],
        current_playbook=playbook,
        target_stage=PlaybookStage.SHADOW,
    )
    curator_output = curator.apply_delta(curator_input)
    playbook = curator_output.updated_playbook

# After 20+ tasks, agent uses learned tool patterns!
```

---

## Troubleshooting

### Issue 1: ToolValidationError

**Error:**
```
ToolValidationError: Tool validation failed: Tool 'my_tool' parameter 'x' missing type annotation
```

**Solution:** Add type annotations to all tool parameters:

```python
# ❌ Bad
def my_tool(x):
    return x

# ✅ Good
def my_tool(x: str) -> str:
    return x
```

### Issue 2: DuplicateToolError

**Error:**
```
DuplicateToolError: Tool 'search' already registered
```

**Solution:** Use unique tool names or check before registering:

```python
# Check before registering
if "search" not in agent.tools:
    agent.register_tool(search)

# Or use unique names
def search_vector(query: str) -> list[str]:
    pass

def search_sql(query: str) -> list[dict]:
    pass
```

### Issue 3: Tools Not Being Used

**Problem:** Agent doesn't call registered tools.

**Solution:**

1. **Check tool availability:**
```python
print(f"Registered tools: {list(agent.tools.keys())}")
```

2. **Check tool exclusion:**
```python
print(f"Excluded tools: {agent.excluded_tools}")
print(f"Failure counts: {agent.tool_failure_counts}")

# Reset failures if needed
agent.reset_tool_failures()
```

3. **Check task restrictions:**
```python
# Ensure available_tools includes your tools
task.available_tools = ["search", "rank"]  # or None for all
```

### Issue 4: Performance Degradation

**Problem:** ReActGenerator is slower than CoTGenerator.

**Solution:**

1. **Check tool execution time:**
```python
stats = agent.get_performance_stats()
print(f"Tool call overhead: {stats['tool_call_overhead_ms']}")
```

2. **Reduce max iterations:**
```python
agent = ReActGenerator(tools=tools, max_iters=5)  # Lower limit
```

3. **Use faster tools:**
```python
# Add timeout to slow tools
def slow_tool(x: str) -> str:
    result = expensive_operation(x, timeout=1)  # 1 second max
    return result
```

4. **Enable LRU caching:**
```python
# Already enabled by default (maxsize=128)
# Clear cache if playbook updates:
agent.clear_strategy_cache()
```

### Issue 5: Missing Tool Context in Errors

**Problem:** Tool errors don't provide enough information.

**Solution:** Use structured logging:

```python
from ace.utils.logging_config import get_logger

# Logs are already enabled in ReActGenerator (T057)
# Check logs for detailed tool execution context:
# - tool_execution_start
# - tool_execution_success
# - tool_argument_error
# - tool_execution_error
# - tool_timeout
# - tool_excluded
```

---

## Performance Comparison

| Metric | CoTGenerator | ReActGenerator (no tools) | ReActGenerator (with tools) |
|--------|--------------|---------------------------|----------------------------|
| Initialization | <100ms | <100ms | <500ms (50 tools) |
| Reasoning overhead | ~10ms | ~10ms | ~10ms |
| Tool call overhead | N/A | N/A | <100ms per iteration |
| Memory usage | Low | Low | +5MB per 100 tools |
| Playbook retrieval | ~5ms | ~5ms (cached) | ~5ms (cached) |

---

## Migration Checklist

- [ ] Update imports from `CoTGenerator` to `ReActGenerator`
- [ ] Replace class instantiation
- [ ] (Optional) Define tools with type annotations
- [ ] (Optional) Register tools during or after init
- [ ] (Optional) Update TaskInput with tool-specific fields
- [ ] (Optional) Update Reflector/Curator for tool metadata
- [ ] Run backward compatibility tests
- [ ] Run performance benchmarks
- [ ] Update documentation and examples
- [ ] Train team on new tool-calling features

---

## Next Steps

1. **Test in isolation:** Start with `ReActGenerator(model="gpt-4")` (no tools)
2. **Add one tool:** Gradually introduce tool capabilities
3. **Monitor performance:** Use `agent.get_performance_stats()`
4. **Review logs:** Check structured logs for tool execution details
5. **Optimize:** Adjust max_iters, tool timeouts, and caching as needed

**Need help?** See:
- [README.md](../README.md) - Quick start guide
- [examples/multi_tool_orchestration.py](../examples/multi_tool_orchestration.py) - Working examples
- [tests/integration/test_backward_compat.py](../tests/integration/test_backward_compat.py) - Compatibility tests

---

**Migration complete!** 🎉 You're now ready to build powerful tool-calling agents with ACE playbook learning.
