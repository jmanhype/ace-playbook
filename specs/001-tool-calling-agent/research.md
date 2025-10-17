# Research: Tool-Calling Agent with ReAct Reasoning

**Feature**: 001-tool-calling-agent
**Date**: October 16, 2025
**Status**: Complete

## Overview

Research findings for implementing ReAct (Reasoning and Acting) agent support in ACE framework, enabling tool-calling workflows with playbook-based learning.

---

## Decision 1: DSPy ReAct Module Integration

**Decision**: Use DSPy's built-in `dspy.ReAct` module as the foundation for ReActGenerator

**Rationale**:
- DSPy already provides mature ReAct implementation with tool-calling support
- Maintains consistency with existing CoTGenerator (also DSPy-based)
- Signature polymorphism allows any task signature while preserving tool-calling capability
- Well-documented API with proven patterns (from dspy.ai documentation)

**Alternatives Considered**:
1. **Custom ReAct implementation from scratch**
   - Rejected: Reinventing the wheel, high maintenance burden, delays delivery
   - Would require implementing: thought generation, tool selection logic, iteration management

2. **LangChain ReAct agent**
   - Rejected: Different framework than ACE's DSPy foundation, integration complexity
   - Would break architectural consistency (Principle V: Modular Black-Box Architecture)

**Implementation Approach**:
```python
import dspy

class ReActGenerator(dspy.Module):
    def __init__(self, tools=None, model=None, max_iters=10):
        super().__init__()
        self.react = dspy.ReAct(
            "task, context -> answer, reasoning_trace",
            tools=tools or [],
            max_iters=max_iters
        )
        if model:
            dspy.settings.configure(lm=dspy.LM(model))
```

---

## Decision 2: Tool Signature Validation Strategy

**Decision**: Implement runtime tool signature validation using Python's `inspect` module before allowing tools to be registered

**Rationale**:
- Prevents runtime errors from malformed tool functions
- Provides clear error messages during agent initialization (not mid-execution)
- Aligns with Principle V (Black-Box Architecture) - validate interfaces upfront
- Enables better developer experience (fail fast with helpful error messages)

**Alternatives Considered**:
1. **Static type checking only (mypy/pyright)**
   - Rejected: Doesn't catch runtime tool registration issues
   - Developers might pass tools dynamically that type checkers can't validate

2. **No validation (duck typing)**
   - Rejected: Leads to cryptic runtime errors during agent execution
   - Violates UX playbook principle (clear error messages)

**Validation Rules**:
- Tool must be callable (function or object with `__call__`)
- Tool must have explicit parameter annotations (for DSPy to generate prompts)
- Tool should have docstring describing purpose (used in ReAct prompts)
- Return type annotation recommended but not required

**Implementation Pattern**:
```python
import inspect
from typing import Callable, Any

def validate_tool(tool: Callable) -> None:
    """Validate tool has proper signature for ReAct use."""
    if not callable(tool):
        raise TypeError(f"Tool must be callable, got {type(tool)}")

    sig = inspect.signature(tool)
    if not sig.parameters:
        raise ValueError(f"Tool {tool.__name__} must have at least one parameter")

    for param in sig.parameters.values():
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Tool {tool.__name__} parameter '{param.name}' missing type annotation"
            )
```

---

## Decision 3: Maximum Iterations Configuration (Hybrid Override System)

**Decision**: Implement 3-level configuration hierarchy: system default → per-agent override → per-task override

**Rationale**:
- Provides flexibility for different use cases (simple tasks vs complex workflows)
- System default (10 iterations) prevents infinite loops as safe fallback
- Per-agent overrides allow specialized agents (e.g., RAG agent gets 15, orchestrator gets 25)
- Per-task overrides enable fine-tuning for known complex queries
- Aligns with specification requirement FR-005 (hybrid approach selected)

**Alternatives Considered**:
1. **Single system-wide default**
   - Rejected: Too rigid, forces all agents to same limit

2. **Only per-task configuration**
   - Rejected: Developer burden to specify for every task

3. **Adaptive limits based on task complexity**
   - Deferred to future: Requires ML model to predict complexity, adds overhead

**Configuration Precedence**:
```
Task-level (if provided) > Agent-level (if set) > System default (10)
```

**Implementation**:
```python
class ReActGenerator(dspy.Module):
    def __init__(self, tools=None, model=None, max_iters=None):
        super().__init__()
        # Agent-level override (defaults to None, uses system default)
        self.max_iters_override = max_iters

    def forward(self, task: TaskInput, max_iters: int = None) -> TaskOutput:
        # Task-level override takes precedence
        effective_max_iters = max_iters or self.max_iters_override or 10  # system default

        self.react = dspy.ReAct(
            "task, context -> answer, reasoning_trace",
            tools=self.tools,
            max_iters=effective_max_iters
        )
        # ... execution
```

---

## Decision 4: Tool-Calling Strategy Metadata Schema

**Decision**: Extend existing PlaybookBullet with optional `tool_sequence` metadata field for tool-calling strategies

**Rationale**:
- Backward compatible - existing bullets without tool metadata still work
- Enables Reflector to capture which tool sequences led to success
- Curator can merge similar tool sequences using semantic similarity
- Supports future analysis of tool usage patterns (which tools work together)

**Alternatives Considered**:
1. **Separate ToolStrategy dataclass**
   - Rejected: Creates two playbook systems, violates DRY principle
   - Harder to maintain consistency between code quality and tool strategies

2. **Tool sequence in bullet content only (unstructured)**
   - Rejected: Loses structured querying capability, hard to analyze patterns

**Schema Extension**:
```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PlaybookBullet:
    content: str
    section: str  # "Strategies", "Pitfalls", "Observations"
    helpful_count: int = 0
    harmful_count: int = 0
    tags: list = field(default_factory=list)

    # NEW: Tool-calling metadata (optional)
    tool_sequence: Optional[List[str]] = None  # e.g., ["search_db", "filter_results", "format_output"]
    tool_success_rate: Optional[float] = None  # Track effectiveness
    avg_iterations: Optional[int] = None       # How many iterations this pattern took
```

**Reflector Integration**:
- After ReAct task completion, extract tool sequence from reasoning trace
- If task succeeded, create bullet with `tool_sequence` and mark as Helpful
- If task failed, create bullet with `tool_sequence` and mark as Harmful
- Curator deduplicates using both content similarity AND tool sequence similarity

---

## Decision 5: Playbook Context Injection for Tool Selection

**Decision**: Inject tool-calling strategies from playbook as context in DSPy ReAct prompt to guide tool selection

**Rationale**:
- Enables agents to learn from past successful tool sequences
- Reduces trial-and-error in tool selection (directly addresses SC-002: 30-50% iteration reduction)
- Maintains Generator-Reflector-Curator separation (Generator uses playbook, doesn't modify it)
- Aligns with ACE's core learning pattern

**Alternatives Considered**:
1. **Fine-tune model on tool sequences**
   - Rejected: Expensive, requires retraining, loses interpretability

2. **Hard-coded tool selection heuristics**
   - Rejected: Brittle, doesn't adapt to new patterns, defeats ACE's learning purpose

**Context Injection Pattern**:
```python
def forward(self, task: TaskInput) -> TaskOutput:
    # Retrieve relevant tool-calling strategies from playbook
    tool_strategies = self.playbook.retrieve(
        query=task.description,
        domain=task.domain,
        filters={"has_tool_sequence": True},
        k=3  # Top 3 relevant strategies
    )

    # Format strategies as context
    context_bullets = []
    for strategy in tool_strategies:
        if strategy.tool_sequence:
            tools_str = " → ".join(strategy.tool_sequence)
            context_bullets.append(f"• {strategy.content} (Pattern: {tools_str})")
        else:
            context_bullets.append(f"• {strategy.content}")

    context = "\n".join(context_bullets)

    # Inject into ReAct prompt
    result = self.react(
        task=task.description,
        context=context  # Learned strategies guide tool selection
    )
```

**Benefit**: Agent sees "For database queries about recent data, try temporal_filter → search_vector_db → rank_results" and applies this pattern, reducing iterations

---

## Decision 6: Reasoning Trace Format

**Decision**: Extend TaskOutput with structured reasoning trace capturing thoughts, tool calls, and observations

**Rationale**:
- Enables Reflector to analyze exactly which tools were used and why
- Supports debugging by developers (readable trace of agent's reasoning)
- Required for playbook learning (can't learn from what you don't capture)
- Aligns with Principle II (Reflection-Based Testing - execution feedback)

**Alternatives Considered**:
1. **Unstructured string trace**
   - Rejected: Hard to parse for Reflector, loses tool call metadata

2. **Separate trace object**
   - Rejected: Complicates interface, requires clients to manage two outputs

**Trace Structure**:
```python
from dataclasses import dataclass
from typing import List, Any

@dataclass
class ReasoningStep:
    iteration: int
    thought: str                    # Agent's reasoning
    action: str                     # "call_tool" or "finish"
    tool_name: Optional[str]        # If action is "call_tool"
    tool_args: Optional[dict]       # Tool parameters
    observation: Optional[Any]      # Tool result or final answer
    timestamp: float

@dataclass
class TaskOutput:
    task_id: str
    answer: str
    reasoning_trace: str                      # Human-readable summary (existing)
    structured_trace: List[ReasoningStep]     # NEW: Structured for Reflector
    confidence: float
    metadata: dict
```

**Reflector Usage**:
```python
def analyze_tool_usage(self, output: TaskOutput) -> List[PlaybookBullet]:
    bullets = []

    # Extract tool sequence
    tool_sequence = [
        step.tool_name
        for step in output.structured_trace
        if step.action == "call_tool"
    ]

    # Create strategy bullet
    if output.confidence > 0.8:  # Success
        bullet = PlaybookBullet(
            content=f"For tasks like '{output.task_id}', use tool sequence: {' → '.join(tool_sequence)}",
            section="Strategies",
            helpful_count=1,
            tool_sequence=tool_sequence,
            tool_success_rate=output.confidence
        )
        bullets.append(bullet)

    return bullets
```

---

## Decision 7: Performance Optimization Strategy

**Decision**: Implement lazy playbook retrieval caching with LRU eviction for frequently accessed tool strategies

**Rationale**:
- Addresses performance budget: <10ms P50 playbook retrieval
- Reduces FAISS calls for repeated similar tasks (hot path optimization)
- Aligns with Principle IV (Performance with Continuous Optimization)
- Simple implementation using `functools.lru_cache`

**Alternatives Considered**:
1. **Redis caching layer**
   - Deferred to future: Adds infrastructure dependency, overkill for initial version

2. **Pre-load all playbook bullets into memory**
   - Rejected: Memory footprint grows unbounded, defeats vector search benefits

3. **No caching**
   - Rejected: Likely misses <10ms P50 target for playbook retrieval

**Implementation**:
```python
from functools import lru_cache

class ReActGenerator(dspy.Module):
    @lru_cache(maxsize=128)  # Cache last 128 playbook retrievals
    def _get_tool_strategies(self, query: str, domain: str) -> List[PlaybookBullet]:
        return self.playbook.retrieve(
            query=query,
            domain=domain,
            filters={"has_tool_sequence": True},
            k=3
        )
```

**Cache Invalidation**: Clear cache when playbook updated (after Curator merges new bullets)

**Expected Impact**: 80%+ cache hit rate for similar tasks, reducing average retrieval from ~8ms to ~0.5ms (in-memory lookup)

---

## Decision 8: Tool Failure Handling Strategy

**Decision**: Implement graceful degradation with fallback strategies when tools fail or timeout

**Rationale**:
- Addresses edge case: "What happens when a tool call times out or fails?"
- Improves SC-005 (95% success in finding alternatives when primary tools fail)
- Prevents agent from crashing on tool errors (resilience)
- Captures failure patterns in playbook (Harmful bullets)

**Alternatives Considered**:
1. **Fail fast on any tool error**
   - Rejected: Brittle, defeats ReAct's iterative nature

2. **Retry with exponential backoff**
   - Deferred: Adds complexity, better suited for network errors not logic errors

**Failure Handling Tiers**:
1. **Timeout**: Set per-tool timeout (default 30s from spec assumptions), return error observation
2. **Exception**: Catch exceptions, format as error observation, continue reasoning
3. **Fallback**: If tool fails repeatedly (>3 times), exclude from available tools for this task
4. **Playbook Learning**: Record failed tool sequences as Harmful bullets

**Implementation**:
```python
import signal
from contextlib import contextmanager

@contextmanager
def tool_timeout(seconds=30):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Tool execution exceeded {seconds}s")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def execute_tool_safely(tool: Callable, args: dict) -> str:
    try:
        with tool_timeout(30):
            result = tool(**args)
            return f"Success: {result}"
    except TimeoutError as e:
        return f"Error: Tool timed out after 30s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
```

**Reflector Integration**: Failed tool calls generate Harmful bullets with `tool_sequence` showing what NOT to do

---

## Best Practices Integration

### DSPy ReAct Module Best Practices
- Always provide tool descriptions (used in LLM prompts to guide selection)
- Use type-annotated parameters (enables DSPy to generate better prompts)
- Keep tools focused (single responsibility, easier for agent to reason about)
- Tool names should be descriptive verbs (e.g., `search_database`, not `db_query`)

### ACE Framework Integration Best Practices
- Generator is stateless except for playbook reference (supports parallel execution)
- Reflector runs asynchronously after task completion (doesn't block agent)
- Curator updates are atomic (no race conditions on playbook updates)
- Playbook bullets include attribution (which task/domain generated the insight)

### Testing Strategy Best Practices
- Unit tests: Mock tools to test ReActGenerator logic in isolation
- Integration tests: Real tools (in-memory databases) to test full workflow
- Performance tests: Benchmark with 10, 50, 100 tools to validate scaling
- Contract tests: Validate tool signatures before deploying new tools

---

## Open Questions (Resolved)

All technical uncertainties from spec.md have been resolved through research:

1. ✅ **How to integrate ReAct with existing ACE architecture?**
   - Use DSPy's ReAct module, inject playbook strategies as context

2. ✅ **How to capture tool-calling patterns in playbook?**
   - Extend PlaybookBullet with optional tool_sequence metadata

3. ✅ **How to handle max iterations configuration?**
   - Hybrid override system: task > agent > system default (10)

4. ✅ **How to optimize playbook retrieval performance?**
   - LRU cache for frequent queries, clear on updates

5. ✅ **How to handle tool failures gracefully?**
   - Timeouts, exception handling, fallback strategies, playbook learning

---

## Next Steps

Ready to proceed to **Phase 1: Design & Contracts**:
1. Generate `data-model.md` with entity definitions
2. Create API contracts in `/contracts/`
3. Generate `quickstart.md` for developers
4. Update agent context files with new technology decisions

All unknowns from Technical Context section have been resolved. No NEEDS CLARIFICATION markers remain.
