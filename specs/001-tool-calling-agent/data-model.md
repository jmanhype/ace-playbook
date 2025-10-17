# Data Model: Tool-Calling Agent with ReAct Reasoning

**Feature**: 001-tool-calling-agent
**Date**: October 16, 2025
**Version**: 1.0.0

## Overview

Data structures and entities for ReAct-based tool-calling agents integrated with ACE's playbook learning system.

---

## Core Entities

### 1. Tool

Represents a callable function or API that the agent can use during reasoning.

**Fields**:
- `name`: `str` - Unique identifier for the tool (e.g., "search_database")
- `function`: `Callable` - The actual callable (function or object with `__call__`)
- `description`: `str` - Human-readable description of what the tool does (used in prompts)
- `parameters`: `Dict[str, Type]` - Parameter names and their types (extracted from signature)
- `return_type`: `Optional[Type]` - Expected return type (if annotated)
- `timeout_seconds`: `int` - Maximum execution time (default: 30)
- `success_count`: `int` - Number of successful invocations (tracked for analytics)
- `failure_count`: `int` - Number of failed invocations (tracked for analytics)

**Validation Rules**:
- `name` must be unique within an agent instance
- `function` must be callable with at least one parameter
- All parameters must have type annotations
- `description` recommended but not required (defaults to function docstring)

**Relationships**:
- Used by: `ReActGenerator` (registered tools available for task execution)
- Referenced in: `ToolCall` (which tool was invoked)
- Tracked in: `ToolCallingStrategy` (which tools appear in successful sequences)

**State Transitions**:
```
Registered → Available → (In Use ↔ Failed ↔ Succeeded) → Deprecated
```

---

### 2. ReasoningStep

Represents one iteration of the ReAct cycle: thought → action → observation.

**Fields**:
- `iteration`: `int` - Step number in the reasoning sequence (1-indexed)
- `thought`: `str` - Agent's internal reasoning about the current situation
- `action`: `Literal["call_tool", "finish"]` - What the agent decided to do
- `tool_name`: `Optional[str]` - Name of tool to call (if action is "call_tool")
- `tool_args`: `Optional[Dict[str, Any]]` - Arguments passed to the tool
- `observation`: `Optional[str]` - Result from tool execution or final answer
- `timestamp`: `float` - Unix timestamp when step occurred
- `duration_ms`: `float` - How long this step took (for performance tracking)

**Validation Rules**:
- `iteration` must be positive integer
- If `action` is "call_tool", `tool_name` and `tool_args` must be provided
- If `action` is "finish", `observation` should contain the final answer
- `thought` should not be empty (agent must reason before acting)

**Relationships**:
- Contained in: `TaskOutput.structured_trace` (list of reasoning steps)
- Analyzed by: `GroundedReflector` (to extract tool usage patterns)

**Example**:
```python
ReasoningStep(
    iteration=1,
    thought="I need to find recent sales data for Q3",
    action="call_tool",
    tool_name="search_database",
    tool_args={"table": "sales", "filters": {"quarter": "Q3"}},
    observation="Found 1,250 sales records",
    timestamp=1729123456.789,
    duration_ms=45.2
)
```

---

### 3. ToolCall

Represents a single invocation of a tool within a reasoning trace.

**Fields**:
- `tool_name`: `str` - Name of the tool that was called
- `arguments`: `Dict[str, Any]` - Parameters passed to the tool
- `result`: `Any` - Value returned by the tool
- `success`: `bool` - Whether the call succeeded (True) or failed/timed out (False)
- `error_message`: `Optional[str]` - Error details if call failed
- `duration_ms`: `float` - Execution time in milliseconds

**Validation Rules**:
- `tool_name` must match a registered Tool.name
- `arguments` must match the tool's parameter signature
- If `success` is False, `error_message` should explain why

**Relationships**:
- Created during: `ReasoningStep` execution
- Extracted from: `TaskOutput.structured_trace`
- Used by: `Reflector` to identify tool usage patterns

---

### 4. ToolCallingStrategy (extends PlaybookBullet)

A playbook bullet that captures proven patterns for tool usage.

**Fields** (in addition to standard PlaybookBullet):
- `content`: `str` - Description of the strategy (inherited from PlaybookBullet)
- `section`: `str` - "Strategies", "Pitfalls", or "Observations" (inherited)
- `helpful_count`: `int` - Times this strategy led to success (inherited)
- `harmful_count`: `int` - Times following this caused failure (inherited)
- `tags`: `List[str]` - Domain/context tags (inherited)
- **`tool_sequence`**: `List[str]` - Ordered list of tool names (e.g., ["search_db", "filter", "format"])
- **`tool_success_rate`**: `float` - Percentage of times this sequence succeeded (0.0-1.0)
- **`avg_iterations`**: `int` - Average number of iterations when this pattern used
- **`example_task`**: `str` - Sample task where this strategy worked

**Validation Rules**:
- `tool_sequence` must contain at least one tool name
- All tools in sequence must exist (validated against registered tools)
- `tool_success_rate` must be between 0.0 and 1.0
- If `helpful_count` > 0, `tool_success_rate` should be > 0.5

**Relationships**:
- Stored in: Playbook (`.specify/memory/playbooks/tool-calling/`)
- Created by: `GroundedReflector` (from successful task outcomes)
- Used by: `ReActGenerator` (injected as context for tool selection)
- Merged by: `SemanticCurator` (deduplication and consolidation)

**Example**:
```python
ToolCallingStrategy(
    content="For database queries about recent data, apply temporal filter first, then search vector DB, then rank results",
    section="Strategies",
    helpful_count=15,
    harmful_count=2,
    tags=["rag", "database", "temporal"],
    tool_sequence=["temporal_filter", "search_vector_db", "rank_results"],
    tool_success_rate=0.88,
    avg_iterations=3,
    example_task="Find Q3 sales data for enterprise customers"
)
```

---

### 5. TaskInput (existing, extends for ReAct)

Input signature for ReActGenerator, compatible with CoTGenerator.

**Fields**:
- `task_id`: `str` - Unique identifier for this task
- `description`: `str` - The task/question to solve
- `playbook_bullets`: `List[str]` - Relevant strategies from playbook (existing)
- `domain`: `str` - Task domain for playbook retrieval (existing)
- `metadata`: `Dict[str, Any]` - Additional context (existing)
- **`available_tools`**: `Optional[List[str]]` - Tool names to use (NEW, optional)
- **`max_iterations`**: `Optional[int]` - Override default iteration limit (NEW, optional)

**Validation Rules**:
- `task_id` must be unique within a batch
- `description` cannot be empty
- If `available_tools` provided, all must match registered tools
- `max_iterations` must be positive if provided

**Relationships**:
- Input to: `ReActGenerator.forward()`
- Contains: Playbook bullets (retrieved from Playbook)

---

### 6. TaskOutput (existing, extends for ReAct)

Output signature for ReActGenerator, compatible with CoTGenerator.

**Fields**:
- `task_id`: `str` - Matches input task_id (existing)
- `answer`: `str` - Final answer/result (existing)
- `reasoning_trace`: `str` - Human-readable summary of reasoning (existing)
- `confidence`: `float` - Agent's confidence in answer, 0.0-1.0 (existing)
- `metadata`: `Dict[str, Any]` - Additional info (existing)
- **`structured_trace`**: `List[ReasoningStep]` - Detailed reasoning steps (NEW)
- **`tools_used`**: `List[str]` - Names of tools called during execution (NEW)
- **`total_iterations`**: `int` - How many ReAct iterations occurred (NEW)
- **`iteration_limit_reached`**: `bool` - True if max_iters hit without finishing (NEW)

**Validation Rules**:
- `answer` should not be empty (unless iteration limit reached)
- `confidence` must be between 0.0 and 1.0
- `structured_trace` length should equal `total_iterations`
- If `iteration_limit_reached` is True, confidence should typically be lower

**Relationships**:
- Output from: `ReActGenerator.forward()`
- Input to: `GroundedReflector` (for analysis and bullet creation)
- Contains: List of `ReasoningStep` entities

---

## Entity Relationships Diagram

```
┌─────────────┐
│    Tool     │ (registered with agent)
└─────┬───────┘
      │ referenced in
      ↓
┌─────────────┐     contains     ┌──────────────────┐
│  ToolCall   │◄─────────────────│  ReasoningStep   │
└─────────────┘                  └────────┬─────────┘
                                          │ part of
                                          ↓
┌──────────────┐    input to   ┌──────────────────┐
│  TaskInput   │──────────────→│ ReActGenerator   │
└──────────────┘                └────────┬─────────┘
                                         │ produces
                                         ↓
┌─────────────────────┐    output    ┌──────────────────┐
│ ToolCallingStrategy │◄─────────────│   TaskOutput     │
│ (PlaybookBullet)    │   analyzed   └────────┬─────────┘
└──────────┬──────────┘      by               │
           │                                   ↓
           │ retrieved by              ┌──────────────────┐
           └──────────────────────────→│ GroundedReflector│
                  context               └──────────────────┘
```

---

## Data Flow

### 1. Task Execution Flow

```
1. TaskInput created with task description
2. ReActGenerator retrieves ToolCallingStrategy from Playbook (via domain)
3. Strategies formatted as context, injected into ReAct prompt
4. ReAct iterations: thought → action → tool_call → observation
5. Each iteration creates ReasoningStep
6. TaskOutput generated with structured_trace and tools_used
```

### 2. Reflection Flow

```
1. TaskOutput analyzed by GroundedReflector
2. Tool sequence extracted from structured_trace
3. If successful (confidence > threshold):
   - Create ToolCallingStrategy with helpful_count=1
   - Include tool_sequence, success_rate, avg_iterations
4. If failed:
   - Create ToolCallingStrategy with harmful_count=1
   - Mark as anti-pattern
5. SemanticCurator merges strategies (deduplicate similar sequences)
6. Updated strategies stored in Playbook
```

### 3. Learning Flow

```
1. New task arrives with same domain
2. Playbook retrieves top-3 ToolCallingStrategy by similarity
3. Strategies with high helpful_count prioritized
4. Tool sequences from strategies guide ReAct tool selection
5. Agent starts with proven tools, reducing trial-and-error
6. Result: 30-50% fewer iterations (SC-002 success criteria)
```

---

## Storage Considerations

### Playbook Storage Schema

**Location**: `.specify/memory/playbooks/tool-calling/{domain}/`

**File Format**: JSON Lines (one strategy per line)

```json
{
  "content": "For database queries about recent data...",
  "section": "Strategies",
  "helpful_count": 15,
  "harmful_count": 2,
  "tags": ["rag", "database", "temporal"],
  "tool_sequence": ["temporal_filter", "search_vector_db", "rank_results"],
  "tool_success_rate": 0.88,
  "avg_iterations": 3,
  "example_task": "Find Q3 sales data for enterprise customers",
  "created_at": "2025-10-16T10:30:00Z",
  "last_updated": "2025-10-16T15:45:00Z"
}
```

### FAISS Index Schema

**Embedding Target**: `ToolCallingStrategy.content` (same as existing bullets)

**Metadata Stored**:
- `has_tool_sequence`: `bool` (filter for tool strategies)
- `domain`: `str` (filter by task domain)
- `tool_count`: `int` (number of tools in sequence)
- `success_rate`: `float` (for ranking/filtering)

### Reasoning Trace Storage

**Location**: Temporary (in-memory during execution)

**Persistence**: Optional logging to `.specify/logs/reasoning-traces/{task_id}.json` for debugging

**Retention**: 7 days for debugging, then purge (not needed for learning - strategies are extracted)

---

## Backward Compatibility

### CoTGenerator Compatibility

- TaskInput/TaskOutput signatures remain compatible
- New fields are optional (default to None)
- Existing code using CoTGenerator continues working
- ReActGenerator can be used as drop-in replacement where tools not needed

### Playbook Compatibility

- Existing PlaybookBullet entries without tool_sequence work unchanged
- Tool strategies are additive (don't affect non-tool bullets)
- SemanticCurator handles both old and new bullet formats
- No migration required for existing playbooks

---

## Versioning

**Data Model Version**: 1.0.0

**Semantic Versioning**:
- MAJOR: Breaking changes to entity structure (e.g., remove required field)
- MINOR: Add optional fields (e.g., new metadata in ToolCallingStrategy)
- PATCH: Documentation updates, validation rule clarifications

**Current Version Features**:
- ReActGenerator with tool-calling support
- Extended PlaybookBullet for tool strategies
- Structured reasoning traces
- Backward compatible with existing ACE components

---

## Next Steps

Data model complete. Ready for:
1. **API Contracts**: Define REST/Python API interfaces in `/contracts/`
2. **Quickstart Guide**: Generate developer onboarding doc
3. **Agent Context Update**: Add technology decisions to Claude/Cursor context files
