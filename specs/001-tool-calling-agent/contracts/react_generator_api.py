"""
API Contract: ReActGenerator

Defines the Python API interface for the ReAct-based tool-calling agent.
This contract ensures forward compatibility and serves as the specification
for implementation.

Version: 1.0.0
"""

from typing import List, Optional, Callable, Any, Dict, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ============================================================================
# INPUT/OUTPUT SIGNATURES (extends existing ACE signatures)
# ============================================================================

@dataclass
class TaskInput:
    """
    Input specification for ReActGenerator.

    Compatible with existing CoTGenerator - new fields are optional.
    """
    task_id: str
    description: str
    playbook_bullets: List[str] = field(default_factory=list)
    domain: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW: ReAct-specific fields (optional for backward compatibility)
    available_tools: Optional[List[str]] = None  # Subset of registered tools to use
    max_iterations: Optional[int] = None         # Task-level override for iteration limit


@dataclass
class ReasoningStep:
    """
    One iteration of the ReAct cycle: thought → action → observation.
    """
    iteration: int
    thought: str
    action: Literal["call_tool", "finish"]
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: float = 0.0
    duration_ms: float = 0.0


@dataclass
class TaskOutput:
    """
    Output specification for ReActGenerator.

    Compatible with existing CoTGenerator - new fields are additive.
    """
    task_id: str
    answer: str
    reasoning_trace: str                          # Human-readable summary
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW: ReAct-specific fields
    structured_trace: List[ReasoningStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    total_iterations: int = 0
    iteration_limit_reached: bool = False


# ============================================================================
# TOOL INTERFACE
# ============================================================================

class Tool(ABC):
    """
    Abstract base class for tools that ReAct agents can use.

    All tools must implement this interface for proper validation and usage.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description (used in LLM prompts)."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given parameters.

        Raises:
            ToolExecutionError: If tool execution fails
            TimeoutError: If execution exceeds timeout
        """
        pass

    @property
    def timeout_seconds(self) -> int:
        """Maximum execution time (default: 30 seconds)."""
        return 30


# Simplified tool registration using functions
ToolFunction = Callable[..., Any]


# ============================================================================
# GENERATOR INTERFACE
# ============================================================================

class ReActGeneratorInterface(ABC):
    """
    Interface contract for ReActGenerator.

    Ensures implementation adheres to ACE's Generator pattern while
    adding tool-calling capabilities.
    """

    @abstractmethod
    def __init__(
        self,
        tools: Optional[List[ToolFunction]] = None,
        model: Optional[str] = None,
        max_iters: Optional[int] = None
    ):
        """
        Initialize ReAct generator.

        Args:
            tools: List of tool functions to register (validated on init)
            model: LLM model identifier (e.g., "gpt-4", "claude-3")
            max_iters: Agent-level max iterations override (default: None, uses system default 10)

        Raises:
            ToolValidationError: If any tool has invalid signature
        """
        pass

    @abstractmethod
    def forward(
        self,
        task: TaskInput,
        max_iters: Optional[int] = None
    ) -> TaskOutput:
        """
        Execute task with ReAct reasoning and tool calling.

        Args:
            task: Task specification (question, context, playbook bullets)
            max_iters: Task-level max iterations override (highest precedence)

        Returns:
            TaskOutput with answer, reasoning trace, and tool usage info

        Raises:
            MaxIterationsExceededError: If agent hits iteration limit without finishing
            ToolNotFoundError: If task.available_tools contains unregistered tool
        """
        pass

    @abstractmethod
    def register_tool(self, tool: ToolFunction) -> None:
        """
        Register a new tool after initialization.

        Args:
            tool: Function with type annotations and docstring

        Raises:
            ToolValidationError: If tool signature is invalid
            DuplicateToolError: If tool name already registered
        """
        pass

    @abstractmethod
    def validate_tools(self) -> List[str]:
        """
        Validate all registered tools have proper signatures.

        Returns:
            List of validation error messages (empty if all valid)
        """
        pass


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ToolValidationError(Exception):
    """Raised when tool signature is invalid."""
    pass


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class DuplicateToolError(Exception):
    """Raised when attempting to register tool with duplicate name."""
    pass


class ToolNotFoundError(Exception):
    """Raised when task references unregistered tool."""
    pass


class MaxIterationsExceededError(Exception):
    """Raised when agent hits max iterations without finishing task."""
    pass


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
Example usage of ReActGenerator API:

```python
from ace.generator import ReActGenerator, TaskInput

# Define tools
def search_database(query: str, table: str) -> str:
    '''Search database for records matching query.'''
    # ... implementation
    return "Found 10 results"

def filter_results(results: str, criteria: dict) -> str:
    '''Filter results based on criteria.'''
    # ... implementation
    return "Filtered to 3 results"

# Initialize generator
agent = ReActGenerator(
    tools=[search_database, filter_results],
    model="gpt-4",
    max_iters=15  # Agent-level override
)

# Execute task
task = TaskInput(
    task_id="task-001",
    description="Find recent sales data for Q3 2024",
    playbook_bullets=[
        "For database queries, apply temporal filter first",
        "Always verify result count before returning"
    ],
    domain="sales-analytics",
    max_iterations=20  # Task-level override (takes precedence)
)

output = agent.forward(task)

print(f"Answer: {output.answer}")
print(f"Tools used: {' → '.join(output.tools_used)}")
print(f"Iterations: {output.total_iterations}")
print(f"Confidence: {output.confidence}")

# Inspect reasoning
for step in output.structured_trace:
    print(f"[{step.iteration}] Thought: {step.thought}")
    if step.action == "call_tool":
        print(f"    → Called {step.tool_name} with {step.tool_args}")
        print(f"    → Observation: {step.observation}")
```

Output:
```
Answer: Found 1,250 sales records for Q3 2024 with total revenue of $5.2M
Tools used: temporal_filter → search_database → filter_results
Iterations: 3
Confidence: 0.92

[1] Thought: Need to find Q3 2024 data, should apply temporal filter as playbook suggests
    → Called temporal_filter with {'quarter': 'Q3', 'year': 2024}
    → Observation: Temporal scope set to Q3 2024
[2] Thought: Now search database with temporal scope applied
    → Called search_database with {'table': 'sales', 'query': '*'}
    → Observation: Found 1,250 matching records
[3] Thought: Results look complete, verify and finish
    → Called filter_results with {'results': '1,250 records', 'criteria': {'verify': True}}
    → Observation: Verified: 1,250 records, total revenue $5.2M
[4] Thought: Have complete answer with verification
    → Action: finish
    → Observation: Found 1,250 sales records for Q3 2024 with total revenue of $5.2M
```
"""


# ============================================================================
# COMPATIBILITY NOTES
# ============================================================================

"""
Backward Compatibility:
- ReActGenerator implements same TaskInput → TaskOutput interface as CoTGenerator
- New fields (available_tools, max_iterations, structured_trace, etc.) are optional
- Existing code using CoTGenerator can switch to ReActGenerator without changes
- If no tools provided, ReActGenerator behaves like CoTGenerator (reasoning only)

Forward Compatibility:
- Semantic versioning for interface (currently v1.0.0)
- Optional fields can be added in MINOR versions
- Required field changes require MAJOR version bump
- Tool interface is extensible (new methods can be added with defaults)

Migration Path:
1. CoTGenerator → ReActGenerator (no changes needed)
2. Add tools gradually (start with 1-2, expand to 10-50)
3. Enable playbook learning (Reflector automatically detects tool usage)
4. Monitor iteration reduction metrics (SC-002: 30-50% improvement expected)
"""
