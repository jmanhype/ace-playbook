"""ReAct-style generator implementation with lightweight tool support."""

from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Tuple

import dspy

from ace.generator.context_builder import build_strings
from ace.generator.cot_generator import TaskOutput
from ace.generator.signatures import ReActStepSignature, TaskInput
from ace.utils.llm_circuit_breaker import protected_predict
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="generator.react")


class ToolRegistry:
    """Simple mapping of tool names to callables."""

    def __init__(self, tools: Dict[str, Callable[[str], str]] | None = None):
        self._tools = {name.lower(): fn for name, fn in (tools or {}).items()}

    def execute(self, name: str, argument: str) -> str:
        handler = self._tools.get((name or "").lower())
        if not handler:
            raise ValueError(f"Unknown tool '{name}'")
        return handler(argument)

    def has_tool(self, name: str) -> bool:
        return (name or "").lower() in self._tools


class ReActGenerator:
    """LLM-driven ReAct agent with deterministic tool invocation."""

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.2,
        max_steps: int = 12,
        tool_registry: ToolRegistry | None = None,
        instructions_no_tools: str | None = None,
        instructions_with_tools: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps
        self.tool_registry = tool_registry or ToolRegistry()
        self.predictor = dspy.Predict(ReActStepSignature)

        default_no_tools = (
            "You are an expert autonomous agent. You do not have any external tools. "
            "Reason step by step, but always finish by emitting action 'finish' with a complete and useful answer. "
            "Never refuse because of missing details—make reasonable assumptions when necessary."
        )

        default_with_tools = (
            "You are an expert autonomous agent. Available tools: {tool_list}. "
            "Call a tool only when it materially improves the solution. When the task is solved, emit action 'finish' with the final answer."
        )

        self.instructions_no_tools = instructions_no_tools or default_no_tools
        self.instructions_with_tools = instructions_with_tools or default_with_tools

        logger.info(
            "react_generator_initialized",
            model=model,
            temperature=temperature,
            max_steps=max_steps,
            num_tools=len(self.tool_registry._tools),
        )

    @staticmethod
    def _format_trace(trace: List[Tuple[str, str, str, str]]) -> str:
        lines = []
        for idx, (thought, action, action_input, observation) in enumerate(trace, start=1):
            lines.append(f"Step {idx}: Thought: {thought}")
            lines.append(f"Step {idx}: Action: {action} | Input: {action_input}")
            lines.append(f"Step {idx}: Observation: {observation}")
        return "\n".join(lines)

    @staticmethod
    def _extract_references(text: str) -> List[str]:
        pattern = r"\[bullet-[a-zA-Z0-9\-]+\]"
        matches = re.findall(pattern, text or "")
        unique, seen = [], set()
        for match in matches:
            token = match.strip("[]")
            if token not in seen:
                unique.append(token)
                seen.add(token)
        return unique

    def _compose_task_description(self, task_input: TaskInput) -> str:
        tool_names = sorted(self.tool_registry._tools.keys())
        if tool_names:
            tool_list = ", ".join(tool_names)
            instruction = self.instructions_with_tools.format(tool_list=tool_list)
        else:
            instruction = self.instructions_no_tools

        return f"{instruction}\n\nTask: {task_input.description}".strip()

    def __call__(self, task_input: TaskInput) -> TaskOutput:
        start_time = time.time()

        structured_entries = getattr(task_input, "playbook_context_entries", []) or []
        if structured_entries:
            context_strings, bullet_ids = build_strings(structured_entries)
        else:
            context_strings = list(task_input.playbook_bullets)
            bullet_ids = None

        playbook_context = "No playbook strategies available."
        if context_strings:
            formatted = []
            for idx, text in enumerate(context_strings, start=1):
                identifier = bullet_ids[idx - 1] if bullet_ids and idx - 1 < len(bullet_ids) else f"bullet-{idx:03d}"
                formatted.append(f"Strategy {idx} [{identifier}]: {text}")
            playbook_context = "\n".join(formatted)

        trace: List[Tuple[str, str, str, str]] = []
        final_answer = None
        final_confidence = 0.0

        task_description_prompt = self._compose_task_description(task_input)

        for step in range(self.max_steps):
            interaction_trace = self._format_trace(trace)
            prediction = protected_predict(
                self.predictor,
                circuit_name="react-generator",
                failure_threshold=5,
                recovery_timeout=60,
                task_description=task_description_prompt,
                playbook_context=playbook_context,
                interaction_trace=interaction_trace,
            )

            thought = prediction.thought.strip()
            action = (prediction.action or "").strip()
            action_input = (prediction.action_input or "").strip()
            final_confidence = float(prediction.confidence or 0.0)

            if action.lower() in {"finish", "answer", "final", "return"}:
                final_answer = prediction.final_answer.strip() if prediction.final_answer else action_input or thought
                trace.append((thought, action, action_input, "<final>"))
                break

            observation: str
            if self.tool_registry.has_tool(action):
                try:
                    observation = self.tool_registry.execute(action, action_input)
                except Exception as exc:  # pragma: no cover - defensive
                    observation = f"ToolError: {exc}"
            else:
                observation = "Unknown action"

            trace.append((thought, action, action_input, observation))

        if final_answer is None:
            final_answer = trace[-1][3] if trace else task_input.description

        reasoning_trace = [f"Thought: {t} | Action: {a} -> {obs}" for t, a, _, obs in trace]
        bullets_referenced = []
        for entry in reasoning_trace:
            bullets_referenced.extend(self._extract_references(entry))

        latency_ms = int((time.time() - start_time) * 1000)

        output = TaskOutput(
            task_id=task_input.task_id,
            reasoning_trace=reasoning_trace,
            answer=final_answer,
            confidence=max(0.0, min(1.0, final_confidence)),
            bullets_referenced=bullets_referenced,
            latency_ms=latency_ms,
            model_name=self.model,
            prompt_tokens=None,
            completion_tokens=None,
        )

        logger.info(
            "react_generation_complete",
            task_id=task_input.task_id,
            steps=len(trace),
            latency_ms=latency_ms,
            confidence=output.confidence,
        )

        return output


def create_react_generator(
    model: str = "gpt-4-turbo",
    temperature: float = 0.2,
    max_steps: int = 12,
    tools: Dict[str, Callable[[str], str]] | None = None,
    instructions_no_tools: str | None = None,
    instructions_with_tools: str | None = None,
) -> ReActGenerator:
    """Factory helper for ReAct generator."""

    registry = ToolRegistry(tools)
    return ReActGenerator(
        model=model,
        temperature=temperature,
        max_steps=max_steps,
        tool_registry=registry,
        instructions_no_tools=instructions_no_tools,
        instructions_with_tools=instructions_with_tools,
    )
