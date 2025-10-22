"""Heuristic + classifier feedback adaptor for agent planning tasks.

Routes each task to an evaluator that returns structured feedback used by the
benchmark runner. The adaptor mirrors the finance guardrail architecture but is
designed for qualitative agent tasks (plans, schedules, checklists).

Key concepts:
    - AgentFeedbackManager: entry point; selects checker by task id/pattern.
    - Checker subclasses: implement evaluate(answer, task) -> FeedbackResult.
    - FeedbackResult: dataclass capturing status/evidence/features.

The module is intentionally conservative: checkers default to ``unknown``
instead of guessing success, helping ACE avoid learning from noisy signals.

Future extensions (roadmap):
    - Train an ML classifier using the heuristics' features.
    - Plug-in symbolic validators for richer constraint satisfaction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="agent_feedback")


# ---------------------------------------------------------------------------
# Data structures


@dataclass
class FeedbackResult:
    """Structured feedback for a task/answer pair."""

    status: str  # "success", "fail", "unknown"
    confidence: float
    evidence: str
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "features": self.features,
        }


class BaseChecker:
    """Base class for all agent feedback checkers."""

    name: str = "base"

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        raise NotImplementedError

    # Helper utilities -----------------------------------------------------

    @staticmethod
    def _normalise_answer(answer: str) -> str:
        return answer.strip()

    @staticmethod
    def _count_bullets(answer: str) -> int:
        patterns = [r"^[-*] ", r"^\d+\. "]
        count = 0
        for line in answer.splitlines():
            for pattern in patterns:
                if re.match(pattern, line.strip()):
                    count += 1
                    break
        return count


class StepCountChecker(BaseChecker):
    name = "step_count"

    def __init__(self, required_steps: int) -> None:
        self.required_steps = required_steps

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        text = self._normalise_answer(answer)
        bullet_count = self._count_bullets(text)
        matches_num = bool(re.search(rf"\b{self.required_steps}\b", text))

        if bullet_count >= self.required_steps:
            return FeedbackResult(
                status="success",
                confidence=0.8 if matches_num else 0.7,
                evidence=f"Detected {bullet_count} bullet items",
                features={"bullet_count": bullet_count, "mentions_required": matches_num},
            )

        if bullet_count == 0:
            return FeedbackResult(
                status="unknown",
                confidence=0.4,
                evidence="No bullet structure detected",
                features={"bullet_count": bullet_count},
            )

        return FeedbackResult(
            status="fail",
            confidence=0.6,
            evidence=f"Expected >= {self.required_steps} steps, found {bullet_count}",
            features={"bullet_count": bullet_count},
        )


class ChecklistChecker(BaseChecker):
    name = "checklist"

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        text = self._normalise_answer(answer)
        bullet_count = self._count_bullets(text)
        keywords = ["check", "ensure", "verify", "prepare"]
        matches = sum(1 for kw in keywords if kw in text.lower())

        if bullet_count >= 5 and matches >= 2:
            return FeedbackResult(
                status="success",
                confidence=0.75,
                evidence="Checklist structure with relevant action verbs",
                features={"bullet_count": bullet_count, "keyword_matches": matches},
            )

        if bullet_count >= 3:
            return FeedbackResult(
                status="unknown",
                confidence=0.5,
                evidence="Checklist detected but few task-specific keywords",
                features={"bullet_count": bullet_count, "keyword_matches": matches},
            )

        return FeedbackResult(
            status="fail",
            confidence=0.6,
            evidence="Insufficient checklist structure",
            features={"bullet_count": bullet_count, "keyword_matches": matches},
        )


class ScheduleChecker(BaseChecker):
    name = "schedule"

    _time_pattern = re.compile(r"\b(\d{1,2})(:\d{2})? (am|pm)\b", re.I)

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        text = self._normalise_answer(answer)
        times = len(self._time_pattern.findall(text))
        bullet_count = self._count_bullets(text)

        if times >= 3 and bullet_count >= 3:
            return FeedbackResult(
                status="success",
                confidence=0.8,
                evidence="Schedule includes time slots and structure",
                features={"time_mentions": times, "bullet_count": bullet_count},
            )

        if times >= 1:
            return FeedbackResult(
                status="unknown",
                confidence=0.5,
                evidence="Some time slots detected but incomplete",
                features={"time_mentions": times, "bullet_count": bullet_count},
            )

        return FeedbackResult(
            status="fail",
            confidence=0.6,
            evidence="No recognisable time slots",
            features={"time_mentions": times, "bullet_count": bullet_count},
        )


class GenericChecker(BaseChecker):
    name = "generic"

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        text = self._normalise_answer(answer)
        bullet_count = self._count_bullets(text)
        if bullet_count >= 3:
            return FeedbackResult(
                status="unknown",
                confidence=0.5,
                evidence="Structured response but no domain-specific heuristic",
                features={"bullet_count": bullet_count},
            )
        return FeedbackResult(
            status="unknown",
            confidence=0.3,
            evidence="Unable to score response",
            features={"bullet_count": bullet_count},
        )


# ---------------------------------------------------------------------------
# Manager


class AgentFeedbackManager:
    """Routes tasks to appropriate checkers and combines heuristic signals."""

    def __init__(self, *, default_checker: Optional[BaseChecker] = None) -> None:
        self._default_checker = default_checker or GenericChecker()
        self._checkers: List[Tuple[str, BaseChecker]] = []

    def register_checker(self, matcher: str, checker: BaseChecker) -> None:
        """Register a checker for tasks whose ID matches the regex pattern."""

        self._checkers.append((matcher, checker))

    def evaluate(self, task: Dict[str, Any], answer: str) -> FeedbackResult:
        task_id = task.get("task_id", "")
        for pattern, checker in self._checkers:
            if re.fullmatch(pattern, task_id):
                logger.debug("agent_checker_matched", task_id=task_id, checker=checker.name)
                return checker.evaluate(answer=answer, task=task)

        logger.debug("agent_checker_default", task_id=task_id, checker=self._default_checker.name)
        return self._default_checker.evaluate(answer=answer, task=task)


def create_default_manager() -> AgentFeedbackManager:
    manager = AgentFeedbackManager()

    manager.register_checker(r"agent-00[1-3]", StepCountChecker(required_steps=3))
    manager.register_checker(r"agent-004", ChecklistChecker())
    manager.register_checker(r"agent-00[5-7]", ChecklistChecker())
    manager.register_checker(r"agent-009", StepCountChecker(required_steps=6))
    manager.register_checker(r"agent-01[0-3]", ChecklistChecker())
    manager.register_checker(r"agent-01[4-6]", ChecklistChecker())
    manager.register_checker(r"agent-017", StepCountChecker(required_steps=4))
    manager.register_checker(r"agent-018", ScheduleChecker())
    manager.register_checker(r"agent-019", ChecklistChecker())
    manager.register_checker(r"agent-020", ChecklistChecker())

    return manager


# ---------------------------------------------------------------------------
# Classifier integration placeholder


class SatisfactionClassifier:
    """LLM-backed satisfaction classifier.

    A simple wrapper that can be swapped out later. For now, it uses the existing
    JSONSafeLLMClient to obtain a judgement when heuristics are unsure.
    """

    def __init__(self, client, *, threshold: float = 0.8) -> None:
        self.client = client
        self.threshold = threshold

    def evaluate(self, task: Dict[str, Any], answer: str, heuristic_result: FeedbackResult) -> FeedbackResult:
        prompt = json.dumps(
            {
                "task": task.get("description", ""),
                "answer": answer,
                "heuristic": heuristic_result.to_dict(),
            },
            indent=2,
        )
        response = self.client.classify(prompt)
        success = response.get("success")
        confidence = float(response.get("confidence", 0.0))
        evidence = response.get("rationale", "")

        if success and confidence >= self.threshold:
            return FeedbackResult(
                status="success",
                confidence=confidence,
                evidence=f"Classifier: {evidence}",
                features={"classifier": response},
            )

        return FeedbackResult(
            status="unknown",
            confidence=confidence,
            evidence=f"Classifier uncertain: {evidence}",
            features={"classifier": response},
        )


__all__ = [
    "AgentFeedbackManager",
    "FeedbackResult",
    "create_default_manager",
    "SatisfactionClassifier",
]
