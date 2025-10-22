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
from functools import lru_cache
from pathlib import Path
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

    def __init__(self, required_steps: int, *, allow_paragraphs: bool = False) -> None:
        self.required_steps = required_steps
        self.allow_paragraphs = allow_paragraphs

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        text = self._normalise_answer(answer)
        bullet_count = self._count_bullets(text)
        contains_numbers = len(re.findall(r"\b\d+[.)]", text))

        if bullet_count >= self.required_steps:
            return FeedbackResult(
                status="success",
                confidence=0.8 if bullet_count > self.required_steps else 0.75,
                evidence=f"Detected {bullet_count} bullet items",
                features={"bullet_count": bullet_count, "paragraph_allowed": self.allow_paragraphs},
            )

        if self.allow_paragraphs and contains_numbers >= self.required_steps:
            return FeedbackResult(
                status="success",
                confidence=0.65,
                evidence="Enumerated steps found in paragraph form",
                features={"numeric_markers": contains_numbers},
            )

        if bullet_count == 0 and contains_numbers == 0:
            return FeedbackResult(
                status="unknown",
                confidence=0.4,
                evidence="No structural markers detected",
                features={"bullet_count": bullet_count, "numeric_markers": contains_numbers},
            )

        return FeedbackResult(
            status="fail",
            confidence=0.6,
            evidence=f"Expected >= {self.required_steps} structured steps",
            features={"bullet_count": bullet_count, "numeric_markers": contains_numbers},
        )


class ChecklistChecker(BaseChecker):
    name = "checklist"

    def __init__(self, *, min_bullets: int = 5, min_keywords: int = 2, keywords: Optional[List[str]] = None) -> None:
        self.min_bullets = min_bullets
        self.min_keywords = min_keywords
        self.keywords = [kw.lower() for kw in (keywords or ["check", "ensure", "verify", "prepare"])]

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        text = self._normalise_answer(answer)
        bullet_count = self._count_bullets(text)
        matches = sum(1 for kw in self.keywords if kw in text.lower())

        if bullet_count >= self.min_bullets and matches >= self.min_keywords:
            return FeedbackResult(
                status="success",
                confidence=0.75,
                evidence="Checklist structure with relevant action verbs",
                features={"bullet_count": bullet_count, "keyword_matches": matches},
            )

        if bullet_count >= max(3, self.min_bullets // 2):
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


class KeywordsChecker(BaseChecker):
    name = "keywords"

    def __init__(
        self,
        *,
        required: Optional[List[str]] = None,
        any_of: Optional[List[str]] = None,
        min_matches: int = 1,
        min_bullets: int = 0,
    ) -> None:
        self.required = [kw.lower() for kw in (required or [])]
        self.any_of = [kw.lower() for kw in (any_of or [])]
        self.min_matches = min_matches
        self.min_bullets = min_bullets

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        text = self._normalise_answer(answer)
        lower = text.lower()
        bullet_count = self._count_bullets(text)

        missing_required = [kw for kw in self.required if kw not in lower]
        optional_matches = sum(1 for kw in self.any_of if kw in lower)

        if missing_required:
            return FeedbackResult(
                status="fail",
                confidence=0.7,
                evidence=f"Missing required keywords: {missing_required}",
                features={"missing_required": missing_required, "optional_matches": optional_matches},
            )

        if optional_matches >= self.min_matches and bullet_count >= self.min_bullets:
            return FeedbackResult(
                status="success",
                confidence=0.7,
                evidence="Required keywords present with adequate detail",
                features={"optional_matches": optional_matches, "bullet_count": bullet_count},
            )

        return FeedbackResult(
            status="unknown",
            confidence=0.5,
            evidence="Insufficient optional keyword coverage",
            features={"optional_matches": optional_matches, "bullet_count": bullet_count},
        )


class GenericChecker(BaseChecker):
    name = "generic"

    def __init__(self, *, min_bullets: int = 3) -> None:
        self.min_bullets = min_bullets

    def evaluate(self, *, answer: str, task: Dict[str, Any]) -> FeedbackResult:
        text = self._normalise_answer(answer)
        bullet_count = self._count_bullets(text)
        if bullet_count >= self.min_bullets:
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


CONFIG_FILENAME = Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "agent_feedback_config.json"

CHECKER_REGISTRY = {
    "StepCountChecker": StepCountChecker,
    "ChecklistChecker": ChecklistChecker,
    "ScheduleChecker": ScheduleChecker,
    "GenericChecker": GenericChecker,
    "KeywordsChecker": KeywordsChecker,
}

DEFAULT_CONFIG = [
    {
        "dataset": "agent_small",
        "patterns": ["agent-00[1-3]"],
        "checker": "StepCountChecker",
        "params": {"required_steps": 3},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-004"],
        "checker": "ChecklistChecker",
        "params": {},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-00[5-7]"],
        "checker": "ChecklistChecker",
        "params": {},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-009"],
        "checker": "StepCountChecker",
        "params": {"required_steps": 6},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-01[0-3]"],
        "checker": "ChecklistChecker",
        "params": {},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-01[4-6]"],
        "checker": "ChecklistChecker",
        "params": {},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-017"],
        "checker": "StepCountChecker",
        "params": {"required_steps": 4},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-018"],
        "checker": "ScheduleChecker",
        "params": {},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-019"],
        "checker": "ChecklistChecker",
        "params": {},
    },
    {
        "dataset": "agent_small",
        "patterns": ["agent-020"],
        "checker": "ChecklistChecker",
        "params": {},
    },
    {
        "dataset": "appworld_text",
        "patterns": ["app-.*"],
        "checker": "KeywordsChecker",
        "params": {
            "required": ["success"],
            "any_of": ["completed", "delivered", "resolved", "satisfied"],
            "min_matches": 1,
            "min_bullets": 3,
        },
    },
    {
        "dataset": "planning_constraints",
        "patterns": ["plan-.*"],
        "checker": "KeywordsChecker",
        "params": {
            "required": ["step", "owner"],
            "any_of": ["deadline", "timeline", "date"],
            "min_matches": 1,
            "min_bullets": 4,
        },
    },
]


@lru_cache(maxsize=1)
def _load_config() -> List[Dict[str, Any]]:
    if CONFIG_FILENAME.exists():
        try:
            return json.loads(CONFIG_FILENAME.read_text())
        except Exception as exc:
            logger.warning("agent_feedback_config_invalid", error=str(exc))
    return DEFAULT_CONFIG


def create_manager_for_dataset(dataset: str) -> Optional[AgentFeedbackManager]:
    config = [entry for entry in _load_config() if entry.get("dataset") == dataset]
    if not config:
        return None

    manager = AgentFeedbackManager()
    for entry in config:
        checker_name = entry.get("checker")
        checker_cls = CHECKER_REGISTRY.get(checker_name)
        if not checker_cls:
            logger.warning("agent_feedback_unknown_checker", dataset=dataset, checker=checker_name)
            continue
        params = entry.get("params") or {}
        try:
            checker = checker_cls(**params)
        except Exception as exc:
            logger.error("agent_feedback_checker_init_failed", dataset=dataset, checker=checker_name, error=str(exc))
            continue
        for pattern in entry.get("patterns", []):
            manager.register_checker(pattern, checker)

    return manager if manager._checkers else None


def create_default_manager() -> Optional[AgentFeedbackManager]:
    return create_manager_for_dataset("agent_small")


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
