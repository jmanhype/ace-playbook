"""Context building utilities for generator modules.

Rank, deduplicate, and format playbook bullets for prompt injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class ContextEntry:
    """Normalized playbook context entry with minimal metadata."""

    bullet_id: str
    content: str
    stage: str = "shadow"
    helpful_count: int = 0
    harmful_count: int = 0

    def weight(self) -> float:
        """Compute ranking weight using stage and helpful/harmful counters."""

        stage_weights = {
            "prod": 1.5,
            "staging": 1.2,
            "shadow": 1.0,
        }

        base = stage_weights.get(self.stage.lower(), 1.0)
        helpful = max(self.helpful_count, 0)
        harmful = max(self.harmful_count, 0)
        ratio = (1.0 + helpful) / (1.0 + harmful)
        return base * ratio


@dataclass(frozen=True)
class ContextBundle:
    """Formatted context ready for prompt consumption."""

    contents: List[str]
    bullet_ids: List[str]


def _to_entry(raw: dict) -> ContextEntry:
    """Convert raw mapping into :class:`ContextEntry`."""

    return ContextEntry(
        bullet_id=str(raw.get("id") or raw.get("bullet_id") or ""),
        content=str(raw.get("content") or ""),
        stage=str(raw.get("stage") or "shadow"),
        helpful_count=int(raw.get("helpful_count") or 0),
        harmful_count=int(raw.get("harmful_count") or 0),
    )


def build_bundle(
    entries: Iterable[dict],
    budget: int = 40,
) -> ContextBundle:
    """Rank and deduplicate context entries.

    Args:
        entries: Iterable of raw dicts describing bullets.
        budget: Maximum number of entries to retain.

    Returns:
        :class:`ContextBundle` with ordered contents and bullet IDs.
    """

    normalized: List[ContextEntry] = []
    seen_fingerprints = set()

    for raw in entries:
        entry = _to_entry(raw)
        if not entry.content:
            continue
        fingerprint = entry.content.strip().lower()
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        normalized.append(entry)

    ranked = sorted(normalized, key=lambda e: e.weight(), reverse=True)
    limited = ranked[: budget if budget > 0 else len(ranked)]

    contents: List[str] = [entry.content for entry in limited]
    bullet_ids: List[str] = [entry.bullet_id for entry in limited]

    return ContextBundle(contents=contents, bullet_ids=bullet_ids)


def build_strings(entries: Iterable[dict], budget: int = 40) -> Tuple[List[str], List[str]]:
    """Helper returning parallel lists of contents and IDs."""

    bundle = build_bundle(entries, budget=budget)
    return bundle.contents, bundle.bullet_ids

