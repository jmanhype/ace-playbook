"""Experience buffer and episode results for the live loop."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, List

from ace.agent_learning.types import EpisodeResult


class ExperienceBuffer:
    """Bounded buffer that tracks recent agent episodes."""

    def __init__(self, *, maxlen: int = 50) -> None:
        self._buffer: Deque[EpisodeResult] = deque(maxlen=maxlen)

    def append(self, result: EpisodeResult) -> None:
        self._buffer.append(result)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)

    def __iter__(self) -> Iterator[EpisodeResult]:  # pragma: no cover - trivial
        return iter(self._buffer)

    def to_dict(self) -> List[dict]:
        """Return a JSON-friendly representation of the buffer."""

        return [item.to_summary() for item in self._buffer]


__all__ = ["EpisodeResult", "ExperienceBuffer"]
