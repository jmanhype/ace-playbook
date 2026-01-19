"""Token and cost budget management for BLACKICE 3.0.

Tracks resource usage and enforces budgets to prevent
runaway costs or token consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from blackice.primitives.errors import BlackiceError


class BudgetExceededError(BlackiceError):
    """Budget limit exceeded."""

    code = "E7002"

    def __init__(
        self,
        budget_type: str,
        limit: float,
        used: float,
        requested: float,
    ) -> None:
        super().__init__(
            f"{budget_type} budget exceeded: {used + requested:.2f} > {limit:.2f}",
            context={
                "budget_type": budget_type,
                "limit": limit,
                "used": used,
                "requested": requested,
            },
        )


@dataclass
class UsageRecord:
    """Record of resource usage."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tokens: int = 0
    cost_usd: float = 0.0
    operation: str = ""
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetLimits:
    """Budget limits for a run."""

    max_tokens: int = 100_000
    max_cost_usd: float = 10.0
    max_requests: int = 1000
    max_duration_seconds: float = 3600.0

    # Per-operation limits
    max_tokens_per_request: int = 10_000
    max_cost_per_request: float = 1.0


@dataclass
class BudgetStatus:
    """Current budget status."""

    tokens_used: int = 0
    tokens_remaining: int = 0
    tokens_percent: float = 0.0

    cost_used: float = 0.0
    cost_remaining: float = 0.0
    cost_percent: float = 0.0

    requests_used: int = 0
    requests_remaining: int = 0

    elapsed_seconds: float = 0.0
    remaining_seconds: float = 0.0

    is_exceeded: bool = False
    exceeded_type: str | None = None


class BudgetManager:
    """Manages resource budgets for a run.

    Tracks token usage, costs, and enforces limits to prevent
    runaway resource consumption.
    """

    def __init__(
        self,
        limits: BudgetLimits | None = None,
        run_id: str | None = None,
    ) -> None:
        self.limits = limits or BudgetLimits()
        self.run_id = run_id
        self._start_time = datetime.now(timezone.utc)
        self._records: list[UsageRecord] = []

        # Running totals
        self._total_tokens = 0
        self._total_cost = 0.0
        self._request_count = 0

    @property
    def tokens_used(self) -> int:
        """Get total tokens used."""
        return self._total_tokens

    @property
    def cost_used(self) -> float:
        """Get total cost in USD."""
        return self._total_cost

    @property
    def requests_made(self) -> int:
        """Get total requests made."""
        return self._request_count

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since start."""
        now = datetime.now(timezone.utc)
        return (now - self._start_time).total_seconds()

    def get_status(self) -> BudgetStatus:
        """Get current budget status."""
        elapsed = self.elapsed_seconds

        tokens_remaining = max(0, self.limits.max_tokens - self._total_tokens)
        cost_remaining = max(0, self.limits.max_cost_usd - self._total_cost)
        requests_remaining = max(0, self.limits.max_requests - self._request_count)
        time_remaining = max(0, self.limits.max_duration_seconds - elapsed)

        # Check what's exceeded
        exceeded_type = None
        is_exceeded = False

        if self._total_tokens >= self.limits.max_tokens:
            is_exceeded = True
            exceeded_type = "tokens"
        elif self._total_cost >= self.limits.max_cost_usd:
            is_exceeded = True
            exceeded_type = "cost"
        elif self._request_count >= self.limits.max_requests:
            is_exceeded = True
            exceeded_type = "requests"
        elif elapsed >= self.limits.max_duration_seconds:
            is_exceeded = True
            exceeded_type = "duration"

        return BudgetStatus(
            tokens_used=self._total_tokens,
            tokens_remaining=tokens_remaining,
            tokens_percent=(self._total_tokens / self.limits.max_tokens) * 100,
            cost_used=self._total_cost,
            cost_remaining=cost_remaining,
            cost_percent=(self._total_cost / self.limits.max_cost_usd) * 100,
            requests_used=self._request_count,
            requests_remaining=requests_remaining,
            elapsed_seconds=elapsed,
            remaining_seconds=time_remaining,
            is_exceeded=is_exceeded,
            exceeded_type=exceeded_type,
        )

    def check_budget(
        self,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Check if a request would exceed budget.

        Args:
            tokens: Tokens the request would use
            cost_usd: Cost the request would incur

        Raises:
            BudgetExceededError: If budget would be exceeded
        """
        # Check per-request limits
        if tokens > self.limits.max_tokens_per_request:
            raise BudgetExceededError(
                "tokens_per_request",
                self.limits.max_tokens_per_request,
                0,
                tokens,
            )

        if cost_usd > self.limits.max_cost_per_request:
            raise BudgetExceededError(
                "cost_per_request",
                self.limits.max_cost_per_request,
                0,
                cost_usd,
            )

        # Check total limits
        if self._total_tokens + tokens > self.limits.max_tokens:
            raise BudgetExceededError(
                "total_tokens",
                self.limits.max_tokens,
                self._total_tokens,
                tokens,
            )

        if self._total_cost + cost_usd > self.limits.max_cost_usd:
            raise BudgetExceededError(
                "total_cost",
                self.limits.max_cost_usd,
                self._total_cost,
                cost_usd,
            )

        if self._request_count + 1 > self.limits.max_requests:
            raise BudgetExceededError(
                "total_requests",
                self.limits.max_requests,
                self._request_count,
                1,
            )

        # Check duration
        if self.elapsed_seconds > self.limits.max_duration_seconds:
            raise BudgetExceededError(
                "duration",
                self.limits.max_duration_seconds,
                self.elapsed_seconds,
                0,
            )

    def record_usage(
        self,
        tokens: int = 0,
        cost_usd: float = 0.0,
        operation: str = "",
        model: str = "",
        **metadata: Any,
    ) -> UsageRecord:
        """Record resource usage.

        Args:
            tokens: Tokens used
            cost_usd: Cost in USD
            operation: Operation name
            model: Model used
            **metadata: Additional metadata

        Returns:
            The usage record
        """
        record = UsageRecord(
            tokens=tokens,
            cost_usd=cost_usd,
            operation=operation,
            model=model,
            metadata=metadata,
        )
        self._records.append(record)

        # Update totals
        self._total_tokens += tokens
        self._total_cost += cost_usd
        self._request_count += 1

        return record

    def get_usage_by_model(self) -> dict[str, dict[str, float]]:
        """Get usage breakdown by model."""
        by_model: dict[str, dict[str, float]] = {}

        for record in self._records:
            if record.model not in by_model:
                by_model[record.model] = {"tokens": 0, "cost_usd": 0.0, "requests": 0}

            by_model[record.model]["tokens"] += record.tokens
            by_model[record.model]["cost_usd"] += record.cost_usd
            by_model[record.model]["requests"] += 1

        return by_model

    def get_usage_by_operation(self) -> dict[str, dict[str, float]]:
        """Get usage breakdown by operation."""
        by_op: dict[str, dict[str, float]] = {}

        for record in self._records:
            if record.operation not in by_op:
                by_op[record.operation] = {"tokens": 0, "cost_usd": 0.0, "requests": 0}

            by_op[record.operation]["tokens"] += record.tokens
            by_op[record.operation]["cost_usd"] += record.cost_usd
            by_op[record.operation]["requests"] += 1

        return by_op

    def to_dict(self) -> dict[str, Any]:
        """Serialize budget state to dictionary."""
        status = self.get_status()
        return {
            "run_id": self.run_id,
            "limits": {
                "max_tokens": self.limits.max_tokens,
                "max_cost_usd": self.limits.max_cost_usd,
                "max_requests": self.limits.max_requests,
                "max_duration_seconds": self.limits.max_duration_seconds,
            },
            "usage": {
                "tokens": self._total_tokens,
                "cost_usd": self._total_cost,
                "requests": self._request_count,
                "elapsed_seconds": self.elapsed_seconds,
            },
            "status": {
                "is_exceeded": status.is_exceeded,
                "exceeded_type": status.exceeded_type,
                "tokens_percent": status.tokens_percent,
                "cost_percent": status.cost_percent,
            },
        }
