"""Cancellation token support for BLACKICE 3.0.

Provides cooperative cancellation for long-running operations
with propagation to child operations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from blackice.primitives.errors import BlackiceError


class CancellationError(BlackiceError):
    """Operation was cancelled."""

    code = "E7003"

    def __init__(
        self,
        reason: str = "Operation cancelled",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason, context=context)


@dataclass
class CancellationToken:
    """Token for cooperative cancellation.

    Cancellation tokens allow operations to check for cancellation
    requests and stop gracefully. Tokens can form a hierarchy where
    cancelling a parent cancels all children.

    Example:
        token = CancellationToken()

        async def long_operation(token: CancellationToken):
            while not token.is_cancelled:
                await do_work()
                token.check()  # Raises if cancelled

        # Later, to cancel:
        token.cancel("User requested stop")
    """

    reason: str | None = field(default=None)
    cancelled_at: datetime | None = field(default=None)
    _is_cancelled: bool = field(default=False, repr=False)
    _callbacks: list[Callable[[], Awaitable[None] | None]] = field(
        default_factory=list, repr=False
    )
    _children: list[CancellationToken] = field(default_factory=list, repr=False)
    _parent: CancellationToken | None = field(default=None, repr=False)

    @property
    def is_cancelled(self) -> bool:
        """Check if this token or any parent is cancelled."""
        if self._is_cancelled:
            return True
        if self._parent:
            return self._parent.is_cancelled
        return False

    def check(self) -> None:
        """Check for cancellation and raise if cancelled.

        Raises:
            CancellationError: If the token is cancelled
        """
        if self.is_cancelled:
            raise CancellationError(
                self.reason or "Operation cancelled",
                context={"cancelled_at": str(self.cancelled_at)},
            )

    def cancel(self, reason: str | None = None) -> None:
        """Cancel this token and all children.

        Args:
            reason: Optional reason for cancellation
        """
        if self._is_cancelled:
            return

        self._is_cancelled = True
        self.reason = reason
        self.cancelled_at = datetime.now(timezone.utc)

        # Cancel all children
        for child in self._children:
            child.cancel(reason)

        # Execute callbacks
        for callback in self._callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    # Schedule for later execution
                    asyncio.create_task(result)
            except Exception:
                pass  # Ignore callback errors

    def on_cancel(
        self, callback: Callable[[], Awaitable[None] | None]
    ) -> CancellationToken:
        """Register a callback to run when cancelled.

        Args:
            callback: Function to call on cancellation

        Returns:
            Self for chaining
        """
        self._callbacks.append(callback)

        # If already cancelled, execute immediately
        if self._is_cancelled:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                pass

        return self

    def create_child(self) -> CancellationToken:
        """Create a child token that cancels when this token cancels.

        Returns:
            New child CancellationToken
        """
        child = CancellationToken()
        child._parent = self
        self._children.append(child)

        # If parent already cancelled, cancel child immediately
        if self._is_cancelled:
            child.cancel(self.reason)

        return child

    def create_linked(self, *others: CancellationToken) -> CancellationToken:
        """Create a token that cancels when this or any other cancels.

        Args:
            *others: Other tokens to link

        Returns:
            New linked CancellationToken
        """
        linked = self.create_child()

        # Register on others to cancel linked
        def cancel_linked() -> None:
            if not linked._is_cancelled:
                linked.cancel("Linked token cancelled")

        for other in others:
            other.on_cancel(cancel_linked)

        return linked


class CancellationTokenSource:
    """Source for creating and managing cancellation tokens.

    Provides factory methods for creating tokens with
    timeout or linked cancellation.
    """

    def __init__(self) -> None:
        self._tokens: list[CancellationToken] = []

    def create(self) -> CancellationToken:
        """Create a new cancellation token."""
        token = CancellationToken()
        self._tokens.append(token)
        return token

    def create_with_timeout(
        self,
        timeout_seconds: float,
        reason: str = "Operation timed out",
    ) -> CancellationToken:
        """Create a token that auto-cancels after timeout.

        Args:
            timeout_seconds: Timeout in seconds
            reason: Cancellation reason

        Returns:
            CancellationToken that will auto-cancel
        """
        token = self.create()

        async def timeout_cancel() -> None:
            await asyncio.sleep(timeout_seconds)
            if not token.is_cancelled:
                token.cancel(reason)

        asyncio.create_task(timeout_cancel())
        return token

    def cancel_all(self, reason: str = "All operations cancelled") -> None:
        """Cancel all tokens created by this source."""
        for token in self._tokens:
            if not token.is_cancelled:
                token.cancel(reason)


async def with_cancellation(
    operation: Callable[[], Awaitable[Any]],
    token: CancellationToken,
    check_interval: float = 0.1,
) -> Any:
    """Execute an operation with cancellation support.

    Periodically checks for cancellation while waiting for
    the operation to complete.

    Args:
        operation: Async operation to execute
        token: Cancellation token to monitor
        check_interval: How often to check for cancellation

    Returns:
        Operation result

    Raises:
        CancellationError: If cancelled during execution
    """
    task = asyncio.create_task(operation())

    try:
        while not task.done():
            token.check()
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=check_interval,
                )
            except asyncio.TimeoutError:
                continue
    except CancellationError:
        task.cancel()
        raise

    return await task


@dataclass
class CancellableOperation:
    """Wrapper for cancellable async operations.

    Example:
        op = CancellableOperation(token)

        @op.wrap
        async def my_operation():
            for i in range(100):
                await do_work(i)

        await op.run()
    """

    token: CancellationToken
    check_interval: float = 0.1

    def wrap(
        self, fn: Callable[..., Awaitable[Any]]
    ) -> Callable[..., Awaitable[Any]]:
        """Decorator to make a function cancellable."""

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            return await with_cancellation(
                lambda: fn(*args, **kwargs),
                self.token,
                self.check_interval,
            )

        return wrapped
