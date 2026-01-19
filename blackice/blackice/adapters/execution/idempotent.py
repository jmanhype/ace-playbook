"""Idempotent execution provider wrapper for BLACKICE 3.0.

Wraps an ExecutionProvider with idempotency guarantees:
- Generates idempotency keys for each execution
- Checks for duplicate executions before running
- Records execution results for crash recovery
- Supports skip-on-resume semantics
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from blackice.adapters.execution.base import (
    BaseExecutionProvider,
    ExecutionCapabilities,
    ExecutionConfig,
    ExecutionProvider,
    ExecutionResult,
    HealthStatus,
)
from blackice.recovery.idempotency import (
    EffectType,
    IdempotencyKeyGenerator,
    IdempotencyRecord,
    IdempotencyStore,
)


@dataclass
class IdempotentExecutionConfig(ExecutionConfig):
    """Extended config with idempotency options."""

    run_id: str | None = None
    task_id: str | None = None
    attempt: int = 1
    skip_if_executed: bool = True  # Skip execution if key exists


@dataclass
class CachedExecutionResult:
    """Cached result from previous execution."""

    result: ExecutionResult
    record: IdempotencyRecord
    was_cached: bool = True


class IdempotentExecutionProvider:
    """Wrapper that adds idempotency to any ExecutionProvider.

    This wrapper ensures that commands are executed at most once
    per (run_id, task_id, attempt, command) combination. On resume,
    previously executed commands are skipped.

    Usage:
        provider = LocalExecutionProvider()
        idempotent = IdempotentExecutionProvider(
            provider,
            idempotency_store=IdempotencyStore(storage_dir),
        )

        result = await idempotent.execute(
            "npm test",
            config=IdempotentExecutionConfig(
                run_id=run_id,
                task_id=task_id,
                attempt=1,
            ),
        )
    """

    def __init__(
        self,
        provider: ExecutionProvider,
        idempotency_store: IdempotencyStore,
    ) -> None:
        """Initialize the idempotent wrapper.

        Args:
            provider: Underlying execution provider
            idempotency_store: Store for tracking executed commands
        """
        self.provider = provider
        self.store = idempotency_store
        self.key_generator = IdempotencyKeyGenerator()

    @property
    def name(self) -> str:
        """Provider name with idempotent prefix."""
        return f"idempotent-{self.provider.name}"

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Delegate capabilities to wrapped provider."""
        return self.provider.capabilities

    def _generate_key(
        self,
        command: str | list[str],
        config: IdempotentExecutionConfig,
    ) -> str:
        """Generate idempotency key for a command.

        Args:
            command: Command to execute
            config: Execution configuration

        Returns:
            Idempotency key string
        """
        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Generate deterministic key from command and context
        return self.key_generator.generate_for_command(
            run_id=config.run_id or "unknown",
            task_id=config.task_id or "unknown",
            attempt=config.attempt,
            command=cmd_str,
        )

    async def execute(
        self,
        command: str | list[str],
        *,
        config: IdempotentExecutionConfig | ExecutionConfig | None = None,
    ) -> ExecutionResult:
        """Execute a command with idempotency guarantees.

        If the same command was already executed for this run/task/attempt,
        returns a cached result indicating the command was skipped.

        Args:
            command: Command string or list of arguments
            config: Execution configuration with idempotency options

        Returns:
            ExecutionResult with output and status
        """
        # Convert to idempotent config if needed
        if config is None:
            config = IdempotentExecutionConfig()
        elif not isinstance(config, IdempotentExecutionConfig):
            config = IdempotentExecutionConfig(
                working_dir=config.working_dir,
                env=config.env,
                timeout=config.timeout,
                max_output_bytes=config.max_output_bytes,
                capture_stderr=config.capture_stderr,
                shell=config.shell,
            )

        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Generate idempotency key
        key = self._generate_key(command, config)

        # Check if already executed
        if config.skip_if_executed and config.run_id:
            existing = await self.store.check(key)
            if existing:
                # Return a result indicating we skipped
                return ExecutionResult(
                    exit_code=0,
                    stdout=f"[SKIPPED] Command already executed (key: {key[:16]}...)",
                    stderr="",
                    duration_seconds=0.0,
                    command=cmd_str,
                )

        # Execute the command
        result = await self.provider.execute(command, config=config)

        # Record execution if we have context
        if config.run_id and config.task_id:
            await self.store.record(
                key=key,
                run_id=config.run_id,
                task_id=config.task_id,
                effect_type=EffectType.COMMAND_EXEC,
                attempt=config.attempt,
                result_hash=self._hash_result(result),
                metadata={
                    "command": cmd_str,
                    "exit_code": result.exit_code,
                    "duration_seconds": result.duration_seconds,
                    "timed_out": result.timed_out,
                },
            )

        return result

    async def execute_with_record(
        self,
        command: str | list[str],
        *,
        config: IdempotentExecutionConfig | None = None,
    ) -> tuple[ExecutionResult, IdempotencyRecord | None]:
        """Execute and return both result and idempotency record.

        Args:
            command: Command string or list of arguments
            config: Execution configuration

        Returns:
            Tuple of (ExecutionResult, IdempotencyRecord or None)
        """
        if config is None:
            config = IdempotentExecutionConfig()

        result = await self.execute(command, config=config)

        # Get the record if it exists
        record = None
        if config.run_id:
            key = self._generate_key(command, config)
            record = await self.store.check(key)

        return result, record

    async def stream(
        self,
        command: str | list[str],
        *,
        config: IdempotentExecutionConfig | ExecutionConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream command output with idempotency tracking.

        Note: Streaming commands are recorded after completion,
        so they can be skipped on resume but not replayed.

        Args:
            command: Command string or list of arguments
            config: Execution configuration

        Yields:
            Output lines as they're produced
        """
        # Convert to idempotent config if needed
        if config is None:
            config = IdempotentExecutionConfig()
        elif not isinstance(config, IdempotentExecutionConfig):
            config = IdempotentExecutionConfig(
                working_dir=config.working_dir,
                env=config.env,
                timeout=config.timeout,
                max_output_bytes=config.max_output_bytes,
                capture_stderr=config.capture_stderr,
                shell=config.shell,
            )

        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Generate key and check for existing execution
        key = self._generate_key(command, config)

        if config.skip_if_executed and config.run_id:
            existing = await self.store.check(key)
            if existing:
                yield f"[SKIPPED] Command already executed (key: {key[:16]}...)"
                return

        # Stream from underlying provider
        output_lines: list[str] = []
        async for line in self.provider.stream(command, config=config):
            output_lines.append(line)
            yield line

        # Record after streaming completes
        if config.run_id and config.task_id:
            await self.store.record(
                key=key,
                run_id=config.run_id,
                task_id=config.task_id,
                effect_type=EffectType.COMMAND_EXEC,
                attempt=config.attempt,
                metadata={
                    "command": cmd_str,
                    "streaming": True,
                    "line_count": len(output_lines),
                },
            )

    async def health(self) -> HealthStatus:
        """Delegate health check to wrapped provider."""
        return await self.provider.health()

    async def was_executed(
        self,
        command: str | list[str],
        run_id: str,
        task_id: str,
        attempt: int = 1,
    ) -> bool:
        """Check if a command was already executed.

        Args:
            command: Command to check
            run_id: Run identifier
            task_id: Task identifier
            attempt: Attempt number

        Returns:
            True if command was already executed
        """
        config = IdempotentExecutionConfig(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
        )
        key = self._generate_key(command, config)
        return await self.store.is_executed(key)

    async def clear_run_executions(self, run_id: str) -> int:
        """Clear all execution records for a run.

        Args:
            run_id: Run to clear

        Returns:
            Number of records cleared
        """
        return await self.store.clear_run(run_id)

    def _hash_result(self, result: ExecutionResult) -> str:
        """Hash an execution result for verification.

        Args:
            result: Execution result

        Returns:
            SHA-256 hash of result content
        """
        content = json.dumps(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout[:1000],  # Truncate for hashing
                "stderr": result.stderr[:1000],
                "timed_out": result.timed_out,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def wrap_with_idempotency(
    provider: ExecutionProvider,
    storage_dir: Path,
) -> IdempotentExecutionProvider:
    """Convenience function to wrap a provider with idempotency.

    Args:
        provider: Execution provider to wrap
        storage_dir: Directory for idempotency records

    Returns:
        IdempotentExecutionProvider wrapping the provider
    """
    store = IdempotencyStore(storage_dir / "idempotency")
    return IdempotentExecutionProvider(provider, store)
