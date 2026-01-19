"""Local execution provider for BLACKICE 3.0.

Implements the ExecutionProvider protocol for local subprocess execution.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import AsyncIterator

from blackice.adapters.execution.base import (
    BaseExecutionProvider,
    ExecutionCapabilities,
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionResult,
    HealthStatus,
)
from blackice.primitives.errors import ExecutionError


class LocalExecutionProvider(BaseExecutionProvider):
    """Local execution provider using subprocess.

    Executes commands directly on the host system using asyncio.subprocess.
    Supports streaming output and timeout handling.
    """

    def __init__(
        self,
        default_timeout: float = 300.0,
        default_working_dir: Path | None = None,
        allowed_commands: list[str] | None = None,
    ) -> None:
        super().__init__(default_timeout, default_working_dir)
        self.allowed_commands = allowed_commands

    @property
    def name(self) -> str:
        return "local"

    @property
    def capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            environment=ExecutionEnvironment.LOCAL,
            supports_streaming=True,
            supports_attach=False,
            supports_isolation=False,
            max_timeout=3600.0,
            max_concurrent=50,
        )

    async def execute(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> ExecutionResult:
        """Execute a command using subprocess."""
        config = self._merge_config(config)
        cmd_list = self._normalize_command(command)
        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Validate command if allowlist is set
        if self.allowed_commands and cmd_list[0] not in self.allowed_commands:
            raise ExecutionError(
                f"Command '{cmd_list[0]}' not in allowed commands",
                context={"command": cmd_str, "allowed": self.allowed_commands},
            )

        # Build environment
        env = os.environ.copy()
        env.update(config.env)

        start = time.monotonic()
        timed_out = False
        killed = False

        try:
            if config.shell:
                process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE if config.capture_stderr else None,
                    cwd=config.working_dir,
                    env=env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd_list,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE if config.capture_stderr else None,
                    cwd=config.working_dir,
                    env=env,
                )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                process.kill()
                await process.wait()
                stdout_bytes = b""
                stderr_bytes = b"Process killed due to timeout"

            duration = time.monotonic() - start

            # Truncate output if too large
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")

            if len(stdout) > config.max_output_bytes:
                stdout = stdout[: config.max_output_bytes] + "\n[OUTPUT TRUNCATED]"
            if len(stderr) > config.max_output_bytes:
                stderr = stderr[: config.max_output_bytes] + "\n[OUTPUT TRUNCATED]"

            return ExecutionResult(
                exit_code=process.returncode or -1,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                command=cmd_str,
                timed_out=timed_out,
                killed=killed,
            )

        except Exception as e:
            duration = time.monotonic() - start
            raise ExecutionError(
                f"Command execution failed: {e}",
                context={
                    "command": cmd_str,
                    "duration": duration,
                },
            ) from e

    async def stream(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream command output line by line."""
        config = self._merge_config(config)
        cmd_list = self._normalize_command(command)
        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Validate command if allowlist is set
        if self.allowed_commands and cmd_list[0] not in self.allowed_commands:
            raise ExecutionError(
                f"Command '{cmd_list[0]}' not in allowed commands",
                context={"command": cmd_str},
            )

        # Build environment
        env = os.environ.copy()
        env.update(config.env)

        try:
            if config.shell:
                process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT if config.capture_stderr else None,
                    cwd=config.working_dir,
                    env=env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd_list,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT if config.capture_stderr else None,
                    cwd=config.working_dir,
                    env=env,
                )

            assert process.stdout is not None

            async def read_with_timeout() -> bytes | None:
                try:
                    return await asyncio.wait_for(
                        process.stdout.readline(),  # type: ignore
                        timeout=config.timeout,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    return None

            while True:
                line = await read_with_timeout()
                if line is None:
                    yield "[PROCESS TIMED OUT]"
                    break
                if not line:
                    break
                yield line.decode("utf-8", errors="replace").rstrip("\n")

            await process.wait()

        except Exception as e:
            raise ExecutionError(
                f"Streaming execution failed: {e}",
                context={"command": cmd_str},
            ) from e

    async def health(self) -> HealthStatus:
        """Check local execution health."""
        start = time.monotonic()
        try:
            result = await self.execute(["echo", "health"])
            latency = (time.monotonic() - start) * 1000

            return HealthStatus(
                healthy=result.exit_code == 0,
                environment=ExecutionEnvironment.LOCAL,
                latency_ms=latency,
                details={
                    "provider": self.name,
                    "stdout": result.stdout.strip(),
                },
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                environment=ExecutionEnvironment.LOCAL,
                error=str(e),
                details={"provider": self.name},
            )
