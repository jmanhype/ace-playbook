"""Sandbox execution provider for BLACKICE 3.0.

Implements the ExecutionProvider protocol for ephemeral isolated execution.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
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


class SandboxExecutionProvider(BaseExecutionProvider):
    """Sandbox execution provider for ephemeral isolated execution.

    Creates temporary isolated environments for each execution,
    with automatic cleanup and resource limits.
    """

    def __init__(
        self,
        default_timeout: float = 60.0,
        default_working_dir: Path | None = None,
        max_file_size: int = 10_000_000,  # 10MB
        max_files: int = 100,
        cleanup_on_exit: bool = True,
    ) -> None:
        super().__init__(default_timeout, default_working_dir)
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.cleanup_on_exit = cleanup_on_exit
        self._sandboxes: dict[str, Path] = {}

    @property
    def name(self) -> str:
        return "sandbox"

    @property
    def capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            environment=ExecutionEnvironment.SANDBOX,
            supports_streaming=True,
            supports_attach=False,
            supports_isolation=True,
            max_timeout=300.0,
            max_concurrent=10,
        )

    def _create_sandbox(self) -> tuple[str, Path]:
        """Create a new sandbox directory."""
        sandbox_id = str(uuid.uuid4())[:8]
        sandbox_path = Path(tempfile.mkdtemp(prefix=f"blackice_sandbox_{sandbox_id}_"))
        self._sandboxes[sandbox_id] = sandbox_path
        return sandbox_id, sandbox_path

    def _cleanup_sandbox(self, sandbox_id: str) -> None:
        """Clean up a sandbox directory."""
        if sandbox_id in self._sandboxes:
            import shutil

            sandbox_path = self._sandboxes[sandbox_id]
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path, ignore_errors=True)
            del self._sandboxes[sandbox_id]

    async def execute(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
        sandbox_id: str | None = None,
        persist_sandbox: bool = False,
    ) -> ExecutionResult:
        """Execute a command in an isolated sandbox.

        Args:
            command: Command to execute
            config: Execution configuration
            sandbox_id: Reuse existing sandbox (if provided)
            persist_sandbox: Keep sandbox after execution

        Returns:
            ExecutionResult with output and sandbox_id in context
        """
        config = self._merge_config(config)
        cmd_list = self._normalize_command(command)
        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Create or reuse sandbox
        if sandbox_id and sandbox_id in self._sandboxes:
            sandbox_path = self._sandboxes[sandbox_id]
        else:
            sandbox_id, sandbox_path = self._create_sandbox()

        # Use sandbox as working directory if not specified
        working_dir = config.working_dir or sandbox_path

        # Build restricted environment
        env = {
            "HOME": str(sandbox_path),
            "TMPDIR": str(sandbox_path),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        env.update(config.env)

        start = time.monotonic()
        timed_out = False

        try:
            if config.shell:
                process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE if config.capture_stderr else None,
                    cwd=working_dir,
                    env=env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd_list,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE if config.capture_stderr else None,
                    cwd=working_dir,
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
                stderr_bytes = b"Sandbox execution timed out"

            duration = time.monotonic() - start

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
                killed=False,
            )

        except Exception as e:
            duration = time.monotonic() - start
            raise ExecutionError(
                f"Sandbox execution failed: {e}",
                context={
                    "command": cmd_str,
                    "sandbox_id": sandbox_id,
                    "duration": duration,
                },
            ) from e

        finally:
            if not persist_sandbox and self.cleanup_on_exit:
                self._cleanup_sandbox(sandbox_id)

    async def stream(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream command output from a sandbox."""
        config = self._merge_config(config)
        cmd_list = self._normalize_command(command)
        cmd_str = command if isinstance(command, str) else " ".join(command)

        sandbox_id, sandbox_path = self._create_sandbox()
        working_dir = config.working_dir or sandbox_path

        env = {
            "HOME": str(sandbox_path),
            "TMPDIR": str(sandbox_path),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        env.update(config.env)

        try:
            if config.shell:
                process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT if config.capture_stderr else None,
                    cwd=working_dir,
                    env=env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd_list,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT if config.capture_stderr else None,
                    cwd=working_dir,
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
                    yield "[SANDBOX TIMED OUT]"
                    break
                if not line:
                    break
                yield line.decode("utf-8", errors="replace").rstrip("\n")

            await process.wait()

        except Exception as e:
            raise ExecutionError(
                f"Sandbox streaming failed: {e}",
                context={"command": cmd_str, "sandbox_id": sandbox_id},
            ) from e

        finally:
            if self.cleanup_on_exit:
                self._cleanup_sandbox(sandbox_id)

    async def health(self) -> HealthStatus:
        """Check sandbox execution health."""
        start = time.monotonic()
        try:
            result = await self.execute(["echo", "health"])
            latency = (time.monotonic() - start) * 1000

            return HealthStatus(
                healthy=result.exit_code == 0,
                environment=ExecutionEnvironment.SANDBOX,
                latency_ms=latency,
                details={
                    "provider": self.name,
                    "active_sandboxes": len(self._sandboxes),
                },
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                environment=ExecutionEnvironment.SANDBOX,
                error=str(e),
                details={"provider": self.name},
            )

    async def cleanup_all(self) -> int:
        """Clean up all sandboxes. Returns count of cleaned sandboxes."""
        count = len(self._sandboxes)
        sandbox_ids = list(self._sandboxes.keys())
        for sandbox_id in sandbox_ids:
            self._cleanup_sandbox(sandbox_id)
        return count
