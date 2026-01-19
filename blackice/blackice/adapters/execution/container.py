"""Container execution provider for BLACKICE 3.0.

Implements the ExecutionProvider protocol for Docker container execution.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncIterator

from blackice.adapters.execution.base import (
    BaseExecutionProvider,
    ExecutionCapabilities,
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionResult,
    HealthStatus,
)
from blackice.primitives.errors import ExecutionError


class ContainerExecutionProvider(BaseExecutionProvider):
    """Container execution provider using Docker.

    Executes commands inside Docker containers with isolation,
    resource limits, and network control.
    """

    DEFAULT_IMAGE = "python:3.11-slim"

    def __init__(
        self,
        image: str | None = None,
        default_timeout: float = 300.0,
        default_working_dir: Path | None = None,
        network_mode: str = "none",
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        auto_remove: bool = True,
    ) -> None:
        super().__init__(default_timeout, default_working_dir)
        self.image = image or self.DEFAULT_IMAGE
        self.network_mode = network_mode
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.auto_remove = auto_remove
        self._container_id: str | None = None

    @property
    def name(self) -> str:
        return "container"

    @property
    def capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            environment=ExecutionEnvironment.CONTAINER,
            supports_streaming=True,
            supports_attach=True,
            supports_isolation=True,
            max_timeout=3600.0,
            max_concurrent=20,
        )

    def _build_docker_command(
        self,
        command: list[str],
        config: ExecutionConfig,
    ) -> list[str]:
        """Build the docker run command."""
        docker_cmd = [
            "docker", "run",
            "--rm" if self.auto_remove else "",
            f"--network={self.network_mode}",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
        ]

        # Remove empty strings
        docker_cmd = [c for c in docker_cmd if c]

        # Add working directory
        if config.working_dir:
            docker_cmd.extend(["-w", str(config.working_dir)])

        # Add environment variables
        for key, value in config.env.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Add image and command
        docker_cmd.append(self.image)
        docker_cmd.extend(command)

        return docker_cmd

    async def execute(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> ExecutionResult:
        """Execute a command in a Docker container."""
        config = self._merge_config(config)
        cmd_list = self._normalize_command(command)
        cmd_str = command if isinstance(command, str) else " ".join(command)

        docker_cmd = self._build_docker_command(cmd_list, config)

        start = time.monotonic()
        timed_out = False

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if config.capture_stderr else None,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                # Kill the docker container
                if self._container_id:
                    kill_process = await asyncio.create_subprocess_exec(
                        "docker", "kill", self._container_id,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await kill_process.wait()
                process.kill()
                await process.wait()
                stdout_bytes = b""
                stderr_bytes = b"Container killed due to timeout"

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
                f"Container execution failed: {e}",
                context={
                    "command": cmd_str,
                    "image": self.image,
                    "duration": duration,
                },
            ) from e

    async def stream(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream command output from a Docker container."""
        config = self._merge_config(config)
        cmd_list = self._normalize_command(command)
        cmd_str = command if isinstance(command, str) else " ".join(command)

        docker_cmd = self._build_docker_command(cmd_list, config)

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT if config.capture_stderr else None,
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
                    yield "[CONTAINER TIMED OUT]"
                    break
                if not line:
                    break
                yield line.decode("utf-8", errors="replace").rstrip("\n")

            await process.wait()

        except Exception as e:
            raise ExecutionError(
                f"Container streaming failed: {e}",
                context={"command": cmd_str, "image": self.image},
            ) from e

    async def attach(
        self,
        session_id: str,
    ) -> tuple[AsyncIterator[str], Any]:
        """Attach to a running container."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "attach", session_id,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            assert process.stdout is not None
            assert process.stdin is not None

            async def output_iterator() -> AsyncIterator[str]:
                while True:
                    line = await process.stdout.readline()  # type: ignore
                    if not line:
                        break
                    yield line.decode("utf-8", errors="replace").rstrip("\n")

            async def write(data: str) -> None:
                process.stdin.write(data.encode())  # type: ignore
                await process.stdin.drain()  # type: ignore

            return output_iterator(), write

        except Exception as e:
            raise ExecutionError(
                f"Container attach failed: {e}",
                context={"session_id": session_id},
            ) from e

    async def health(self) -> HealthStatus:
        """Check Docker availability."""
        start = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            latency = (time.monotonic() - start) * 1000

            if process.returncode != 0:
                return HealthStatus(
                    healthy=False,
                    environment=ExecutionEnvironment.CONTAINER,
                    error="Docker daemon not running",
                    details={"provider": self.name},
                )

            return HealthStatus(
                healthy=True,
                environment=ExecutionEnvironment.CONTAINER,
                latency_ms=latency,
                details={
                    "provider": self.name,
                    "image": self.image,
                    "network_mode": self.network_mode,
                },
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                environment=ExecutionEnvironment.CONTAINER,
                error=str(e),
                details={"provider": self.name},
            )

    async def pull_image(self) -> bool:
        """Pull the container image if not present."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "pull", self.image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
            return process.returncode == 0
        except Exception:
            return False
