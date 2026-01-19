"""Safety pipeline for command execution in BLACKICE 3.0.

Implements multi-stage safety checks to prevent dangerous
command execution: shell unwrap, semantic parse, allowlist, policy check.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from blackice.primitives.errors import SecurityError


class RiskLevel(str, Enum):
    """Risk levels for commands."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


@dataclass
class CommandAnalysis:
    """Result of analyzing a command for safety."""

    original: str
    normalized: list[str]
    risk_level: RiskLevel
    executable: str
    arguments: list[str]
    flags: dict[str, str | bool]
    risks: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None


@dataclass
class SafetyPolicy:
    """Safety policy configuration."""

    # Command allowlist (if set, only these commands are allowed)
    allowed_commands: set[str] | None = None

    # Explicit blocklist (always blocked)
    blocked_commands: set[str] = field(
        default_factory=lambda: {
            # System destruction
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero",
            ":(){ :|:& };:",  # Fork bomb
            # Privilege escalation
            "chmod 777 /",
            "chown -R",
            # Network attacks
            "nc -l",  # Listener
            "ncat -l",
            # Crypto mining
            "xmrig",
            "minerd",
            # Reverse shells
            "bash -i >& /dev/tcp",
            "python -c 'import socket",
        }
    )

    # Dangerous patterns (regex)
    dangerous_patterns: list[str] = field(
        default_factory=lambda: [
            r"rm\s+(-rf?|--recursive).*(/|~|\$HOME)",
            r">(>)?\s*/dev/sd",
            r"curl.*\|\s*(ba)?sh",
            r"wget.*\|\s*(ba)?sh",
            r"eval\s*\(",
            r"\$\(.*\)",  # Command substitution
            r"`.*`",  # Backtick substitution
            r";\s*rm\s",  # Command chaining with rm
            r"&&\s*rm\s",
            r"\|\|\s*rm\s",
        ]
    )

    # High-risk commands requiring extra scrutiny
    high_risk_commands: set[str] = field(
        default_factory=lambda: {
            "rm",
            "mv",
            "chmod",
            "chown",
            "sudo",
            "su",
            "dd",
            "mkfs",
            "fdisk",
            "kill",
            "killall",
            "reboot",
            "shutdown",
            "init",
            "systemctl",
        }
    )

    # Safe commands (bypass most checks)
    safe_commands: set[str] = field(
        default_factory=lambda: {
            "echo",
            "cat",
            "head",
            "tail",
            "grep",
            "ls",
            "pwd",
            "date",
            "whoami",
            "hostname",
            "uname",
            "env",
            "printenv",
            "which",
            "type",
            "wc",
            "sort",
            "uniq",
            "cut",
            "awk",
            "sed",
            "tr",
            "tee",
            "diff",
            "comm",
            "join",
            "paste",
        }
    )

    # Maximum risk level allowed
    max_risk_level: RiskLevel = RiskLevel.MEDIUM

    # Allow shell features
    allow_pipes: bool = True
    allow_redirects: bool = True
    allow_background: bool = False
    allow_subshells: bool = False


class SafetyPipeline:
    """Multi-stage safety pipeline for command execution.

    Pipeline stages:
    1. Shell unwrap - Detect and handle shell wrappers
    2. Semantic parse - Parse command structure
    3. Allowlist check - Verify against allowed commands
    4. Policy check - Apply safety policies
    """

    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.policy.dangerous_patterns
        ]

    def analyze(self, command: str | list[str]) -> CommandAnalysis:
        """Analyze a command for safety.

        Args:
            command: Command string or list

        Returns:
            CommandAnalysis with risk assessment
        """
        # Normalize to string for pattern matching
        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Stage 1: Shell unwrap
        unwrapped = self._shell_unwrap(cmd_str)

        # Stage 2: Semantic parse
        parsed = self._semantic_parse(unwrapped)

        # Stage 3: Allowlist check
        if self.policy.allowed_commands:
            if parsed["executable"] not in self.policy.allowed_commands:
                return CommandAnalysis(
                    original=cmd_str,
                    normalized=parsed["args"],
                    risk_level=RiskLevel.BLOCKED,
                    executable=parsed["executable"],
                    arguments=parsed["arguments"],
                    flags=parsed["flags"],
                    blocked=True,
                    block_reason=f"Command '{parsed['executable']}' not in allowlist",
                )

        # Stage 4: Policy check
        risks: list[str] = []
        risk_level = RiskLevel.SAFE

        # Check explicit blocklist
        for blocked in self.policy.blocked_commands:
            if blocked in cmd_str:
                return CommandAnalysis(
                    original=cmd_str,
                    normalized=parsed["args"],
                    risk_level=RiskLevel.BLOCKED,
                    executable=parsed["executable"],
                    arguments=parsed["arguments"],
                    flags=parsed["flags"],
                    risks=[f"Matches blocklist: {blocked}"],
                    blocked=True,
                    block_reason=f"Command contains blocked pattern: {blocked}",
                )

        # Check dangerous patterns
        for pattern in self._compiled_patterns:
            if pattern.search(cmd_str):
                risks.append(f"Matches dangerous pattern: {pattern.pattern}")
                risk_level = RiskLevel.CRITICAL

        # Check high-risk commands
        if parsed["executable"] in self.policy.high_risk_commands:
            risks.append(f"High-risk command: {parsed['executable']}")
            if risk_level.value < RiskLevel.HIGH.value:
                risk_level = RiskLevel.HIGH

        # Check shell features
        if not self.policy.allow_pipes and "|" in cmd_str:
            risks.append("Pipes not allowed")
            risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: list(RiskLevel).index(x))

        if not self.policy.allow_redirects and (">" in cmd_str or "<" in cmd_str):
            risks.append("Redirects not allowed")
            risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: list(RiskLevel).index(x))

        if not self.policy.allow_background and "&" in cmd_str:
            risks.append("Background execution not allowed")
            risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: list(RiskLevel).index(x))

        if not self.policy.allow_subshells and ("$(" in cmd_str or "`" in cmd_str):
            risks.append("Subshells not allowed")
            risk_level = RiskLevel.HIGH

        # Safe commands get lower risk
        if parsed["executable"] in self.policy.safe_commands and not risks:
            risk_level = RiskLevel.SAFE

        # Determine if blocked
        blocked = list(RiskLevel).index(risk_level) > list(RiskLevel).index(
            self.policy.max_risk_level
        )

        return CommandAnalysis(
            original=cmd_str,
            normalized=parsed["args"],
            risk_level=risk_level,
            executable=parsed["executable"],
            arguments=parsed["arguments"],
            flags=parsed["flags"],
            risks=risks,
            blocked=blocked,
            block_reason=f"Risk level {risk_level.value} exceeds maximum {self.policy.max_risk_level.value}"
            if blocked
            else None,
        )

    def check(self, command: str | list[str]) -> CommandAnalysis:
        """Check a command and raise if blocked.

        Args:
            command: Command to check

        Returns:
            CommandAnalysis if allowed

        Raises:
            SecurityError: If command is blocked
        """
        analysis = self.analyze(command)
        if analysis.blocked:
            raise SecurityError(
                f"Command blocked: {analysis.block_reason}",
                context={
                    "command": analysis.original,
                    "risk_level": analysis.risk_level.value,
                    "risks": analysis.risks,
                },
            )
        return analysis

    def _shell_unwrap(self, command: str) -> str:
        """Unwrap shell wrappers like bash -c, sh -c, etc."""
        # Common shell wrapper patterns
        wrapper_patterns = [
            r"^(bash|sh|zsh)\s+(-c\s+)?['\"](.+)['\"]$",
            r"^/bin/(bash|sh|zsh)\s+(-c\s+)?['\"](.+)['\"]$",
            r"^env\s+(bash|sh|zsh)\s+(-c\s+)?['\"](.+)['\"]$",
        ]

        for pattern in wrapper_patterns:
            match = re.match(pattern, command.strip(), re.IGNORECASE)
            if match:
                # Return the inner command
                groups = match.groups()
                inner = groups[-1] if groups else command
                return inner.strip("'\"")

        return command

    def _semantic_parse(self, command: str) -> dict[str, Any]:
        """Parse command semantically."""
        try:
            args = shlex.split(command)
        except ValueError:
            # If parsing fails, treat the whole thing as one arg
            args = [command]

        if not args:
            return {
                "args": [],
                "executable": "",
                "arguments": [],
                "flags": {},
            }

        executable = args[0].split("/")[-1]  # Get basename
        arguments = []
        flags: dict[str, str | bool] = {}

        i = 1
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                # Long flag
                if "=" in arg:
                    key, value = arg[2:].split("=", 1)
                    flags[key] = value
                elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                    flags[arg[2:]] = args[i + 1]
                    i += 1
                else:
                    flags[arg[2:]] = True
            elif arg.startswith("-") and len(arg) > 1:
                # Short flag(s)
                for char in arg[1:]:
                    flags[char] = True
            else:
                arguments.append(arg)
            i += 1

        return {
            "args": args,
            "executable": executable,
            "arguments": arguments,
            "flags": flags,
        }


class SafeExecutor:
    """Wrapper that applies safety pipeline before execution."""

    def __init__(
        self,
        executor: Any,  # ExecutionProvider
        policy: SafetyPolicy | None = None,
    ) -> None:
        self.executor = executor
        self.pipeline = SafetyPipeline(policy)

    async def execute(
        self,
        command: str | list[str],
        **kwargs: Any,
    ) -> Any:
        """Execute with safety checks."""
        # Check command safety
        analysis = self.pipeline.check(command)

        # Log the risk assessment
        from blackice.instrumentation import get_logger

        logger = get_logger(__name__)
        logger.info(
            "Command safety check passed",
            command=analysis.original[:100],
            risk_level=analysis.risk_level.value,
            executable=analysis.executable,
        )

        # Execute via underlying provider
        return await self.executor.execute(command, **kwargs)

    async def stream(
        self,
        command: str | list[str],
        **kwargs: Any,
    ) -> Any:
        """Stream with safety checks."""
        analysis = self.pipeline.check(command)

        from blackice.instrumentation import get_logger

        logger = get_logger(__name__)
        logger.info(
            "Command safety check passed (streaming)",
            command=analysis.original[:100],
            risk_level=analysis.risk_level.value,
        )

        async for line in self.executor.stream(command, **kwargs):
            yield line
