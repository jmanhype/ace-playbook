"""Secret redaction utilities for BLACKICE 3.0.

Provides comprehensive redaction of secrets from text, logs, and outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class RedactionLevel(str, Enum):
    """Redaction aggressiveness levels."""

    MINIMAL = "minimal"  # Only known secrets
    STANDARD = "standard"  # Known secrets + common patterns
    AGGRESSIVE = "aggressive"  # All potential secrets
    PARANOID = "paranoid"  # Anything that looks secret-like


@dataclass
class RedactionPattern:
    """A pattern for secret redaction."""

    name: str
    pattern: str
    flags: int = re.IGNORECASE
    replacement: str | None = None  # None means use global replacement
    priority: int = 0  # Higher = matched first


# Pre-compiled patterns for performance
SECRET_PATTERNS: list[RedactionPattern] = [
    # API Keys - High priority
    RedactionPattern(
        "anthropic_api_key",
        r"sk-ant-[a-zA-Z0-9_-]{32,}",
        priority=100,
    ),
    RedactionPattern(
        "openai_api_key",
        r"sk-[a-zA-Z0-9]{32,}",
        priority=100,
    ),
    RedactionPattern(
        "openai_org",
        r"org-[a-zA-Z0-9]{24,}",
        priority=100,
    ),
    RedactionPattern(
        "stripe_key",
        r"(sk|pk)_(test|live)_[a-zA-Z0-9]{24,}",
        priority=100,
    ),
    RedactionPattern(
        "aws_access_key",
        r"AKIA[0-9A-Z]{16}",
        priority=100,
    ),
    RedactionPattern(
        "aws_secret_key",
        r"[a-zA-Z0-9/+=]{40}",
        priority=50,  # Lower priority - more generic
    ),
    RedactionPattern(
        "github_token",
        r"(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}",
        priority=100,
    ),
    RedactionPattern(
        "github_oauth",
        r"gho_[a-zA-Z0-9]{36}",
        priority=100,
    ),
    RedactionPattern(
        "slack_token",
        r"xox[baprs]-[a-zA-Z0-9-]{10,}",
        priority=100,
    ),
    RedactionPattern(
        "google_api_key",
        r"AIza[a-zA-Z0-9_-]{35}",
        priority=100,
    ),
    RedactionPattern(
        "firebase_key",
        r"[a-zA-Z0-9]{40}",  # Generic but combined with context
        priority=30,
    ),
    # Tokens - Medium priority
    RedactionPattern(
        "bearer_token",
        r"Bearer\s+[a-zA-Z0-9._-]+",
        priority=90,
    ),
    RedactionPattern(
        "jwt_token",
        r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
        priority=95,
    ),
    RedactionPattern(
        "basic_auth",
        r"Basic\s+[a-zA-Z0-9+/=]+",
        priority=90,
    ),
    # Connection strings
    RedactionPattern(
        "database_url",
        r"(postgres|mysql|mongodb|redis)://[^\s]+:[^\s]+@[^\s]+",
        priority=85,
    ),
    RedactionPattern(
        "connection_string",
        r"(Server|Host|Data Source)=[^;]+;.*Password=[^;]+",
        priority=80,
    ),
    # Private keys
    RedactionPattern(
        "private_key_header",
        r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        priority=100,
    ),
    RedactionPattern(
        "private_key_content",
        r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----",
        priority=100,
    ),
    # Generic patterns - Low priority
    RedactionPattern(
        "hex_secret",
        r"\b[a-f0-9]{32,}\b",
        priority=20,
    ),
    RedactionPattern(
        "base64_secret",
        r"\b[a-zA-Z0-9+/]{32,}={0,2}\b",
        priority=15,
    ),
    # Assignment patterns
    RedactionPattern(
        "key_value_secret",
        r'(api[_-]?key|secret|token|password|credential|auth)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{16,})["\']?',
        priority=70,
    ),
    RedactionPattern(
        "env_var_secret",
        r"(API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH)=['\"]?[a-zA-Z0-9_-]+['\"]?",
        priority=70,
    ),
]

# Base patterns for standard level
_STANDARD_PATTERNS = [
    "anthropic_api_key",
    "openai_api_key",
    "openai_org",
    "stripe_key",
    "aws_access_key",
    "github_token",
    "github_oauth",
    "slack_token",
    "google_api_key",
    "bearer_token",
    "jwt_token",
    "basic_auth",
    "database_url",
    "private_key_header",
    "private_key_content",
]

# Aggressive adds more patterns
_AGGRESSIVE_PATTERNS = [
    *_STANDARD_PATTERNS,
    "connection_string",
    "key_value_secret",
    "env_var_secret",
    "aws_secret_key",
]

# Paranoid adds generic patterns
_PARANOID_PATTERNS = [
    *_AGGRESSIVE_PATTERNS,
    "hex_secret",
    "base64_secret",
    "firebase_key",
]

# Patterns by redaction level
LEVEL_PATTERNS: dict[RedactionLevel, list[str]] = {
    RedactionLevel.MINIMAL: [],  # Only known secrets
    RedactionLevel.STANDARD: _STANDARD_PATTERNS,
    RedactionLevel.AGGRESSIVE: _AGGRESSIVE_PATTERNS,
    RedactionLevel.PARANOID: _PARANOID_PATTERNS,
}


@dataclass
class RedactionResult:
    """Result of a redaction operation."""

    original_length: int
    redacted_length: int
    redacted_text: str
    redaction_count: int
    patterns_matched: list[str] = field(default_factory=list)
    known_secrets_redacted: list[str] = field(default_factory=list)


class SecretRedactor:
    """Redacts secrets from text with configurable patterns and levels."""

    def __init__(
        self,
        level: RedactionLevel = RedactionLevel.STANDARD,
        replacement: str = "[REDACTED]",
        custom_patterns: list[RedactionPattern] | None = None,
    ) -> None:
        """Initialize the redactor.

        Args:
            level: Aggressiveness of redaction
            replacement: Default replacement string
            custom_patterns: Additional patterns to include
        """
        self.level = level
        self.replacement = replacement
        self.custom_patterns = custom_patterns or []

        # Known secret values (added via add_known_secret)
        self._known_secrets: dict[str, str] = {}  # value -> name

        # Compile patterns for this level
        self._compiled: list[tuple[re.Pattern, str, str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for the current level."""
        self._compiled.clear()

        # Get patterns for this level
        enabled_names = set(LEVEL_PATTERNS.get(self.level, []))

        # Add custom patterns
        all_patterns = SECRET_PATTERNS + self.custom_patterns

        # Filter and sort by priority
        relevant = [p for p in all_patterns if p.name in enabled_names or p in self.custom_patterns]
        relevant.sort(key=lambda p: -p.priority)

        # Compile
        for pattern in relevant:
            try:
                compiled = re.compile(pattern.pattern, pattern.flags)
                replacement = pattern.replacement or self.replacement
                self._compiled.append((compiled, pattern.name, replacement))
            except re.error:
                # Skip invalid patterns
                pass

    def add_known_secret(self, value: str, name: str | None = None) -> None:
        """Add a known secret value for exact-match redaction.

        Args:
            value: The secret value to redact
            name: Optional name for logging/tracking
        """
        if value and len(value) >= 4:  # Only track meaningful values
            self._known_secrets[value] = name or f"secret_{len(self._known_secrets)}"

    def remove_known_secret(self, value: str) -> bool:
        """Remove a known secret value.

        Returns:
            True if the secret was found and removed
        """
        if value in self._known_secrets:
            del self._known_secrets[value]
            return True
        return False

    def clear_known_secrets(self) -> None:
        """Clear all known secrets."""
        self._known_secrets.clear()

    def redact(
        self,
        text: str,
        *,
        replacement: str | None = None,
        track_patterns: bool = True,
    ) -> RedactionResult:
        """Redact secrets from text.

        Args:
            text: Text that may contain secrets
            replacement: Override default replacement
            track_patterns: Whether to track which patterns matched

        Returns:
            RedactionResult with cleaned text and statistics
        """
        if not text:
            return RedactionResult(
                original_length=0,
                redacted_length=0,
                redacted_text="",
                redaction_count=0,
            )

        replacement = replacement or self.replacement
        redacted = text
        count = 0
        patterns_matched: list[str] = []
        secrets_redacted: list[str] = []

        # First pass: Known secrets (exact match)
        for value, name in self._known_secrets.items():
            if value in redacted:
                occurrences = redacted.count(value)
                redacted = redacted.replace(value, replacement)
                count += occurrences
                if track_patterns:
                    secrets_redacted.append(name)

        # Second pass: Pattern matching
        for compiled, name, pattern_replacement in self._compiled:
            matches = compiled.findall(redacted)
            if matches:
                redacted = compiled.sub(pattern_replacement or replacement, redacted)
                count += len(matches)
                if track_patterns and name not in patterns_matched:
                    patterns_matched.append(name)

        return RedactionResult(
            original_length=len(text),
            redacted_length=len(redacted),
            redacted_text=redacted,
            redaction_count=count,
            patterns_matched=patterns_matched,
            known_secrets_redacted=secrets_redacted,
        )

    def redact_dict(
        self,
        data: dict,
        *,
        keys_to_redact: set[str] | None = None,
        deep: bool = True,
    ) -> dict:
        """Redact secrets from a dictionary.

        Args:
            data: Dictionary to redact
            keys_to_redact: Specific keys to always redact values for
            deep: Whether to recursively process nested dicts

        Returns:
            New dictionary with redacted values
        """
        default_keys = {
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "auth",
            "credential",
            "private_key",
            "access_key",
            "secret_key",
        }
        keys_to_redact = keys_to_redact or default_keys

        def should_redact_key(key: str) -> bool:
            key_lower = key.lower()
            return any(k in key_lower for k in keys_to_redact)

        def process_value(key: str, value) -> any:
            if isinstance(value, str):
                if should_redact_key(key):
                    return self.replacement
                return self.redact(value).redacted_text
            elif isinstance(value, dict) and deep:
                return self.redact_dict(value, keys_to_redact=keys_to_redact, deep=deep)
            elif isinstance(value, list) and deep:
                return [
                    process_value(key, item) if isinstance(item, (str, dict, list)) else item
                    for item in value
                ]
            return value

        return {key: process_value(key, value) for key, value in data.items()}


def create_log_filter(redactor: SecretRedactor) -> Callable[[str], str]:
    """Create a log filter function for use with logging handlers.

    Args:
        redactor: The redactor to use

    Returns:
        A function that redacts secrets from log messages
    """

    def filter_log(message: str) -> str:
        return redactor.redact(message).redacted_text

    return filter_log


# Singleton redactor for convenience
_default_redactor: SecretRedactor | None = None


def get_default_redactor() -> SecretRedactor:
    """Get the default redactor instance."""
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = SecretRedactor(level=RedactionLevel.STANDARD)
    return _default_redactor


def redact(text: str, *, replacement: str = "[REDACTED]") -> str:
    """Convenience function to redact secrets using the default redactor.

    Args:
        text: Text to redact
        replacement: Replacement string

    Returns:
        Redacted text
    """
    return get_default_redactor().redact(text, replacement=replacement).redacted_text


def register_secret(value: str, name: str | None = None) -> None:
    """Register a known secret with the default redactor.

    Args:
        value: Secret value to track
        name: Optional name for the secret
    """
    get_default_redactor().add_known_secret(value, name)
