"""BLACKICE Configuration Management.

Central configuration for the BLACKICE system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Edition(str, Enum):
    """BLACKICE edition tiers."""

    LITE = "lite"
    CORE = "core"
    ENTERPRISE = "enterprise"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ModelConfig(BaseModel):
    """Model provider configuration."""

    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model to use for generation",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens per response",
    )
    timeout: float = Field(
        default=120.0,
        gt=0,
        description="Request timeout in seconds",
    )


class ExecutionConfig(BaseModel):
    """Execution provider configuration."""

    default_provider: str = Field(
        default="local",
        description="Default execution provider (local, container, sandbox)",
    )
    command_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Default command timeout in seconds",
    )
    allow_network: bool = Field(
        default=True,
        description="Allow network access in execution",
    )
    allow_filesystem: bool = Field(
        default=True,
        description="Allow filesystem access in execution",
    )


class SafetyConfig(BaseModel):
    """Safety pipeline configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable safety pipeline",
    )
    strict_mode: bool = Field(
        default=False,
        description="Enable strict mode (block more commands)",
    )
    log_blocked: bool = Field(
        default=True,
        description="Log blocked commands",
    )


class FlywheelSettings(BaseModel):
    """Flywheel execution settings."""

    plan_timeout: float = Field(
        default=300.0,
        description="Planning phase timeout",
    )
    implement_timeout: float = Field(
        default=1800.0,
        description="Implementation phase timeout",
    )
    test_timeout: float = Field(
        default=600.0,
        description="Testing phase timeout",
    )
    verify_timeout: float = Field(
        default=300.0,
        description="Verification phase timeout",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum phase retries",
    )
    min_coverage: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Minimum test coverage percentage",
    )


class BlackiceConfig(BaseModel):
    """Main BLACKICE configuration."""

    # Core settings
    edition: Edition = Field(
        default=Edition.LITE,
        description="BLACKICE edition tier",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    workspace_root: Path = Field(
        default=Path(".blackice"),
        description="Root directory for workspaces",
    )

    # Sub-configurations
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model provider settings",
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Execution provider settings",
    )
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety pipeline settings",
    )
    flywheel: FlywheelSettings = Field(
        default_factory=FlywheelSettings,
        description="Flywheel settings",
    )

    # Feature flags
    enable_event_sourcing: bool = Field(
        default=False,
        description="Enable event sourcing (Core+)",
    )
    enable_spec_first: bool = Field(
        default=False,
        description="Enable spec-first execution (Enterprise)",
    )
    enable_verifiable_receipts: bool = Field(
        default=False,
        description="Enable verifiable receipts (Enterprise)",
    )

    model_config = {"extra": "ignore"}

    @classmethod
    def from_env(cls) -> "BlackiceConfig":
        """Create configuration from environment variables."""
        edition_str = os.environ.get("BLACKICE_EDITION", "lite")
        try:
            edition = Edition(edition_str.lower())
        except ValueError:
            edition = Edition.LITE

        log_level_str = os.environ.get("BLACKICE_LOG_LEVEL", "info")
        try:
            log_level = LogLevel(log_level_str.lower())
        except ValueError:
            log_level = LogLevel.INFO

        workspace_str = os.environ.get("BLACKICE_WORKSPACE", ".blackice")

        # Apply edition-specific defaults
        enable_event_sourcing = edition in (Edition.CORE, Edition.ENTERPRISE)
        enable_spec_first = edition == Edition.ENTERPRISE
        enable_verifiable_receipts = edition == Edition.ENTERPRISE

        return cls(
            edition=edition,
            log_level=log_level,
            workspace_root=Path(workspace_str),
            enable_event_sourcing=enable_event_sourcing,
            enable_spec_first=enable_spec_first,
            enable_verifiable_receipts=enable_verifiable_receipts,
        )

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled for the current edition."""
        feature_map = {
            "event_sourcing": self.enable_event_sourcing,
            "spec_first": self.enable_spec_first,
            "verifiable_receipts": self.enable_verifiable_receipts,
            "multi_agent": self.edition in (Edition.CORE, Edition.ENTERPRISE),
            "memory_provider": True,  # Always available
            "safety_pipeline": True,  # Always available
        }
        return feature_map.get(feature, False)

    def require_edition(self, required: Edition) -> None:
        """Raise an error if the current edition doesn't meet requirements."""
        edition_order = [Edition.LITE, Edition.CORE, Edition.ENTERPRISE]
        current_idx = edition_order.index(self.edition)
        required_idx = edition_order.index(required)

        if current_idx < required_idx:
            from blackice.primitives.errors import EditionError
            raise EditionError(
                feature="requested feature",
                required=required.value,
                current=self.edition.value,
            )


# Global configuration instance
_config: BlackiceConfig | None = None


def get_config() -> BlackiceConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = BlackiceConfig.from_env()
    return _config


def set_config(config: BlackiceConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
