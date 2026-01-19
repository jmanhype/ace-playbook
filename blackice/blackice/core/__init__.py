"""Core infrastructure for BLACKICE 3.0.

Provides retry logic, budget management, cancellation support, and configuration.
"""

from blackice.core.config import (
    BlackiceConfig,
    Edition,
    ExecutionConfig,
    FlywheelSettings,
    LogLevel,
    ModelConfig,
    SafetyConfig,
    get_config,
    set_config,
)
from blackice.core.budget import (
    BudgetExceededError,
    BudgetLimits,
    BudgetManager,
    BudgetStatus,
    UsageRecord,
)
from blackice.core.cancellation import (
    CancellableOperation,
    CancellationError,
    CancellationToken,
    CancellationTokenSource,
    with_cancellation,
)
from blackice.core.retry import (
    RetryConfig,
    RetryContext,
    RetryExhaustedError,
    RetryStats,
    retry_async,
    retry_sync,
)
from blackice.core.providers import (
    ProviderSet,
    create_execution_provider,
    create_memory_provider,
    create_model_provider,
    create_provider_set,
    verify_ai_factory_connection,
)

__all__ = [
    # Config
    "BlackiceConfig",
    "Edition",
    "LogLevel",
    "ModelConfig",
    "ExecutionConfig",
    "SafetyConfig",
    "FlywheelSettings",
    "get_config",
    "set_config",
    # Retry
    "RetryConfig",
    "RetryStats",
    "RetryExhaustedError",
    "retry_async",
    "retry_sync",
    "RetryContext",
    # Budget
    "BudgetExceededError",
    "UsageRecord",
    "BudgetLimits",
    "BudgetStatus",
    "BudgetManager",
    # Cancellation
    "CancellationError",
    "CancellationToken",
    "CancellationTokenSource",
    "with_cancellation",
    "CancellableOperation",
    # Providers
    "ProviderSet",
    "create_model_provider",
    "create_memory_provider",
    "create_execution_provider",
    "create_provider_set",
    "verify_ai_factory_connection",
]
