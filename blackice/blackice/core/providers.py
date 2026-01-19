"""Provider Factory for BLACKICE 3.0.

Central factory for creating pre-configured providers connected to the AI Factory
infrastructure at 192.168.1.143 (or via WireGuard at 10.0.0.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Literal, Union

from blackice.adapters.execution import LocalExecutionProvider
from blackice.adapters.memory import LettaMemoryProvider
from blackice.adapters.models.base import BaseModelProvider
from blackice.adapters.models.claude import ClaudeProvider
from blackice.adapters.models.ollama import OllamaProvider
from blackice.adapters.models.openai import OpenAIProvider
from blackice.infrastructure import AIFactoryConfig, get_ai_factory_config


# Type alias for provider selection
ProviderType = Literal["ollama", "claude", "openai", "z.ai"]


@dataclass
class ProviderSet:
    """A complete set of providers for BLACKICE operations.

    Contains all providers needed for the flywheel:
    - model_provider: For LLM inference (planning, coding, verification)
    - memory_provider: For persistent memory and context
    - execution_provider: For running commands and tests
    """

    model_provider: BaseModelProvider
    memory_provider: LettaMemoryProvider
    execution_provider: LocalExecutionProvider

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers.

        Returns:
            Dict mapping provider names to health status
        """
        results = {}

        # Model provider
        model_health = await self.model_provider.health()
        results["model"] = model_health.healthy

        # Memory provider
        memory_health = await self.memory_provider.health()
        results["memory"] = memory_health.healthy

        # Execution provider
        exec_health = await self.execution_provider.health()
        results["execution"] = exec_health.healthy

        return results

    async def close(self) -> None:
        """Close all provider connections."""
        await self.model_provider.close()
        await self.memory_provider.close()
        # LocalExecutionProvider doesn't need closing


def create_model_provider(
    config: AIFactoryConfig | None = None,
    model: str | None = None,
    provider_type: ProviderType = "ollama",
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseModelProvider:
    """Create a model provider of the specified type.

    Args:
        config: AI Factory configuration (uses global if not provided)
        model: Model name override (defaults vary by provider)
        provider_type: Provider to use ("ollama", "claude", "openai", "z.ai")
        api_key: API key (required for Claude/OpenAI, uses env var if not provided)
        base_url: Base URL override (for z.ai or custom endpoints)

    Returns:
        Configured model provider (OllamaProvider, ClaudeProvider, or OpenAIProvider)

    Example:
        ```python
        # Use local Ollama (default)
        provider = create_model_provider()

        # Use Claude API
        provider = create_model_provider(provider_type="claude")

        # Use z.ai (OpenAI-compatible)
        provider = create_model_provider(
            provider_type="z.ai",
            base_url="https://api.zeroone.ai",
            api_key="your-api-key",
        )
        ```
    """
    if provider_type == "ollama":
        if config is None:
            config = get_ai_factory_config()

        return OllamaProvider(
            base_url=base_url or config.ollama.base_url,
            model=model or config.ollama.default_model,
            timeout=config.ollama.timeout,
        )

    elif provider_type == "claude":
        return ClaudeProvider(
            api_key=api_key,  # Falls back to ANTHROPIC_API_KEY env var
            base_url=base_url,  # Falls back to default Anthropic API
            model=model,  # Falls back to claude-sonnet-4
            timeout=120.0,
        )

    elif provider_type in ("openai", "z.ai"):
        # z.ai is OpenAI-compatible, just needs different base_url
        effective_base_url = base_url
        if provider_type == "z.ai" and not base_url:
            effective_base_url = "https://api.zeroone.ai"

        return OpenAIProvider(
            api_key=api_key,  # Falls back to OPENAI_API_KEY env var
            base_url=effective_base_url,  # Falls back to OpenAI default
            model=model,  # Falls back to gpt-4o
            timeout=120.0,
        )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Use 'ollama', 'claude', 'openai', or 'z.ai'")


def create_memory_provider(
    config: AIFactoryConfig | None = None,
    agent_id: str | None = None,
) -> LettaMemoryProvider:
    """Create a memory provider connected to Letta on the AI Factory.

    Args:
        config: AI Factory configuration (uses global if not provided)
        agent_id: Optional Letta agent ID to use

    Returns:
        Configured LettaMemoryProvider

    Example:
        ```python
        provider = create_memory_provider()
        await provider.put(MemoryEntry(
            id="mem-001",
            memory_type=MemoryType.PATTERN,
            content="Use pytest for testing",
        ))
        ```
    """
    if config is None:
        config = get_ai_factory_config()

    return LettaMemoryProvider(
        base_url=config.letta.base_url,
        api_token=config.letta.api_token,
        timeout=config.letta.timeout,
        agent_id=agent_id,
    )


def create_execution_provider(
    working_dir: str | None = None,
    timeout: float = 300.0,
) -> LocalExecutionProvider:
    """Create a local execution provider.

    Args:
        working_dir: Working directory for command execution
        timeout: Default timeout for commands

    Returns:
        Configured LocalExecutionProvider

    Example:
        ```python
        provider = create_execution_provider()
        result = await provider.execute("pytest tests/ -v")
        ```
    """
    from pathlib import Path

    working_path = Path(working_dir) if working_dir else None
    return LocalExecutionProvider(
        default_working_dir=working_path,
        default_timeout=timeout,
    )


def create_provider_set(
    config: AIFactoryConfig | None = None,
    model: str | None = None,
    working_dir: str | None = None,
    agent_id: str | None = None,
    provider_type: ProviderType = "ollama",
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProviderSet:
    """Create a complete set of providers for BLACKICE.

    This is the primary entry point for setting up all providers
    needed for the flywheel.

    Args:
        config: AI Factory configuration (uses global if not provided)
        model: Model name override
        working_dir: Working directory for execution
        agent_id: Letta agent ID for memory
        provider_type: Model provider type ("ollama", "claude", "openai", "z.ai")
        api_key: API key for cloud providers
        base_url: Base URL override for custom endpoints

    Returns:
        ProviderSet with all configured providers

    Example:
        ```python
        # Use local Ollama (default)
        providers = create_provider_set()

        # Use Claude API
        providers = create_provider_set(provider_type="claude")

        # Check all providers are healthy
        health = await providers.health_check()
        if all(health.values()):
            print("All providers ready!")

        # Use providers
        result = await providers.model_provider.chat(messages)
        await providers.memory_provider.put(entry)

        # Cleanup
        await providers.close()
        ```
    """
    if config is None:
        config = get_ai_factory_config()

    return ProviderSet(
        model_provider=create_model_provider(
            config=config,
            model=model,
            provider_type=provider_type,
            api_key=api_key,
            base_url=base_url,
        ),
        memory_provider=create_memory_provider(config, agent_id),
        execution_provider=create_execution_provider(working_dir),
    )


async def verify_ai_factory_connection(
    config: AIFactoryConfig | None = None,
    check_memory: bool = True,
) -> dict[str, dict]:
    """Verify connection to all AI Factory services.

    Performs health checks on Ollama and Letta to ensure
    the AI Factory is reachable and operational.

    Args:
        config: AI Factory configuration (uses global if not provided)
        check_memory: Whether to check Letta memory provider (default: True)

    Returns:
        Dict with health status for each service and 'all_healthy' summary

    Example:
        ```python
        status = await verify_ai_factory_connection()
        if status['all_healthy']:
            print("AI Factory is ready!")
        print(f"Ollama: {'OK' if status['ollama']['healthy'] else 'FAIL'}")
        print(f"Letta: {'OK' if status['letta']['healthy'] else 'FAIL'}")
        ```
    """
    if config is None:
        config = get_ai_factory_config()

    results = {}
    all_healthy = True

    # Check Ollama
    model_provider = create_model_provider(config)
    try:
        health = await model_provider.health()
        results["ollama"] = {
            "healthy": health.healthy,
            "latency_ms": health.latency_ms,
            "error": health.error,
            "base_url": config.ollama.base_url,
            "model": config.ollama.default_model,
            **health.details,
        }
        if not health.healthy:
            all_healthy = False
    except Exception as e:
        results["ollama"] = {
            "healthy": False,
            "error": str(e),
            "base_url": config.ollama.base_url,
        }
        all_healthy = False
    finally:
        await model_provider.close()

    # Check Letta (optional)
    if check_memory:
        memory_provider = create_memory_provider(config)
        try:
            health = await memory_provider.health()
            results["letta"] = {
                "healthy": health.healthy,
                "latency_ms": health.latency_ms,
                "error": health.error,
                "base_url": config.letta.base_url,
                **health.details,
            }
            # Letta is not strictly required, so don't affect all_healthy
        except Exception as e:
            results["letta"] = {
                "healthy": False,
                "error": str(e),
                "base_url": config.letta.base_url,
            }
        finally:
            await memory_provider.close()

    results["all_healthy"] = all_healthy
    return results


__all__ = [
    "ProviderType",
    "ProviderSet",
    "create_model_provider",
    "create_memory_provider",
    "create_execution_provider",
    "create_provider_set",
    "verify_ai_factory_connection",
]
