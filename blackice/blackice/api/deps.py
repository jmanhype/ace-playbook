"""API dependencies for BLACKICE 3.0.

Dependency injection for FastAPI endpoints.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from blackice.core.providers import (
    create_execution_provider,
    create_memory_provider,
    create_model_provider,
)
from blackice.infrastructure import AIFactoryConfig, get_ai_factory_config


@lru_cache
def get_config() -> AIFactoryConfig:
    """Get AI Factory configuration (cached)."""
    return get_ai_factory_config()


async def get_model_provider(
    provider_type: str = "ollama",
    model: str | None = None,
):
    """Get model provider for a request."""
    provider = create_model_provider(
        provider_type=provider_type,
        model=model,
    )
    try:
        yield provider
    finally:
        await provider.close()


async def get_memory_provider():
    """Get memory provider (Letta)."""
    provider = create_memory_provider()
    try:
        yield provider
    finally:
        await provider.close()


async def get_execution_provider(workspace: str | None = None):
    """Get execution provider."""
    provider = create_execution_provider(workspace)
    yield provider


# Type aliases for dependency injection
ConfigDep = Annotated[AIFactoryConfig, Depends(get_config)]
