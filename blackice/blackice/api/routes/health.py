"""Health and status endpoints for BLACKICE API."""

from __future__ import annotations

import time

from fastapi import APIRouter

from blackice.api.deps import ConfigDep
from blackice.api.schemas import HealthResponse, ProviderInfo, ProvidersResponse, ProviderType
from blackice.core.providers import create_memory_provider, create_model_provider

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check(config: ConfigDep) -> HealthResponse:
    """Check overall system health."""
    start = time.monotonic()
    providers_status: dict[str, bool] = {}

    # Check Ollama
    try:
        ollama = create_model_provider(provider_type="ollama")
        health = await ollama.health()
        providers_status["ollama"] = health.healthy
        await ollama.close()
    except Exception:
        providers_status["ollama"] = False

    # Check Letta
    try:
        letta = create_memory_provider()
        health = await letta.health()
        providers_status["letta"] = health.healthy
        await letta.close()
    except Exception:
        providers_status["letta"] = False

    latency = (time.monotonic() - start) * 1000
    all_healthy = all(providers_status.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="3.0.0",
        providers=providers_status,
        latency_ms=latency,
    )


@router.get("/live")
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness(config: ConfigDep) -> dict[str, str]:
    """Kubernetes readiness probe."""
    # Quick check - just verify config is loaded
    if config.ollama.host:
        return {"status": "ready"}
    return {"status": "not_ready"}


providers_router = APIRouter(prefix="/providers", tags=["providers"])


@providers_router.get("", response_model=ProvidersResponse)
async def list_providers(config: ConfigDep) -> ProvidersResponse:
    """List all available LLM providers with their status."""
    providers: list[ProviderInfo] = []

    # Check each provider type
    provider_configs = [
        ("ollama", ProviderType.OLLAMA, config.ollama.base_url, "qwen2.5-coder:32b-instruct-q4_K_M"),
        ("claude-max", ProviderType.CLAUDE_MAX, config.claude_router.base_url, "claude-sonnet-4-20250514"),
        ("zhipu", ProviderType.ZHIPU, "https://open.bigmodel.cn/api/coding/paas", "glm-4.5"),
        ("z.ai", ProviderType.ZAI, "https://open.bigmodel.cn/api/coding/paas", "glm-4.5"),
    ]

    for provider_type_str, provider_type, base_url, default_model in provider_configs:
        try:
            provider = create_model_provider(provider_type=provider_type_str)
            health = await provider.health()
            providers.append(
                ProviderInfo(
                    name=provider_type_str,
                    type=provider_type,
                    healthy=health.healthy,
                    base_url=base_url,
                    model=default_model,
                    latency_ms=health.latency_ms,
                )
            )
            await provider.close()
        except Exception as e:
            providers.append(
                ProviderInfo(
                    name=provider_type_str,
                    type=provider_type,
                    healthy=False,
                    base_url=base_url,
                    model=default_model,
                )
            )

    return ProvidersResponse(
        providers=providers,
        default=ProviderType.OLLAMA,
    )


@providers_router.get("/{provider_type}", response_model=ProviderInfo)
async def get_provider(provider_type: str, config: ConfigDep) -> ProviderInfo:
    """Get details about a specific provider."""
    try:
        provider = create_model_provider(provider_type=provider_type)
        health = await provider.health()
        await provider.close()

        return ProviderInfo(
            name=provider_type,
            type=ProviderType(provider_type) if provider_type in [e.value for e in ProviderType] else ProviderType.OLLAMA,
            healthy=health.healthy,
            base_url=provider.base_url,
            model=provider.model,
            latency_ms=health.latency_ms,
        )
    except Exception as e:
        return ProviderInfo(
            name=provider_type,
            type=ProviderType.OLLAMA,
            healthy=False,
        )
