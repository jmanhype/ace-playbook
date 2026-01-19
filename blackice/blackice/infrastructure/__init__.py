"""AI Factory Infrastructure Configuration.

Defines the connection settings for the AI Factory running on the 3090 PC.
Supports both LAN (192.168.1.143) and WireGuard VPN (10.0.0.3) access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NetworkMode(str, Enum):
    """Network mode for connecting to AI Factory."""

    LAN = "lan"  # Direct LAN access (192.168.1.143)
    VPN = "vpn"  # WireGuard VPN access (10.0.0.3)
    LOCAL = "local"  # Local development (localhost)


@dataclass
class OllamaConfig:
    """Ollama server configuration."""

    host: str = "192.168.1.143"
    port: int = 11434
    default_model: str = "qwen2.5-coder:32b-instruct-q4_K_M"
    embedding_model: str = "nomic-embed-text:latest"
    timeout: float = 300.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class LettaConfig:
    """Letta MAS server configuration."""

    host: str = "192.168.1.143"
    port: int = 8283
    api_token: str = "e9acq0WgooNgt5ncWWSJEQ"
    timeout: float = 120.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class ClaudeRouterConfig:
    """Claude Max Router configuration (Anthropic proxy via OAuth)."""

    host: str = "192.168.1.143"
    port: int = 3000
    timeout: float = 180.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class PostgresConfig:
    """PostgreSQL database configuration."""

    host: str = "192.168.1.143"
    port: int = 5432
    database: str = "ai_factory"
    user: str = "vectorgraph"
    password: str = "vectorgraph_secret"

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis cache configuration."""

    host: str = "192.168.1.143"
    port: int = 6379
    db: int = 0
    password: str | None = None

    @property
    def connection_string(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class ComfyUIConfig:
    """ComfyUI server configuration."""

    host: str = "192.168.1.143"
    port: int = 8188
    timeout: float = 600.0  # Longer timeout for image generation

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class WireGuardConfig:
    """WireGuard VPN configuration."""

    # ZimaBoard VPN gateway
    gateway_ip: str = "10.0.0.1"
    # 3090 PC on VPN
    server_ip: str = "10.0.0.3"
    # MacBook on VPN
    client_ip: str = "10.0.0.2"


@dataclass
class AIFactoryConfig:
    """Complete AI Factory infrastructure configuration.

    This represents the full stack running on the 3090 PC at 192.168.1.143.
    Can also be accessed via WireGuard VPN at 10.0.0.3.
    """

    network_mode: NetworkMode = NetworkMode.LAN
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    letta: LettaConfig = field(default_factory=LettaConfig)
    claude_router: ClaudeRouterConfig = field(default_factory=ClaudeRouterConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    comfyui: ComfyUIConfig = field(default_factory=ComfyUIConfig)
    wireguard: WireGuardConfig = field(default_factory=WireGuardConfig)

    @classmethod
    def from_env(cls) -> "AIFactoryConfig":
        """Create configuration from environment variables."""
        network_mode_str = os.environ.get("BLACKICE_NETWORK", "lan")
        try:
            network_mode = NetworkMode(network_mode_str.lower())
        except ValueError:
            network_mode = NetworkMode.LAN

        # Determine host based on network mode
        if network_mode == NetworkMode.VPN:
            host = "10.0.0.3"
        elif network_mode == NetworkMode.LOCAL:
            host = "localhost"
        else:
            host = os.environ.get("AI_FACTORY_HOST", "192.168.1.143")

        return cls(
            network_mode=network_mode,
            ollama=OllamaConfig(
                host=host,
                port=int(os.environ.get("OLLAMA_PORT", "11434")),
                default_model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M"),
            ),
            letta=LettaConfig(
                host=host,
                port=int(os.environ.get("LETTA_PORT", "8283")),
                api_token=os.environ.get("LETTA_TOKEN", "e9acq0WgooNgt5ncWWSJEQ"),
            ),
            claude_router=ClaudeRouterConfig(
                host=host,
                port=int(os.environ.get("CLAUDE_ROUTER_PORT", "3000")),
            ),
            postgres=PostgresConfig(
                host=host,
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                database=os.environ.get("POSTGRES_DB", "ai_factory"),
                user=os.environ.get("POSTGRES_USER", "vectorgraph"),
                password=os.environ.get("POSTGRES_PASSWORD", "vectorgraph_secret"),
            ),
            redis=RedisConfig(
                host=host,
                port=int(os.environ.get("REDIS_PORT", "6379")),
            ),
            comfyui=ComfyUIConfig(
                host=host,
                port=int(os.environ.get("COMFYUI_PORT", "8188")),
            ),
        )

    def switch_to_vpn(self) -> "AIFactoryConfig":
        """Return a new config using VPN addresses."""
        vpn_host = self.wireguard.server_ip
        return AIFactoryConfig(
            network_mode=NetworkMode.VPN,
            ollama=OllamaConfig(host=vpn_host, port=self.ollama.port),
            letta=LettaConfig(host=vpn_host, port=self.letta.port, api_token=self.letta.api_token),
            claude_router=ClaudeRouterConfig(host=vpn_host, port=self.claude_router.port),
            postgres=PostgresConfig(
                host=vpn_host,
                port=self.postgres.port,
                database=self.postgres.database,
                user=self.postgres.user,
                password=self.postgres.password,
            ),
            redis=RedisConfig(host=vpn_host, port=self.redis.port),
            comfyui=ComfyUIConfig(host=vpn_host, port=self.comfyui.port),
            wireguard=self.wireguard,
        )


# Global configuration instance
_config: AIFactoryConfig | None = None


def get_ai_factory_config() -> AIFactoryConfig:
    """Get the global AI Factory configuration."""
    global _config
    if _config is None:
        _config = AIFactoryConfig.from_env()
    return _config


def set_ai_factory_config(config: AIFactoryConfig) -> None:
    """Set the global AI Factory configuration."""
    global _config
    _config = config


__all__ = [
    "NetworkMode",
    "OllamaConfig",
    "LettaConfig",
    "ClaudeRouterConfig",
    "PostgresConfig",
    "RedisConfig",
    "ComfyUIConfig",
    "WireGuardConfig",
    "AIFactoryConfig",
    "get_ai_factory_config",
    "set_ai_factory_config",
]
