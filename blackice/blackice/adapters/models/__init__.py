"""Model provider adapters for BLACKICE 3.0."""

from blackice.adapters.models.base import (
    BaseModelProvider,
    EmbeddingResult,
    GenerationResult,
    HealthStatus,
    Message,
    ModelCapabilities,
    ModelProvider,
    ToolCall,
    ToolResult,
)
from blackice.adapters.models.claude import ClaudeProvider
from blackice.adapters.models.ollama import OllamaProvider
from blackice.adapters.models.openai import OpenAIProvider

__all__ = [
    # Protocol and base
    "ModelProvider",
    "BaseModelProvider",
    # Data classes
    "Message",
    "ToolCall",
    "ToolResult",
    "GenerationResult",
    "EmbeddingResult",
    "HealthStatus",
    "ModelCapabilities",
    # Providers
    "ClaudeProvider",
    "OpenAIProvider",
    "OllamaProvider",
]
