"""Base interface for Model Providers in BLACKICE 3.0.

Model providers abstract LLM interactions, supporting multiple
providers (Claude, OpenAI, Ollama) with a unified interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result of a tool execution."""

    call_id: str
    output: str
    is_error: bool = False


@dataclass
class GenerationResult:
    """Result of a text generation request."""

    content: str
    finish_reason: str  # "stop", "length", "tool_use", "error"
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class EmbeddingResult:
    """Result of an embedding request."""

    embedding: list[float]
    dimensions: int
    model: str = ""
    tokens: int = 0


@dataclass
class HealthStatus:
    """Health status of a provider."""

    healthy: bool
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCapabilities:
    """Capabilities of a model provider."""

    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_embeddings: bool = False
    max_tokens: int = 100_000
    context_window: int = 200_000


@runtime_checkable
class ModelProvider(Protocol):
    """Protocol for model providers.

    Implementations must provide methods for:
    - Text generation (generate, chat)
    - Embeddings (embed) - optional
    - Health checks (health)
    """

    @property
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities."""
        ...

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences that stop generation

        Returns:
            GenerationResult with the generated text
        """
        ...

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_results: list[ToolResult] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate a response in a conversation.

        Args:
            messages: Conversation history
            tools: Available tools the model can call
            tool_results: Results from previous tool calls
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences that stop generation

        Returns:
            GenerationResult with the response
        """
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a response token by token.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Response tokens as they're generated
        """
        ...

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with the embedding vector
        """
        ...

    async def health(self) -> HealthStatus:
        """Check provider health.

        Returns:
            HealthStatus indicating if provider is operational
        """
        ...


class BaseModelProvider(ABC):
    """Abstract base class for model providers.

    Provides common functionality and default implementations
    for the ModelProvider protocol.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities. Override for specific providers."""
        return ModelCapabilities()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_results: list[ToolResult] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate a response in a conversation."""
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Default streaming implementation via regular generation."""
        result = await self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        yield result.content

    async def embed(self, text: str) -> EmbeddingResult:
        """Default embed raises NotImplementedError."""
        raise NotImplementedError(f"{self.name} does not support embeddings")

    async def health(self) -> HealthStatus:
        """Default health check via lightweight generation."""
        import time

        start = time.monotonic()
        try:
            await self.generate("Hello", max_tokens=5)
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(healthy=True, latency_ms=latency)
        except Exception as e:
            return HealthStatus(healthy=False, error=str(e))
