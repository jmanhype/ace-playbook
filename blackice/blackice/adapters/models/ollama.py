"""Ollama model provider for BLACKICE 3.0.

Implements the ModelProvider protocol for local Ollama inference.
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator

import httpx

from blackice.adapters.models.base import (
    BaseModelProvider,
    EmbeddingResult,
    GenerationResult,
    HealthStatus,
    Message,
    ModelCapabilities,
    ToolCall,
    ToolResult,
)
from blackice.primitives.errors import ProviderError


class OllamaProvider(BaseModelProvider):
    """Ollama model provider for local inference.

    Supports any model available through Ollama with streaming
    and tool use capabilities (for supported models).
    """

    DEFAULT_MODEL = "qwen2.5-coder:32b"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,  # Longer timeout for local inference
    ) -> None:
        super().__init__(
            api_key=None,  # Ollama doesn't require API key
            base_url=base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            timeout=timeout,
        )
        self.model = model or self.DEFAULT_MODEL
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_streaming=True,
            supports_tools=True,  # Some models support tools
            supports_vision=False,
            supports_embeddings=True,
            max_tokens=32768,
            context_window=128_000,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _format_messages(
        self,
        messages: list[Message],
        tool_results: list[ToolResult] | None = None,
    ) -> list[dict[str, Any]]:
        """Format messages for the Ollama API."""
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            formatted.append({
                "role": msg.role,
                "content": msg.content,
            })

        # Add tool results if provided
        if tool_results:
            for result in tool_results:
                formatted.append({
                    "role": "tool",
                    "content": result.output,
                })

        return formatted

    def _format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for the Ollama API."""
        if not tools:
            return None

        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", tool.get("input_schema", {})),
                },
            })
        return formatted

    def _parse_response(
        self,
        response: dict[str, Any],
        latency_ms: float,
    ) -> GenerationResult:
        """Parse the Ollama API response."""
        message = response.get("message", {})
        content = message.get("content", "")

        tool_calls: list[ToolCall] = []
        if message.get("tool_calls"):
            import json
            for i, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        name=func.get("name", ""),
                        arguments=args,
                    )
                )

        # Ollama doesn't provide detailed token counts
        return GenerationResult(
            content=content,
            finish_reason="stop" if response.get("done") else "length",
            tool_calls=tool_calls,
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            model=response.get("model", self.model),
            latency_ms=latency_ms,
        )

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
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        start = time.monotonic()
        try:
            response = await client.post("/api/generate", json=payload)
            latency_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                raise ProviderError(
                    f"Ollama API error: {response.text}",
                    context={
                        "status_code": response.status_code,
                        "provider": self.name,
                        "model": self.model,
                    },
                )

            data = response.json()
            return GenerationResult(
                content=data.get("response", ""),
                finish_reason="stop" if data.get("done") else "length",
                tool_calls=[],
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                model=data.get("model", self.model),
                latency_ms=latency_ms,
            )

        except httpx.RequestError as e:
            raise ProviderError(
                f"Ollama API request failed: {e}",
                context={"provider": self.name, "model": self.model},
            ) from e

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
        client = await self._get_client()
        formatted_messages = self._format_messages(messages, tool_results)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            payload["tools"] = formatted_tools

        start = time.monotonic()
        try:
            response = await client.post("/api/chat", json=payload)
            latency_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                raise ProviderError(
                    f"Ollama API error: {response.text}",
                    context={
                        "status_code": response.status_code,
                        "provider": self.name,
                        "model": self.model,
                    },
                )

            return self._parse_response(response.json(), latency_ms)

        except httpx.RequestError as e:
            raise ProviderError(
                f"Ollama API request failed: {e}",
                context={"provider": self.name, "model": self.model},
            ) from e

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a response token by token."""
        client = await self._get_client()
        formatted_messages = self._format_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise ProviderError(
                        f"Ollama streaming error: {error_text.decode()}",
                        context={"provider": self.name, "status_code": response.status_code},
                    )

                import json
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        message = data.get("message", {})
                        if "content" in message:
                            yield message["content"]
                        if data.get("done"):
                            break

        except httpx.RequestError as e:
            raise ProviderError(
                f"Ollama streaming request failed: {e}",
                context={"provider": self.name},
            ) from e

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embeddings for text."""
        client = await self._get_client()

        payload = {
            "model": self.model,
            "prompt": text,
        }

        try:
            response = await client.post("/api/embeddings", json=payload)

            if response.status_code != 200:
                raise ProviderError(
                    f"Ollama embeddings error: {response.text}",
                    context={"provider": self.name},
                )

            data = response.json()
            embedding = data.get("embedding", [])
            return EmbeddingResult(
                embedding=embedding,
                dimensions=len(embedding),
                model=self.model,
                tokens=0,  # Ollama doesn't report token count for embeddings
            )

        except httpx.RequestError as e:
            raise ProviderError(
                f"Ollama embeddings request failed: {e}",
                context={"provider": self.name},
            ) from e

    async def health(self) -> HealthStatus:
        """Check Ollama API health."""
        start = time.monotonic()
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            latency = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                return HealthStatus(
                    healthy=False,
                    error=f"Ollama returned status {response.status_code}",
                    details={"provider": self.name},
                )

            models = response.json().get("models", [])
            return HealthStatus(
                healthy=True,
                latency_ms=latency,
                details={
                    "provider": self.name,
                    "model": self.model,
                    "available_models": [m["name"] for m in models],
                },
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                error=str(e),
                details={"provider": self.name},
            )
