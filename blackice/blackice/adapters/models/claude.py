"""Claude model provider for BLACKICE 3.0.

Implements the ModelProvider protocol for Anthropic's Claude models.
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


class ClaudeProvider(BaseModelProvider):
    """Claude model provider using Anthropic's API.

    Supports Claude 3 models (Opus, Sonnet, Haiku) with full
    tool use and streaming capabilities.
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url or "https://api.anthropic.com",
            timeout=timeout,
        )
        self.model = model or self.DEFAULT_MODEL
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "claude"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_embeddings=False,
            max_tokens=8192,
            context_window=200_000,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "x-api-key": self.api_key or "",
                    "anthropic-version": self.API_VERSION,
                    "content-type": "application/json",
                },
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
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Format messages for the Claude API.

        Returns:
            Tuple of (system_prompt, formatted_messages)
        """
        system_prompt: str | None = None
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                formatted.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Add tool results if provided
        if tool_results:
            for result in tool_results:
                formatted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": result.output,
                            "is_error": result.is_error,
                        }
                    ],
                })

        return system_prompt, formatted

    def _format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for the Claude API."""
        if not tools:
            return None

        formatted = []
        for tool in tools:
            formatted.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", tool.get("input_schema", {})),
            })
        return formatted

    def _parse_response(
        self,
        response: dict[str, Any],
        latency_ms: float,
    ) -> GenerationResult:
        """Parse the Claude API response."""
        content = ""
        tool_calls: list[ToolCall] = []

        for block in response.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block["input"],
                    )
                )

        usage = response.get("usage", {})
        return GenerationResult(
            content=content,
            finish_reason=response.get("stop_reason", "stop"),
            tool_calls=tool_calls,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
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
        messages = [Message(role="user", content=prompt)]
        if system_prompt:
            messages.insert(0, Message(role="system", content=system_prompt))

        return await self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

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
        system_prompt, formatted_messages = self._format_messages(messages, tool_results)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stop_sequences:
            payload["stop_sequences"] = stop_sequences

        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            payload["tools"] = formatted_tools

        start = time.monotonic()
        try:
            response = await client.post("/v1/messages", json=payload)
            latency_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                error_data = response.json()
                raise ProviderError(
                    f"Claude API error: {error_data.get('error', {}).get('message', 'Unknown error')}",
                    context={
                        "status_code": response.status_code,
                        "provider": self.name,
                        "model": self.model,
                    },
                )

            return self._parse_response(response.json(), latency_ms)

        except httpx.RequestError as e:
            raise ProviderError(
                f"Claude API request failed: {e}",
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
        system_prompt, formatted_messages = self._format_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with client.stream("POST", "/v1/messages", json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise ProviderError(
                        f"Claude streaming error: {error_text.decode()}",
                        context={"provider": self.name, "status_code": response.status_code},
                    )

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        data = json.loads(line[6:])
                        if data["type"] == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")

        except httpx.RequestError as e:
            raise ProviderError(
                f"Claude streaming request failed: {e}",
                context={"provider": self.name},
            ) from e

    async def embed(self, text: str) -> EmbeddingResult:
        """Claude does not support embeddings."""
        raise NotImplementedError("Claude does not support embeddings. Use a dedicated embedding model.")

    async def health(self) -> HealthStatus:
        """Check Claude API health."""
        start = time.monotonic()
        try:
            # Use a minimal request to check health
            await self.generate("Hi", max_tokens=5)
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=True,
                latency_ms=latency,
                details={"model": self.model, "provider": self.name},
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                error=str(e),
                details={"model": self.model, "provider": self.name},
            )
