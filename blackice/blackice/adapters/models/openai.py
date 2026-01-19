"""OpenAI model provider for BLACKICE 3.0.

Implements the ModelProvider protocol for OpenAI's GPT models.
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


class OpenAIProvider(BaseModelProvider):
    """OpenAI model provider using the OpenAI API.

    Supports GPT-4 and GPT-3.5 models with tool use and streaming.
    """

    DEFAULT_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or "https://api.openai.com",
            timeout=timeout,
        )
        self.model = model or self.DEFAULT_MODEL
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_embeddings=True,
            max_tokens=16384,
            context_window=128_000,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key or ''}",
                    "Content-Type": "application/json",
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
    ) -> list[dict[str, Any]]:
        """Format messages for the OpenAI API."""
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
                    "tool_call_id": result.call_id,
                    "content": result.output,
                })

        return formatted

    def _format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for the OpenAI API."""
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
        """Parse the OpenAI API response."""
        choice = response["choices"][0]
        message = choice["message"]

        content = message.get("content", "") or ""
        tool_calls: list[ToolCall] = []

        if message.get("tool_calls"):
            import json
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                )

        usage = response.get("usage", {})
        return GenerationResult(
            content=content,
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=tool_calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
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
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

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
        formatted_messages = self._format_messages(messages, tool_results)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            payload["tools"] = formatted_tools

        start = time.monotonic()
        try:
            response = await client.post("/v1/chat/completions", json=payload)
            latency_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                error_data = response.json()
                raise ProviderError(
                    f"OpenAI API error: {error_data.get('error', {}).get('message', 'Unknown error')}",
                    context={
                        "status_code": response.status_code,
                        "provider": self.name,
                        "model": self.model,
                    },
                )

            return self._parse_response(response.json(), latency_ms)

        except httpx.RequestError as e:
            raise ProviderError(
                f"OpenAI API request failed: {e}",
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
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise ProviderError(
                        f"OpenAI streaming error: {error_text.decode()}",
                        context={"provider": self.name, "status_code": response.status_code},
                    )

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        import json
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]

        except httpx.RequestError as e:
            raise ProviderError(
                f"OpenAI streaming request failed: {e}",
                context={"provider": self.name},
            ) from e

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embeddings for text."""
        client = await self._get_client()

        payload = {
            "model": self.EMBEDDING_MODEL,
            "input": text,
        }

        try:
            response = await client.post("/v1/embeddings", json=payload)

            if response.status_code != 200:
                error_data = response.json()
                raise ProviderError(
                    f"OpenAI embeddings error: {error_data.get('error', {}).get('message', 'Unknown error')}",
                    context={"provider": self.name},
                )

            data = response.json()
            embedding = data["data"][0]["embedding"]
            return EmbeddingResult(
                embedding=embedding,
                dimensions=len(embedding),
                model=self.EMBEDDING_MODEL,
                tokens=data["usage"]["total_tokens"],
            )

        except httpx.RequestError as e:
            raise ProviderError(
                f"OpenAI embeddings request failed: {e}",
                context={"provider": self.name},
            ) from e

    async def health(self) -> HealthStatus:
        """Check OpenAI API health."""
        start = time.monotonic()
        try:
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
