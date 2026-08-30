"""An OpenAI-compatible chat client.

`base_url` makes this work against vLLM, TGI, Together, Groq, OpenRouter, Ollama
and OpenAI itself. Those servers disagree about structured output, so the client
negotiates once per run and remembers the answer:

    json_schema  ->  json_object  ->  prompt-only

Downgrades happen only on errors that actually indicate the feature is missing.
A rate limit or a network blip is retried at the current rung instead, so a
transient failure never permanently degrades output quality.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..config import EndpointConfig
from ..errors import GenerationError
from .base import StructuredMode

logger = logging.getLogger("pyqa.llm")

# Substrings that mark "this server does not support that response_format",
# as opposed to a request that was malformed for some other reason.
_UNSUPPORTED_HINTS = (
    "response_format",
    "json_schema",
    "guided_json",
    "structured output",
    "not supported",
    "unsupported",
    "unrecognized",
    "extra fields",
)

_JSON_INSTRUCTION = (
    "Respond with a single valid JSON value and nothing else. "
    "Do not wrap it in markdown fences or add commentary."
)


class OpenAICompatClient:
    """Default `ChatClient`. Safe to share across concurrent tasks."""

    def __init__(self, config: EndpointConfig) -> None:
        self.config = config
        self.mode = StructuredMode.UNKNOWN
        self._lock = asyncio.Lock()
        self._client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value() or "not-needed",
            base_url=config.base_url,
            timeout=config.timeout,
            # Retries are handled here so backoff is shared with our own policy.
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self.config.model

    async def close(self) -> None:
        await self._client.close()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> str:
        """Send messages and return the assistant's text."""
        if schema is None:
            return await self._send(messages, response_format=None)
        return await self._send_structured(messages, schema, schema_name)

    async def _send_structured(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        """Try progressively weaker structured-output mechanisms."""
        async with self._lock:
            if self.mode is StructuredMode.UNKNOWN:
                self.mode = StructuredMode.JSON_SCHEMA

        while True:
            mode = self.mode
            try:
                return await self._send(
                    self._prepare(messages, mode),
                    response_format=self._response_format(mode, schema, schema_name),
                )
            except APIStatusError as exc:
                downgraded = self._downgrade(mode, exc)
                if downgraded is None:
                    raise
                async with self._lock:
                    if self.mode is mode:  # another task may have downgraded already
                        logger.info(
                            "endpoint rejected %s structured output; falling back to %s",
                            mode.value,
                            downgraded.value,
                        )
                        self.mode = downgraded

    def _downgrade(self, mode: StructuredMode, exc: APIStatusError) -> StructuredMode | None:
        """Decide whether an error means "unsupported" and what to try next."""
        if exc.status_code not in (400, 404, 422, 500, 501):
            return None
        if not _looks_unsupported(exc):
            return None
        if mode is StructuredMode.JSON_SCHEMA:
            return StructuredMode.JSON_OBJECT
        if mode is StructuredMode.JSON_OBJECT:
            return StructuredMode.PROMPT
        return None

    @staticmethod
    def _response_format(
        mode: StructuredMode, schema: dict[str, Any], schema_name: str
    ) -> dict[str, Any] | None:
        if mode is StructuredMode.JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        if mode is StructuredMode.JSON_OBJECT:
            return {"type": "json_object"}
        return None

    @staticmethod
    def _prepare(messages: list[dict[str, Any]], mode: StructuredMode) -> list[dict[str, Any]]:
        """Add an explicit JSON instruction when the server won't enforce one."""
        if mode is StructuredMode.JSON_SCHEMA:
            return messages
        patched = [dict(m) for m in messages]
        for message in patched:
            if message.get("role") == "system" and isinstance(message.get("content"), str):
                message["content"] = f"{message['content']}\n\n{_JSON_INSTRUCTION}"
                return patched
        return [{"role": "system", "content": _JSON_INSTRUCTION}, *patched]

    async def _send(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None,
    ) -> str:
        """One request, with retry on transient failures only."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        attempts = self.config.max_retries + 1
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            retry=retry_if_exception_type((RateLimitError, APIConnectionError, _TransientStatus)),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await self._client.chat.completions.create(**kwargs)
                except APIStatusError as exc:
                    # 5xx is worth retrying; 4xx is the caller's problem.
                    if exc.status_code >= 500 and not _looks_unsupported(exc):
                        raise _TransientStatus(str(exc)) from exc
                    raise

                if not response.choices:
                    raise GenerationError("model returned no choices")
                return response.choices[0].message.content or ""

        raise GenerationError("retry loop exited without a result")  # pragma: no cover


class _TransientStatus(Exception):
    """Server-side failure worth retrying."""


def _looks_unsupported(exc: APIStatusError) -> bool:
    text = str(getattr(exc, "message", "") or exc).lower()
    return any(hint in text for hint in _UNSUPPORTED_HINTS)
