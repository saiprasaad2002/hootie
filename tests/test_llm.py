"""Structured-output negotiation against servers with differing support."""

from __future__ import annotations

import httpx
import pytest
from openai import APIStatusError, RateLimitError

from hootie.config import EndpointConfig
from hootie.generation import OpenAICompatClient, StructuredMode

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def _status_error(status: int, message: str) -> APIStatusError:
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return APIStatusError(message, response=response, body=None)


class _FakeCompletions:
    """Stands in for `client.chat.completions`, recording what it was sent."""

    def __init__(self, behaviour) -> None:
        self.behaviour = behaviour
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.behaviour(kwargs, len(self.calls))
        if isinstance(result, Exception):
            raise result

        class _Msg:
            content = result

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _client(behaviour, **overrides) -> tuple[OpenAICompatClient, _FakeCompletions]:
    cfg = EndpointConfig(model="test-model", base_url="http://localhost:8000/v1", **overrides)
    client = OpenAICompatClient(cfg)
    fake = _FakeCompletions(behaviour)
    client._client.chat.completions = fake  # type: ignore[assignment]
    return client, fake


def _format_type(call: dict) -> str | None:
    fmt = call.get("response_format")
    return fmt.get("type") if fmt else None


async def test_json_schema_used_when_supported():
    client, fake = _client(lambda kw, n: '{"a": 1}')
    assert await client.complete([{"role": "user", "content": "hi"}], schema=SCHEMA) == '{"a": 1}'
    assert _format_type(fake.calls[0]) == "json_schema"
    assert client.mode is StructuredMode.JSON_SCHEMA


async def test_falls_back_to_json_object_then_prompt():
    """A server supporting neither mechanism must still produce output."""

    def behaviour(kw, n):
        fmt = _format_type(kw)
        if fmt == "json_schema":
            return _status_error(400, "response_format json_schema is not supported")
        if fmt == "json_object":
            return _status_error(400, "response_format is unsupported by this model")
        return '{"a": 3}'

    client, fake = _client(behaviour)
    assert await client.complete([{"role": "user", "content": "hi"}], schema=SCHEMA) == '{"a": 3}'
    assert [_format_type(c) for c in fake.calls] == ["json_schema", "json_object", None]
    assert client.mode is StructuredMode.PROMPT


async def test_prompt_mode_injects_a_json_instruction():
    """With nothing enforcing shape, the instruction is all we have."""

    def behaviour(kw, n):
        if _format_type(kw) is not None:
            return _status_error(400, "response_format not supported")
        return "{}"

    client, fake = _client(behaviour)
    await client.complete([{"role": "system", "content": "You are terse."}], schema=SCHEMA)
    system = fake.calls[-1]["messages"][0]["content"]
    assert "You are terse." in system
    assert "valid JSON" in system


async def test_negotiated_mode_is_reused_not_rediscovered():
    def behaviour(kw, n):
        if _format_type(kw) == "json_schema":
            return _status_error(400, "json_schema not supported")
        return '{"a": 1}'

    client, fake = _client(behaviour)
    for _ in range(3):
        await client.complete([{"role": "user", "content": "hi"}], schema=SCHEMA)
    # One probe, then three calls at the working rung: no repeated probing.
    assert [_format_type(c) for c in fake.calls].count("json_schema") == 1


async def test_unrelated_400_is_not_treated_as_missing_support():
    """A genuine bad request must surface, not silently degrade output quality."""
    client, _ = _client(lambda kw, n: _status_error(400, "context length exceeded"))
    with pytest.raises(APIStatusError):
        await client.complete([{"role": "user", "content": "hi"}], schema=SCHEMA)
    assert client.mode is StructuredMode.JSON_SCHEMA


async def test_rate_limit_is_retried_without_downgrading():
    """A transient failure must not permanently weaken structured output."""
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")

    def behaviour(kw, n):
        if n == 1:
            return RateLimitError(
                "slow down", response=httpx.Response(429, request=request), body=None
            )
        return '{"a": 9}'

    client, fake = _client(behaviour, max_retries=2)
    assert await client.complete([{"role": "user", "content": "hi"}], schema=SCHEMA) == '{"a": 9}'
    assert client.mode is StructuredMode.JSON_SCHEMA
    assert all(_format_type(c) == "json_schema" for c in fake.calls)


async def test_server_error_is_retried():
    def behaviour(kw, n):
        return _status_error(503, "upstream busy") if n == 1 else '{"a": 5}'

    client, fake = _client(behaviour, max_retries=2)
    assert await client.complete([{"role": "user", "content": "hi"}], schema=SCHEMA) == '{"a": 5}'
    assert len(fake.calls) == 2


async def test_no_schema_means_no_response_format():
    client, fake = _client(lambda kw, n: "plain text")
    assert await client.complete([{"role": "user", "content": "hi"}]) == "plain text"
    assert "response_format" not in fake.calls[0]
