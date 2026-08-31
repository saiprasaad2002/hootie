"""The LLM extension point, shared by generation, grounding, and vision."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class StructuredMode(StrEnum):
    """How a server accepts a JSON schema.

    OpenAI-compatible servers vary widely here, so the client discovers which
    rung of the ladder works and stays there for the rest of the run.
    """

    JSON_SCHEMA = "json_schema"  # strict schema enforcement
    JSON_OBJECT = "json_object"  # "must be valid JSON", shape unenforced
    PROMPT = "prompt"  # nothing but instructions and hope
    UNKNOWN = "unknown"  # not yet negotiated


@runtime_checkable
class ChatClient(Protocol):
    """Sends chat messages to a model and returns raw text.

    `schema` is a JSON Schema describing the desired response. Implementations
    should use it if the server supports it and fall back gracefully if not;
    callers validate the result regardless.
    """

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> str: ...
