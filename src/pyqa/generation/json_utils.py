"""Recovering JSON from models that were asked nicely rather than constrained."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON out of a model response.

    Tries, in order: the whole string, any fenced code block, and finally the
    first balanced `{...}` or `[...]`. Models routinely wrap JSON in prose or
    fences even when told not to, and re-prompting for that is a wasted call.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for match in _FENCE.finditer(text):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

    span = _balanced_span(text)
    if span is not None:
        try:
            return json.loads(span)
        except json.JSONDecodeError as exc:
            raise ValueError(f"no parseable JSON in response: {exc}") from exc

    raise ValueError("no JSON object or array found in response")


def _balanced_span(text: str) -> str | None:
    """Return the first balanced JSON object/array, ignoring braces in strings."""
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    if not starts:
        return None

    begin = starts[0]
    opener = text[begin]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escaped = False
    for i in range(begin, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[begin : i + 1]
    return None
