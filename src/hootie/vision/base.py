"""The vision engine extension point."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import PagePlan


@runtime_checkable
class VisionEngine(Protocol):
    """Reads a rendered page image according to the task assigned to it.

    `context` carries whatever text the parser already recovered for the page.
    Implementations should use it to avoid re-transcribing prose that was already
    extracted perfectly well, and return markdown.
    """

    async def read_page(self, image: bytes, plan: PagePlan, context: str) -> str: ...
