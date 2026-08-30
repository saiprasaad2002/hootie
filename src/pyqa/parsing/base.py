"""The parser extension point."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..types import ParsedDocument


@runtime_checkable
class Parser(Protocol):
    """Turns a PDF into per-page markdown with an OCR verdict per page.

    Implementations must return pages in document order with **1-indexed** page
    numbers, whatever convention the underlying library uses.
    """

    def parse(self, path: Path) -> ParsedDocument: ...
