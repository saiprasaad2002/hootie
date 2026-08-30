"""The chunker extension point."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import AssembledDoc, SourceChunk


@runtime_checkable
class Chunker(Protocol):
    """Splits an assembled document into chunks that carry page provenance."""

    def chunk(self, doc: AssembledDoc) -> list[SourceChunk]: ...
