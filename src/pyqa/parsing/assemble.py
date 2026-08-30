"""Assemble per-page markdown into one document with a page offset index.

The index is what lets a chunk's character offsets be translated back into page
numbers, which is how every QA pair keeps its provenance.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..types import AssembledDoc, PageSpan, ParsedDocument, ParsedPage

PAGE_SEPARATOR = "\n\n"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file for cache keying, without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def assemble_document(
    pages: Sequence[ParsedPage] | ParsedDocument,
    *,
    overrides: Mapping[int, str] | None = None,
    source_path: str = "",
    source_sha256: str = "",
) -> AssembledDoc:
    """Splice pages into one markdown string in page order.

    `overrides` maps a 1-indexed page number to replacement markdown, which is
    how vision results (OCR transcriptions, table re-reads) take the place of the
    parser's text for that page.

    Every page gets a span even when its text is empty, so page numbering stays
    aligned with the document; empty pages simply have `start == end`.
    """
    if isinstance(pages, ParsedDocument):
        pages = pages.pages
    overrides = overrides or {}
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0

    for i, page in enumerate(sorted(pages, key=lambda p: p.page)):
        text = overrides.get(page.page, page.markdown) or ""
        text = text.strip()

        if i > 0:
            parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)

        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(PageSpan(page=page.page, start=start, end=cursor))

    return AssembledDoc(
        markdown="".join(parts),
        spans=spans,
        source_path=source_path,
        source_sha256=source_sha256,
    )
