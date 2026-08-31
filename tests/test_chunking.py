"""Chunk page-provenance mapping and the post-chunk repairs.

These target the pure helpers rather than the real chunker: loading an embedding
model is slow, and the logic worth pinning down is the offset arithmetic.
"""

from __future__ import annotations

import pytest

from hootie.chunking.semantic import (
    FIGURE_BLOCK,
    SemanticChunkerAdapter,
    _merge_short,
    _page_at,
    _repair_figure_blocks,
)
from hootie.config import ChunkConfig
from hootie.parsing import assemble_document
from hootie.types import ParsedPage


@pytest.fixture
def doc():
    return assemble_document(
        [
            ParsedPage(1, "alpha alpha"),
            ParsedPage(2, "bravo bravo"),
            ParsedPage(3, "charlie charlie"),
        ]
    )


def test_offset_maps_to_the_containing_page(doc):
    starts = [s.start for s in doc.spans]
    for span in doc.spans:
        assert _page_at(doc, starts, span.start) == span.page
        assert _page_at(doc, starts, span.end - 1) == span.page


def test_offsets_outside_the_document_clamp(doc):
    starts = [s.start for s in doc.spans]
    assert _page_at(doc, starts, -5) == 1
    assert _page_at(doc, starts, 10_000) == 3


def test_chunk_spanning_pages_reports_a_range(doc):
    starts = [s.start for s in doc.spans]
    first = _page_at(doc, starts, doc.spans[0].start)
    last = _page_at(doc, starts, doc.spans[2].end - 1)
    assert (first, last) == (1, 3)


def test_figure_regex_ignores_ordinary_code_fences():
    """Extracted markdown routinely contains ``` blocks; only ```figure counts."""
    text = "```\nplain code\n```\n\n```figure page=2\nA bar chart.\n```\n"
    matches = [m.group(0) for m in FIGURE_BLOCK.finditer(text)]
    assert len(matches) == 1
    assert "bar chart" in matches[0]


def test_split_figure_block_is_merged_back():
    text = "intro text\n```figure page=1\nA flowchart of the approval process.\n```\ntail"
    block = FIGURE_BLOCK.search(text)
    boundary = block.start() + 20  # cut inside the block
    bounds = [(0, boundary, 5), (boundary, len(text), 5)]
    repaired = _repair_figure_blocks(bounds, text)
    assert len(repaired) == 1, "a chunk boundary inside a figure block must be healed"
    assert repaired[0] == (0, len(text), 10)


def test_boundary_outside_a_figure_block_is_left_alone():
    text = "one\n```figure page=1\nchart\n```\ntwo three four"
    boundary = text.index("two")
    bounds = [(0, boundary, 5), (boundary, len(text), 5)]
    assert _repair_figure_blocks(bounds, text) == bounds


def test_short_chunks_fold_into_the_previous_one():
    assert _merge_short([(0, 100, 20), (100, 105, 1)], 80) == [(0, 105, 21)]


def test_short_leading_chunk_folds_forward():
    """The first chunk has no predecessor, so it must merge with what follows."""
    assert _merge_short([(0, 5, 1), (5, 200, 40)], 80) == [(0, 200, 41)]


def test_merge_short_disabled_when_minimum_is_zero():
    bounds = [(0, 5, 1), (5, 10, 1)]
    assert _merge_short(bounds, 0) == bounds


def test_empty_document_yields_no_chunks():
    empty = assemble_document([ParsedPage(1, "   ")])
    assert SemanticChunkerAdapter(ChunkConfig()).chunk(empty) == []
