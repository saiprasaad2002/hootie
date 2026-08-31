"""Semantic chunking via Chonkie, with page provenance restored afterwards.

Chonkie returns character offsets into the text it was given, which is exactly
the assembled document, so a chunk's page span can be recovered by bisecting the
page offset index. That is what lets every generated QA pair name the pages it
came from.

Two corrections are applied to Chonkie's output:

* Figure descriptions are fenced blocks the chunker knows nothing about, so a
  boundary can land inside one. Such chunks are merged, because half a figure
  description is worse than a slightly oversized chunk.
* Very short chunks are folded into their neighbour; a 20-character fragment
  cannot support a useful question.
"""

from __future__ import annotations

import re
from bisect import bisect_right

from ..config import ChunkConfig
from ..errors import ChunkError
from ..types import AssembledDoc, SourceChunk

# Matches the labelled blocks written by the vision stage. Deliberately specific:
# extracted markdown often contains ordinary ``` code fences, which must not be
# treated as figure boundaries.
FIGURE_BLOCK = re.compile(r"^```figure\b.*?^```", re.MULTILINE | re.DOTALL)


class SemanticChunkerAdapter:
    """Default `Chunker`, wrapping `chonkie.SemanticChunker`.

    The underlying chunker is built lazily because constructing it loads an
    embedding model, which is slow and pointless for commands that never chunk.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()
        self._chunker = None

    def _build(self):
        if self._chunker is not None:
            return self._chunker
        try:
            from chonkie import SemanticChunker
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise ChunkError(
                "chonkie is required for semantic chunking; install hootie with its "
                "default dependencies"
            ) from exc

        cfg = self.config
        try:
            self._chunker = SemanticChunker(
                embedding_model=cfg.embedding_model,
                threshold=cfg.threshold,
                chunk_size=cfg.chunk_size,
                similarity_window=cfg.similarity_window,
                min_sentences_per_chunk=cfg.min_sentences_per_chunk,
                min_characters_per_sentence=cfg.min_characters_per_sentence,
                skip_window=cfg.skip_window,
            )
        except Exception as exc:
            raise ChunkError(f"could not build the semantic chunker: {exc}") from exc
        return self._chunker

    def chunk(self, doc: AssembledDoc) -> list[SourceChunk]:
        text = doc.markdown
        if not text.strip():
            return []

        try:
            raw = self._build().chunk(text)
        except Exception as exc:
            raise ChunkError(f"chunking failed: {exc}") from exc

        bounds = [(c.start_index, c.end_index, getattr(c, "token_count", 0)) for c in raw]
        bounds = _repair_figure_blocks(bounds, text)
        bounds = _merge_short(bounds, self.config.min_chunk_chars)

        starts = [s.start for s in doc.spans]
        chunks: list[SourceChunk] = []
        for i, (start, end, tokens) in enumerate(bounds):
            body = text[start:end]
            if not body.strip():
                continue
            chunks.append(
                SourceChunk(
                    id=f"c{i:05d}",
                    text=body,
                    start_index=start,
                    end_index=end,
                    first_page=_page_at(doc, starts, start),
                    last_page=_page_at(doc, starts, max(start, end - 1)),
                    token_count=tokens,
                    contains_figure=bool(FIGURE_BLOCK.search(body)),
                )
            )
        return chunks


def _page_at(doc: AssembledDoc, starts: list[int], offset: int) -> int:
    """Translate a character offset into a 1-indexed page number."""
    if not doc.spans:
        return 1
    index = bisect_right(starts, offset) - 1
    index = min(max(index, 0), len(doc.spans) - 1)
    return doc.spans[index].page


def _repair_figure_blocks(
    bounds: list[tuple[int, int, int]], text: str
) -> list[tuple[int, int, int]]:
    """Merge chunks whose boundary falls inside a ```figure block.

    Chonkie splits on semantic similarity and has no idea these blocks exist, so
    a description can be cut in half. Merging is the conservative repair: an
    oversized chunk still generates fine, a truncated figure does not.
    """
    blocks = [(m.start(), m.end()) for m in FIGURE_BLOCK.finditer(text)]
    if not blocks or not bounds:
        return bounds

    merged: list[tuple[int, int, int]] = []
    for start, end, tokens in bounds:
        if merged and any(b_start < merged[-1][1] < b_end for b_start, b_end in blocks):
            # The previous chunk ended mid-block; absorb this one into it.
            p_start, _, p_tokens = merged[-1]
            merged[-1] = (p_start, end, p_tokens + tokens)
            continue
        merged.append((start, end, tokens))
    return merged


def _merge_short(bounds: list[tuple[int, int, int]], minimum: int) -> list[tuple[int, int, int]]:
    """Fold chunks below the minimum length into the previous one."""
    if minimum <= 0 or not bounds:
        return bounds

    merged: list[tuple[int, int, int]] = []
    for start, end, tokens in bounds:
        if merged and (end - start) < minimum:
            p_start, _, p_tokens = merged[-1]
            merged[-1] = (p_start, end, p_tokens + tokens)
        else:
            merged.append((start, end, tokens))

    # A short leading chunk has no predecessor; fold it forward instead.
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < minimum:
        s0, _, t0 = merged[0]
        _, e1, t1 = merged[1]
        merged[0:2] = [(s0, e1, t0 + t1)]
    return merged
