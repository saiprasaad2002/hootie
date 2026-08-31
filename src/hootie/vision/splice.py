"""Insert figure descriptions into a page's markdown at the figure's position.

Position matters. If descriptions were appended at the end of the page, the
chunker could easily place a figure in a different chunk from the prose that
introduces it, and a question grounded on the figure would lose its context.

Anchoring uses `pdf-inspector`'s positioned text: the description goes after the
line of text sitting immediately above the figure. Both `pdf-inspector` and
`pypdfium2` report a bottom-left origin, so a larger `y` means higher on the page.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pdf_inspector

from ..types import FigureRegion

logger = logging.getLogger("hootie.vision")

_FIGURE_FENCE = re.compile(r"^```figure\b[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
# Anchors must be substantial enough to locate unambiguously in the markdown.
_MIN_ANCHOR_CHARS = 12


def normalize_blocks(description: str, page: int) -> str:
    """Ensure the model's output is one or more `page`-labelled figure blocks.

    Models comply with the fencing instruction most of the time but not always,
    so unfenced output is wrapped rather than discarded.
    """
    text = (description or "").strip()
    if not text:
        return ""

    bodies = [m.group(1).strip() for m in _FIGURE_FENCE.finditer(text)]
    if not bodies:
        bodies = [text]

    return "\n\n".join(f"```figure page={page}\n{body}\n```" for body in bodies if body)


def anchor_text(path: Path, page: int, figures: Sequence[FigureRegion]) -> str | None:
    """Find the line of text immediately above the topmost figure on a page.

    Returns None when there is nothing usable to anchor to, in which case the
    caller appends instead.
    """
    if not figures:
        return None

    try:
        # NOTE: `extract_text_with_positions` takes and returns 1-INDEXED pages,
        # the opposite of `extract_pages_markdown`, which uses 0-indexed for both.
        # Verified empirically against pdf-inspector 1.17.0; see test_page_indexing.
        items = pdf_inspector.extract_text_with_positions(str(path), pages=[page])
    except Exception as exc:
        logger.debug("could not read text positions for page %s: %s", page, exc)
        return None

    figure_top = max(f.top for f in figures)
    above = [
        item
        for item in items
        if item.page == page
        and getattr(item, "item_type", "text") == "text"
        and item.y > figure_top
        and len((item.text or "").strip()) >= _MIN_ANCHOR_CHARS
    ]
    if not above:
        return None

    # Smallest y among those above the figure is the closest line to it.
    return min(above, key=lambda item: item.y).text.strip()


def splice(page_markdown: str, blocks: str, anchor: str | None) -> str:
    """Insert `blocks` after the line containing `anchor`, else append."""
    blocks = (blocks or "").strip()
    if not blocks:
        return page_markdown

    body = page_markdown or ""
    if anchor:
        position = _line_end_after(body, anchor)
        if position is not None:
            return f"{body[:position]}\n\n{blocks}{body[position:]}"
        logger.debug("anchor %r not found in page markdown; appending instead", anchor[:40])

    separator = "\n\n" if body.strip() else ""
    return f"{body.rstrip()}{separator}{blocks}"


def _line_end_after(text: str, anchor: str) -> int | None:
    """Offset of the end of the line containing `anchor`, if it can be found.

    Extraction reflows whitespace, so an exact match often fails; a prefix of the
    anchor is tried before giving up.
    """
    for candidate in (anchor, " ".join(anchor.split())[:40]):
        if len(candidate) < _MIN_ANCHOR_CHARS:
            continue
        index = text.find(candidate)
        if index == -1:
            continue
        line_end = text.find("\n", index)
        return len(text) if line_end == -1 else line_end
    return None
