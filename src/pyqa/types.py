"""Core data types shared across pipeline stages.

Page numbering convention: **every page number in this package is 1-indexed**,
matching what a human sees in a PDF reader. Adapters that talk to libraries using
other conventions must normalize at their boundary (see `parsing.pdf_inspector`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Stage(StrEnum):
    """Pipeline stages, in execution order. Used to label progress events."""

    PARSE = "parse"
    VISION = "vision"
    CHUNK = "chunk"
    GENERATE = "generate"
    GROUND = "ground"
    WRITE = "write"


class PageTask(StrEnum):
    """What, if anything, the vision stage must do for a page.

    At most one task per page: a page is rendered once and costs at most one VLM
    call. `OCR` subsumes the others, since a transcription prompt already covers
    the page's figures and tables.
    """

    NONE = "none"
    OCR = "ocr"
    TABLE_REREAD = "table_reread"
    DESCRIBE = "describe"


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """One page as returned by the parser, before any vision work."""

    page: int  # 1-indexed
    markdown: str
    needs_ocr: bool = False
    ocr_reason: str | None = None
    has_tables: bool = False
    has_columns: bool = False


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Everything the parser learned about a document.

    Carries document-level signals alongside the pages because some vision
    decisions are document-scoped: a table only warrants a re-read when the
    document's layout is complex enough that heuristic extraction is suspect.
    """

    pages: tuple[ParsedPage, ...]
    pdf_type: str = "unknown"
    confidence: float = 0.0
    is_complex: bool = False

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_pages(self) -> list[int]:
        return [p.page for p in self.pages if p.needs_ocr]


@dataclass(frozen=True, slots=True)
class FigureRegion:
    """A candidate figure detected on a page, in PDF points, origin bottom-left."""

    page: int  # 1-indexed
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str  # "raster" | "vector"
    area_ratio: float
    fingerprint: str | None = None  # raster only; used for boilerplate suppression

    @property
    def top(self) -> float:
        """Upper edge, used to order figures against text in reading order."""
        return max(self.y0, self.y1)


@dataclass(frozen=True, slots=True)
class PagePlan:
    """The vision task assigned to a page, with the reasons behind it."""

    page: int  # 1-indexed
    task: PageTask
    reasons: tuple[str, ...] = ()
    figures: tuple[FigureRegion, ...] = ()

    @property
    def needs_vlm(self) -> bool:
        return self.task is not PageTask.NONE


@dataclass(frozen=True, slots=True)
class PageSpan:
    """Where a page's text lives within the assembled document markdown."""

    page: int  # 1-indexed
    start: int  # inclusive character offset
    end: int  # exclusive character offset


@dataclass(slots=True)
class AssembledDoc:
    """The whole document as one markdown string, plus a page offset index."""

    markdown: str
    spans: list[PageSpan] = field(default_factory=list)
    source_path: str = ""
    source_sha256: str = ""

    @property
    def page_count(self) -> int:
        return len(self.spans)


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """A semantic chunk with provenance back to the pages it came from."""

    id: str
    text: str
    start_index: int
    end_index: int
    first_page: int  # 1-indexed
    last_page: int  # 1-indexed
    token_count: int = 0
    contains_figure: bool = False

    @property
    def page_label(self) -> str:
        if self.first_page == self.last_page:
            return f"p{self.first_page}"
        return f"p{self.first_page}-{self.last_page}"


@dataclass(slots=True)
class QAPair:
    """A generated pair plus everything needed to audit where it came from."""

    question: str
    answer: str
    chunk_id: str
    first_page: int
    last_page: int
    contains_figure: bool = False
    grounded: bool | None = None  # None until the grounding stage runs
    rejection_reason: str | None = None
