"""Adapter over Firecrawl's `pdf-inspector` (Rust core).

One `extract_pages_markdown` call yields per-page markdown together with a
per-page `needs_ocr` verdict, which is what lets the vision stage touch only the
pages that genuinely need a model.

Index bases: `pdf-inspector` mixes conventions. `PageMarkdown.page` is
**0-indexed**, while `pages_needing_ocr`, `pages_with_tables`, `pages_with_columns`
and `PageOcrReasons.page` are **1-indexed** (verified empirically against 1.17.0).
Everything is normalized to 1-indexed here and the invariant is asserted, so the
mismatch cannot leak into the rest of the package.
"""

from __future__ import annotations

from pathlib import Path

import pdf_inspector

from ..errors import ParseError
from ..types import ParsedDocument, ParsedPage


class PdfInspectorParser:
    """Default `Parser` implementation."""

    name = "pdf-inspector"

    def parse(self, path: Path) -> ParsedDocument:
        p = Path(path)
        if not p.is_file():
            raise ParseError(f"PDF not found: {p}")

        try:
            result = pdf_inspector.extract_pages_markdown(str(p))
        except Exception as exc:  # the Rust binding raises plain exceptions
            raise ParseError(f"failed to parse {p}: {exc}") from exc

        if not result.pages:
            raise ParseError(f"{p} produced no pages")

        # These collections are already 1-indexed.
        ocr_pages = set(result.pages_needing_ocr or ())
        table_pages = set(result.pages_with_tables or ())
        column_pages = set(result.pages_with_columns or ())
        reasons = {r.page: ", ".join(r.reasons) for r in (result.ocr_reasons_by_page or ())}

        pages: list[ParsedPage] = []
        for pm in result.pages:
            page_1 = pm.page + 1  # PageMarkdown.page is 0-indexed
            pages.append(
                ParsedPage(
                    page=page_1,
                    markdown=pm.markdown or "",
                    needs_ocr=bool(pm.needs_ocr),
                    ocr_reason=pm.ocr_reason or reasons.get(page_1),
                    has_tables=page_1 in table_pages,
                    has_columns=page_1 in column_pages,
                )
            )

        _assert_index_invariant(pages, ocr_pages, p)

        # Document-level classification comes from a second, very cheap call;
        # `extract_pages_markdown` does not report pdf_type or confidence.
        pdf_type, confidence = "unknown", 0.0
        try:
            classification = pdf_inspector.classify_pdf(str(p))
            pdf_type = classification.pdf_type
            confidence = float(classification.confidence)
        except Exception:
            # Classification is advisory only; never fail a run over it.
            pass

        return ParsedDocument(
            pages=tuple(pages),
            pdf_type=pdf_type,
            confidence=confidence,
            is_complex=bool(result.is_complex),
        )


def _assert_index_invariant(pages: list[ParsedPage], ocr_pages: set[int], path: Path) -> None:
    """Guard against pdf-inspector changing its index conventions under us.

    If the two OCR signals disagree, our +1 normalization is wrong and every
    downstream page number would be off by one. Fail loudly rather than silently
    OCR the wrong pages.
    """
    expected = {p.page for p in pages if p.needs_ocr}
    if ocr_pages and expected != ocr_pages:
        raise ParseError(
            f"pdf-inspector page index mismatch on {path}: per-page flags say "
            f"{sorted(expected)} need OCR but the document summary says "
            f"{sorted(ocr_pages)}. The library's index convention may have changed; "
            "the adapter's normalization needs review."
        )

    numbers = [p.page for p in pages]
    if numbers != list(range(1, len(pages) + 1)):
        raise ParseError(
            f"pdf-inspector returned non-contiguous page numbers for {path}: {numbers[:10]}"
        )
