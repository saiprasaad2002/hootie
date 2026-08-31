"""Parser adapter behaviour, especially the page index normalization."""

from __future__ import annotations

import pytest

from hootie.errors import ParseError
from hootie.parsing import PdfInspectorParser
from hootie.parsing.pdf_inspector import _assert_index_invariant
from hootie.types import ParsedPage


def test_pages_are_one_indexed_and_contiguous(native_pdf):
    doc = PdfInspectorParser().parse(native_pdf)
    assert doc.page_count == 3
    # pdf-inspector reports PageMarkdown.page 0-indexed; the adapter must shift it.
    assert [p.page for p in doc.pages] == [1, 2, 3]


def test_native_pdf_needs_no_ocr(native_pdf):
    doc = PdfInspectorParser().parse(native_pdf)
    assert doc.ocr_pages == []
    assert all(not p.needs_ocr for p in doc.pages)
    assert doc.pdf_type == "text_based"


def test_page_text_lands_on_the_right_page(native_pdf):
    doc = PdfInspectorParser().parse(native_pdf)
    text = {p.page: p.markdown for p in doc.pages}
    # Off-by-one in normalization would shuffle these.
    assert "Alpha" in text[1]
    assert "Bravo" in text[2]
    assert "Charlie" in text[3]


def test_missing_file_raises_parse_error(tmp_path):
    with pytest.raises(ParseError, match="not found"):
        PdfInspectorParser().parse(tmp_path / "nope.pdf")


def test_index_invariant_catches_convention_change(tmp_path):
    """If the two OCR signals disagree, normalization is wrong — fail loudly."""
    pages = [
        ParsedPage(page=1, markdown="a", needs_ocr=False),
        ParsedPage(page=2, markdown="b", needs_ocr=True),
    ]
    _assert_index_invariant(pages, {2}, tmp_path / "x.pdf")  # agrees: fine
    with pytest.raises(ParseError, match="index mismatch"):
        _assert_index_invariant(pages, {1}, tmp_path / "x.pdf")  # off by one


def test_index_invariant_rejects_non_contiguous_pages(tmp_path):
    pages = [
        ParsedPage(page=1, markdown="a", needs_ocr=False),
        ParsedPage(page=3, markdown="c", needs_ocr=False),
    ]
    with pytest.raises(ParseError, match="non-contiguous"):
        _assert_index_invariant(pages, set(), tmp_path / "x.pdf")
