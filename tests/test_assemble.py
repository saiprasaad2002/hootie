"""Page offset index: the basis of every QA pair's provenance."""

from __future__ import annotations

from pyqa.parsing import assemble_document
from pyqa.types import ParsedPage


def _pages(*texts: str) -> list[ParsedPage]:
    return [ParsedPage(page=i, markdown=t, needs_ocr=False) for i, t in enumerate(texts, 1)]


def test_spans_round_trip_to_original_text():
    doc = assemble_document(_pages("alpha", "bravo", "charlie"))
    for span, expected in zip(doc.spans, ["alpha", "bravo", "charlie"], strict=True):
        assert doc.markdown[span.start : span.end] == expected


def test_overrides_replace_page_text():
    doc = assemble_document(_pages("alpha", "bravo"), overrides={2: "REPLACED"})
    assert doc.markdown[doc.spans[1].start : doc.spans[1].end] == "REPLACED"
    assert "bravo" not in doc.markdown


def test_empty_pages_keep_their_span_so_numbering_stays_aligned():
    doc = assemble_document(_pages("alpha", "", "charlie"))
    assert [s.page for s in doc.spans] == [1, 2, 3]
    assert doc.spans[1].start == doc.spans[1].end  # empty, but present
    assert doc.markdown[doc.spans[2].start : doc.spans[2].end] == "charlie"


def test_pages_are_sorted_regardless_of_input_order():
    pages = _pages("alpha", "bravo", "charlie")
    doc = assemble_document([pages[2], pages[0], pages[1]])
    assert [s.page for s in doc.spans] == [1, 2, 3]
    assert doc.markdown.index("alpha") < doc.markdown.index("bravo") < doc.markdown.index("charlie")
