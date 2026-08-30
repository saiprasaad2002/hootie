"""Pin down pdf-inspector's page index conventions.

The library uses **opposite conventions in two functions**, which silently
returns the wrong page rather than raising. These tests fail loudly if a future
version changes either one.

    extract_pages_markdown        pages arg: 0-indexed, PageMarkdown.page: 0-indexed
    extract_text_with_positions   pages arg: 1-INDEXED, TextItem.page:     1-indexed
"""

from __future__ import annotations

import pdf_inspector


def test_extract_pages_markdown_is_zero_indexed(figures_pdf):
    result = pdf_inspector.extract_pages_markdown(str(figures_pdf), pages=[2])
    (page,) = result.pages
    assert page.page == 2, "PageMarkdown.page is 0-indexed"
    assert "Page Three" in page.markdown, "pages=[2] must select the THIRD page"


def test_extract_text_with_positions_is_one_indexed(figures_pdf):
    items = pdf_inspector.extract_text_with_positions(str(figures_pdf), pages=[1])
    assert items, "pages=[1] must select the FIRST page, not the second"
    assert {i.page for i in items} == {1}
    assert any("Page One" in i.text for i in items)


def test_position_page_zero_is_empty(figures_pdf):
    """Proof the positions API is 1-indexed: page 0 does not exist for it."""
    assert pdf_inspector.extract_text_with_positions(str(figures_pdf), pages=[0]) == []


def test_full_extraction_reports_one_indexed_pages(figures_pdf):
    items = pdf_inspector.extract_text_with_positions(str(figures_pdf))
    assert sorted({i.page for i in items}) == [1, 2, 3, 4]


def test_coordinates_use_a_bottom_left_origin(figures_pdf):
    """Text drawn at PDF y=700 on a 792pt page must report y=700, not y=92."""
    items = pdf_inspector.extract_text_with_positions(str(figures_pdf), pages=[1])
    text_items = [i for i in items if i.item_type == "text"]
    assert any(abs(i.y - 700.0) < 1.0 for i in text_items)
