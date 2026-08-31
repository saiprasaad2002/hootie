"""Figure detection: vector diagrams and boilerplate suppression."""

from __future__ import annotations

import pypdfium2 as pdfium

from hootie.config import FigureConfig
from hootie.parsing.figures import _raster_figures, detect_figures


def test_detects_vector_flowchart_not_just_raster(figures_pdf):
    """A vector-drawn diagram has no image object; raster-only detection misses it."""
    found = detect_figures(figures_pdf, FigureConfig())
    assert 2 in found, "page 2's vector flowchart should be detected"
    assert {r.kind for r in found[2]} == {"vector"}


def test_detects_large_raster_chart(figures_pdf):
    found = detect_figures(figures_pdf, FigureConfig())
    assert 3 in found
    assert any(r.kind == "raster" for r in found[3])


def test_text_only_pdf_yields_no_figures(native_pdf):
    assert detect_figures(native_pdf, FigureConfig()) == {}


def test_repeated_logo_is_suppressed_as_boilerplate(figures_pdf):
    """Without suppression a letterhead would bill a VLM call on every page."""
    cfg = FigureConfig(min_raster_area_ratio=0.0005)  # low enough to admit the logo

    pdf = pdfium.PdfDocument(str(figures_pdf))
    try:
        page = pdf[0]
        w, h = page.get_size()
        raw_candidates = _raster_figures(page, 1, w, h, w * h, cfg)
    finally:
        pdf.close()

    # The logo genuinely qualifies on its own...
    assert raw_candidates, "logo should be a candidate before suppression"
    # ...but repeats often enough to be page furniture, so it is dropped.
    final = detect_figures(figures_pdf, cfg)
    assert 1 not in final and 4 not in final
    assert sorted(final) == [2, 3]


def test_margin_band_excludes_header_footer_rules(figures_pdf):
    """Paths in the margin bands are furniture, so a huge band suppresses the chart."""
    permissive = detect_figures(figures_pdf, FigureConfig(margin_band_ratio=0.0))
    aggressive = detect_figures(figures_pdf, FigureConfig(margin_band_ratio=0.49))
    assert 2 in permissive
    assert 2 not in aggressive, "a near-full-page margin band should exclude the flowchart"


def test_min_vector_paths_threshold_is_respected(figures_pdf):
    assert 2 in detect_figures(figures_pdf, FigureConfig(min_vector_paths=5))
    assert 2 not in detect_figures(figures_pdf, FigureConfig(min_vector_paths=500))
