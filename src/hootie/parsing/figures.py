"""Detect pages carrying visual content worth sending to a vision model.

Two signals, because **many PDF diagrams are vector drawings, not embedded
images**. A flowchart drawn with vector paths has no image object at all, so
raster-only detection would miss exactly the diagrams that matter most in policy
and technical documents.

Boilerplate suppression is not optional: without it, a letterhead logo on every
page would trigger a paid VLM call on every page.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from ..config import FigureConfig
from ..types import FigureRegion

# Hashing the whole image stream is wasteful for large scans; a prefix plus the
# total length distinguishes logos from one another perfectly well.
_FINGERPRINT_PREFIX_BYTES = 8192


def detect_figures(path: Path, config: FigureConfig) -> dict[int, list[FigureRegion]]:
    """Find candidate figures per page, keyed by 1-indexed page number.

    Runs in two passes: collect candidates, then drop raster images that repeat
    across enough pages to be page furniture rather than content.
    """
    pdf = pdfium.PdfDocument(str(path))
    try:
        page_count = len(pdf)
        candidates: dict[int, list[FigureRegion]] = {}
        fingerprint_pages: dict[str, set[int]] = defaultdict(set)

        for index in range(page_count):
            page = pdf[index]
            page_no = index + 1
            width, height = page.get_size()
            page_area = width * height
            if page_area <= 0:
                continue

            found: list[FigureRegion] = []
            found.extend(_raster_figures(page, page_no, width, height, page_area, config))
            vector = _vector_figure(page, page_no, width, height, page_area, config)
            if vector is not None:
                found.append(vector)

            for region in found:
                if region.fingerprint:
                    fingerprint_pages[region.fingerprint].add(page_no)
            if found:
                candidates[page_no] = found

        return _suppress_boilerplate(candidates, fingerprint_pages, page_count, config)
    finally:
        pdf.close()


def _raster_figures(
    page: pdfium.PdfPage,
    page_no: int,
    width: float,
    height: float,
    page_area: float,
    config: FigureConfig,
) -> list[FigureRegion]:
    """Embedded raster images above a share of the page area."""
    regions: list[FigureRegion] = []
    for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE]):
        try:
            left, bottom, right, top = obj.get_bounds()
        except Exception:
            continue
        area = abs(right - left) * abs(top - bottom)
        ratio = area / page_area
        if ratio < config.min_raster_area_ratio:
            continue
        regions.append(
            FigureRegion(
                page=page_no,
                x0=left,
                y0=bottom,
                x1=right,
                y1=top,
                kind="raster",
                area_ratio=ratio,
                fingerprint=_fingerprint(obj),
            )
        )
    return regions


def _vector_figure(
    page: pdfium.PdfPage,
    page_no: int,
    width: float,
    height: float,
    page_area: float,
    config: FigureConfig,
) -> FigureRegion | None:
    """A dense cluster of vector paths, reported as one region.

    Paths in the top and bottom margin bands are ignored: those are nearly always
    header and footer rules, not content. Individual paths are also the wrong
    granularity to report, since a single diagram is made of many of them, so the
    union of their bounds is returned as one figure.
    """
    band = height * config.margin_band_ratio
    lower_limit, upper_limit = band, height - band

    left = bottom = float("inf")
    right = top = float("-inf")
    count = 0

    for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_PATH]):
        try:
            px0, py0, px1, py1 = obj.get_bounds()
        except Exception:
            continue
        mid_y = (py0 + py1) / 2
        if mid_y < lower_limit or mid_y > upper_limit:
            continue  # header/footer furniture
        count += 1
        left, bottom = min(left, px0), min(bottom, py0)
        right, top = max(right, px1), max(top, py1)

    if count < config.min_vector_paths:
        return None

    area = abs(right - left) * abs(top - bottom)
    return FigureRegion(
        page=page_no,
        x0=left,
        y0=bottom,
        x1=right,
        y1=top,
        kind="vector",
        area_ratio=area / page_area if page_area else 0.0,
    )


def _fingerprint(obj: pdfium.PdfImage) -> str | None:
    """Identify a raster image cheaply, for repeat detection across pages."""
    try:
        data = obj.get_data(decode_simple=False)
    except Exception:
        return None
    if not data:
        return None
    view = bytes(data[:_FINGERPRINT_PREFIX_BYTES])
    return hashlib.sha256(view + str(len(data)).encode()).hexdigest()[:32]


def _suppress_boilerplate(
    candidates: dict[int, list[FigureRegion]],
    fingerprint_pages: dict[str, set[int]],
    page_count: int,
    config: FigureConfig,
) -> dict[int, list[FigureRegion]]:
    """Drop raster images that appear on too many pages to be content."""
    if page_count <= 2:
        # Repetition means nothing in a document this short.
        return candidates

    threshold = max(2, int(page_count * config.boilerplate_page_ratio))
    boilerplate = {fp for fp, pages in fingerprint_pages.items() if len(pages) >= threshold}
    if not boilerplate:
        return candidates

    kept: dict[int, list[FigureRegion]] = {}
    for page_no, regions in candidates.items():
        surviving = [r for r in regions if r.fingerprint not in boilerplate]
        if surviving:
            kept[page_no] = surviving
    return kept
