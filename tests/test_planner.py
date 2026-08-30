"""Vision planner: one task per page, correct precedence, budget behaviour."""

from __future__ import annotations

from pyqa.config import VisionConfig
from pyqa.types import FigureRegion, PagePlan, PageTask, ParsedDocument, ParsedPage
from pyqa.vision import plan_pages, summarize_plan


def _doc(*pages: ParsedPage, is_complex: bool = False) -> ParsedDocument:
    return ParsedDocument(pages=tuple(pages), is_complex=is_complex)


def _fig(page: int) -> FigureRegion:
    return FigureRegion(page, 0, 0, 100, 100, "raster", 0.4)


def _tasks(plans: list[PagePlan]) -> list[PageTask]:
    return [p.task for p in plans]


def test_ocr_page_never_also_gets_a_figure_or_table_call():
    """OCR subsumes the rest: a page must never cost two VLM calls."""
    doc = _doc(
        ParsedPage(1, "", needs_ocr=True, has_tables=True),
        is_complex=True,
    )
    plans = plan_pages(doc, {1: [_fig(1)]}, VisionConfig())
    assert _tasks(plans) == [PageTask.OCR]
    assert sum(p.needs_vlm for p in plans) == 1


def test_table_reread_requires_both_a_table_and_a_complex_layout():
    page = ParsedPage(1, "text", needs_ocr=False, has_tables=True)
    simple = plan_pages(_doc(page, is_complex=False), {}, VisionConfig())
    complex_ = plan_pages(_doc(page, is_complex=True), {}, VisionConfig())
    assert _tasks(simple) == [PageTask.NONE]
    assert _tasks(complex_) == [PageTask.TABLE_REREAD]


def test_figures_produce_describe_tasks():
    doc = _doc(ParsedPage(1, "text", needs_ocr=False))
    assert _tasks(plan_pages(doc, {1: [_fig(1)]}, VisionConfig())) == [PageTask.DESCRIBE]


def test_toggles_disable_optional_work():
    doc = _doc(ParsedPage(1, "t", has_tables=True), is_complex=True)
    off = VisionConfig(describe_figures=False, table_reread=False)
    assert _tasks(plan_pages(doc, {1: [_fig(1)]}, off)) == [PageTask.NONE]


def test_clean_page_costs_nothing():
    doc = _doc(ParsedPage(1, "plain text", needs_ocr=False))
    plans = plan_pages(doc, {}, VisionConfig())
    assert _tasks(plans) == [PageTask.NONE]
    assert not any(p.needs_vlm for p in plans)


def test_budget_trims_optional_work_but_never_ocr():
    doc = _doc(
        ParsedPage(1, "", needs_ocr=True),
        ParsedPage(2, "t"),
        ParsedPage(3, "t"),
    )
    plans = plan_pages(doc, {2: [_fig(2)], 3: [_fig(3)]}, VisionConfig(max_vision_pages=1))
    assert plans[0].task is PageTask.OCR, "OCR must survive the budget"
    assert sum(p.needs_vlm for p in plans) == 1
    assert any("budget exhausted" in " ".join(p.reasons) for p in plans)


def test_budget_of_zero_still_keeps_ocr():
    """Dropping OCR would lose the page's text entirely, so the cap yields to it."""
    doc = _doc(ParsedPage(1, "", needs_ocr=True), ParsedPage(2, "t"))
    plans = plan_pages(doc, {2: [_fig(2)]}, VisionConfig(max_vision_pages=0))
    assert plans[0].task is PageTask.OCR
    assert plans[1].task is PageTask.NONE


def test_summary_counts_tasks():
    doc = _doc(ParsedPage(1, "", needs_ocr=True), ParsedPage(2, "t"))
    counts = summarize_plan(plan_pages(doc, {}, VisionConfig()))
    assert counts[PageTask.OCR] == 1
    assert counts[PageTask.NONE] == 1
