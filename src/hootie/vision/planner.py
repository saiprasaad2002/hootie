"""Decide what the vision stage must do for each page.

At most one task per page, so a page is rendered once and costs at most one VLM
call. `OCR` deliberately subsumes the other tasks: a transcription prompt for a
scanned page already covers its figures and tables, so paying twice for the same
page would be waste.

Routing uses the **per-page** `needs_ocr` verdict, not the document-level
classification. The two genuinely disagree in practice: a text document with a
logo on every page classifies as `image_based` while every page extracts fine.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from ..config import VisionConfig
from ..types import FigureRegion, PagePlan, PageTask, ParsedDocument


def plan_pages(
    document: ParsedDocument,
    figures: Mapping[int, Sequence[FigureRegion]] | None = None,
    config: VisionConfig | None = None,
) -> list[PagePlan]:
    """Assign one task per page, in precedence order.

    Honours `max_vision_pages` by trimming the *optional* work first — figure
    descriptions and table re-reads — because dropping OCR would lose text
    outright, whereas dropping an enrichment only loses an extra.
    """
    config = config or VisionConfig()
    figures = figures or {}

    plans: list[PagePlan] = []
    for page in document.pages:
        regions = tuple(figures.get(page.page, ()))

        if page.needs_ocr:
            reason = page.ocr_reason or "page has no extractable text layer"
            plans.append(PagePlan(page.page, PageTask.OCR, (reason,), regions))
            continue

        if config.table_reread and page.has_tables and document.is_complex:
            plans.append(
                PagePlan(
                    page.page,
                    PageTask.TABLE_REREAD,
                    ("table on a complex layout; heuristic extraction is unreliable",),
                    regions,
                )
            )
            continue

        if config.describe_figures and regions:
            kinds = ", ".join(sorted({r.kind for r in regions}))
            plans.append(
                PagePlan(
                    page.page,
                    PageTask.DESCRIBE,
                    (f"{len(regions)} {kinds} figure region(s)",),
                    regions,
                )
            )
            continue

        plans.append(PagePlan(page.page, PageTask.NONE, (), regions))

    return _apply_budget(plans, config.max_vision_pages)


def _apply_budget(plans: list[PagePlan], budget: int | None) -> list[PagePlan]:
    """Trim optional vision work to fit a cap, keeping OCR intact."""
    if budget is None:
        return plans

    billable = [p for p in plans if p.needs_vlm]
    if len(billable) <= budget:
        return plans

    # OCR pages are non-negotiable; they are the only source of text on that page.
    ocr = [p for p in billable if p.task is PageTask.OCR]
    optional = [p for p in billable if p.task is not PageTask.OCR]
    allowance = max(0, budget - len(ocr))
    keep = {id(p) for p in optional[:allowance]} | {id(p) for p in ocr}

    return [
        p
        if (not p.needs_vlm or id(p) in keep)
        else PagePlan(p.page, PageTask.NONE, ("skipped: vision budget exhausted",), p.figures)
        for p in plans
    ]


def summarize_plan(plans: Sequence[PagePlan]) -> Counter[PageTask]:
    """Count pages per task, for cost projection and the run manifest."""
    return Counter(p.task for p in plans)
