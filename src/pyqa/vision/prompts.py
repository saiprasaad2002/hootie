"""Prompts for the vision stage, one per task.

Each is written to do the least work that recovers what the parser could not.
The DESCRIBE prompt in particular is told not to transcribe body prose: that text
was already extracted natively and re-transcribing it would both cost tokens and
risk contradicting the reliable copy.
"""

from __future__ import annotations

from ..types import PageTask

OCR_SYSTEM = """\
You transcribe scanned document pages into clean Markdown.

Rules:
- Transcribe ALL visible text faithfully, preserving reading order.
- Render tables as Markdown tables, preserving every row and column.
- For each chart, diagram, or figure, add a description of what it shows,
  including any data values or labels you can read, inside a fenced block:
  ```figure
  ...description...
  ```
- Do not summarise, correct, or editorialise. Transcribe what is there.
- If part of the page is illegible, write [illegible] rather than guessing.
- Output only the Markdown. No commentary, no fences around the whole page."""

TABLE_SYSTEM = """\
You transcribe a document page whose tables were extracted unreliably.

The page's result REPLACES the extracted text, so transcribe the whole page, not
only its tables. Tables are the reason you were called, so give them most care.

Rules:
- Transcribe all visible text on the page, preserving reading order.
- Render every table as a Markdown table, preserving every row, column, header,
  and empty cell exactly. Repeat merged header values across the columns they span.
- Keep numbers, units, and footnote markers exactly as printed.
- For each chart or diagram, add a description inside a fenced block:
  ```figure
  ...description...
  ```
- Do not summarise or editorialise. Output only the Markdown."""

DESCRIBE_SYSTEM = """\
You describe charts, diagrams, and figures on a document page.

The page's text was already extracted correctly and is given to you for context.
Your job is ONLY the visual elements it could not capture.

Rules:
- For each figure, emit one fenced block:
  ```figure
  ...description...
  ```
- Describe what the figure communicates: chart type, axes, series, trends,
  and every label or data value you can read. For flowcharts and diagrams,
  describe the nodes and the direction of the connections between them.
- Be specific and factual. A reader who cannot see the image should be able to
  answer questions about it from your description alone.
- Do NOT transcribe the surrounding body text; it is already captured.
- Ignore logos, headers, footers, page furniture, and decorative rules.
- If there is no substantive figure, output nothing at all."""

SYSTEM_BY_TASK = {
    PageTask.OCR: OCR_SYSTEM,
    PageTask.TABLE_REREAD: TABLE_SYSTEM,
    PageTask.DESCRIBE: DESCRIBE_SYSTEM,
}


def user_prompt(task: PageTask, page: int, context: str) -> str:
    """Build the user turn accompanying the page image."""
    if task is PageTask.OCR:
        return f"Transcribe page {page} of this document."

    context = (context or "").strip()
    if task is PageTask.TABLE_REREAD:
        head = f"Re-read the table(s) on page {page}."
    else:
        head = f"Describe the figures on page {page}."

    if not context:
        return head
    return (
        f"{head}\n\nFor context, here is the text already extracted from this page:\n\n"
        f"---\n{context[:4000]}\n---"
    )
