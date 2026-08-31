"""Progress reporting.

The pipeline emits structured events and never prints. Consumers decide how to
render them, which keeps presentation out of the core and lets library users
plug in their own reporting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .types import Stage

logger = logging.getLogger("hootie")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A unit of progress within a stage.

    `total` is None when the size of the work isn't known yet (for example while
    parsing, before the page count is available).
    """

    stage: Stage
    completed: int
    total: int | None = None
    detail: str | None = None

    @property
    def fraction(self) -> float | None:
        if not self.total:
            return None
        return min(1.0, self.completed / self.total)


ProgressCallback = Callable[[ProgressEvent], None]


def log_progress(event: ProgressEvent) -> None:
    """Default renderer: emit progress to the standard logging machinery.

    Document content never appears here; `detail` carries only page numbers and
    task names, so this is safe at INFO.
    """
    if event.total:
        pct = int((event.fraction or 0) * 100)
        msg = f"[{event.stage}] {event.completed}/{event.total} ({pct}%)"
    else:
        msg = f"[{event.stage}] {event.completed}"
    if event.detail:
        msg = f"{msg} - {event.detail}"
    logger.info(msg)


def null_progress(event: ProgressEvent) -> None:
    """Discard progress. Used in tests and by callers who want silence."""


class ProgressReporter:
    """Tracks per-stage counts and forwards events to a callback.

    Stages call `advance` as work completes; the reporter keeps the running
    total so callers don't have to thread counters through the pipeline.
    """

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self._callback = callback or null_progress
        self._completed: dict[Stage, int] = {}
        self._totals: dict[Stage, int | None] = {}

    def start(self, stage: Stage, total: int | None = None, detail: str | None = None) -> None:
        self._completed[stage] = 0
        self._totals[stage] = total
        self._emit(stage, detail)

    def advance(self, stage: Stage, step: int = 1, detail: str | None = None) -> None:
        self._completed[stage] = self._completed.get(stage, 0) + step
        self._emit(stage, detail)

    def finish(self, stage: Stage, detail: str | None = None) -> None:
        total = self._totals.get(stage)
        if total is not None:
            self._completed[stage] = total
        self._emit(stage, detail)

    def _emit(self, stage: Stage, detail: str | None) -> None:
        self._callback(
            ProgressEvent(
                stage=stage,
                completed=self._completed.get(stage, 0),
                total=self._totals.get(stage),
                detail=detail,
            )
        )
