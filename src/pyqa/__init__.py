"""pyqa — turn PDFs into finetuning-ready QA datasets using your own endpoints."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config
from .types import AssembledDoc, PagePlan, PageTask, ParsedDocument, QAPair, SourceChunk, Stage

__all__ = [
    "AssembledDoc",
    "Config",
    "PagePlan",
    "PageTask",
    "ParsedDocument",
    "QAPair",
    "SourceChunk",
    "Stage",
    "__version__",
    "run",
    "run_async",
]


def __getattr__(name: str):
    # Imported lazily: pulling in the pipeline drags along heavy optional deps.
    if name in ("run", "run_async"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
