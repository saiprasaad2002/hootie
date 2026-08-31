"""The async orchestrator.

Concurrency is per-stage: each remote stage has its own semaphore so a slow
vision endpoint cannot starve generation. Every stage has a failure budget, so a
handful of bad pages degrade a long run instead of aborting it, but a systemic
failure still stops early rather than burning the user's budget.

Results of the expensive stages are cached against the document hash and the
settings that affect them, so an interrupted run resumes without paying twice for
vision calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .chunking import SemanticChunkerAdapter
from .config import Config
from .errors import HootieError, StageBudgetExceeded
from .generation import OpenAICompatClient
from .generation.grounding import GroundingChecker
from .generation.qagen import QAGenerator
from .parsing import PdfInspectorParser, assemble_document
from .parsing.assemble import sha256_file
from .parsing.figures import detect_figures
from .progress import ProgressCallback, ProgressReporter
from .types import (
    AssembledDoc,
    PagePlan,
    PageTask,
    ParsedDocument,
    QAPair,
    SourceChunk,
    Stage,
)
from .vision import plan_pages, summarize_plan
from .vision.rasterize import PageRenderer
from .vision.splice import anchor_text, normalize_blocks, splice
from .vision.vlm import VlmVisionEngine
from .writers import write_dataset, write_manifest, write_rejected

logger = logging.getLogger("hootie")


@dataclass
class RunResult:
    document: ParsedDocument
    doc: AssembledDoc
    chunks: list[SourceChunk]
    kept: list[QAPair]
    rejected: list[QAPair]
    manifest: dict[str, Any] = field(default_factory=dict)
    dataset_path: Path | None = None


class _Budget:
    """Tracks failures for one stage and trips when the allowance runs out."""

    def __init__(self, stage: Stage, allowed: int) -> None:
        self.stage = stage
        self.allowed = allowed
        self.failures = 0

    def record(self, exc: BaseException) -> None:
        self.failures += 1
        logger.warning("[%s] %s", self.stage, exc)
        if self.failures > self.allowed:
            raise StageBudgetExceeded(str(self.stage), self.failures, self.allowed)


async def run_async(
    pdf: Path,
    config: Config | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Run the full pipeline over one PDF."""
    config = config or Config()
    pdf = Path(pdf)
    reporter = ProgressReporter(progress)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    # ---- 1. parse -------------------------------------------------------
    reporter.start(Stage.PARSE, 1, f"reading {pdf.name}")
    t0 = time.perf_counter()
    digest = await asyncio.to_thread(sha256_file, pdf)
    document = await asyncio.to_thread(PdfInspectorParser().parse, pdf)
    timings["parse_s"] = time.perf_counter() - t0
    reporter.finish(Stage.PARSE, f"{document.page_count} pages, {document.pdf_type}")

    cache = _Cache(config, digest) if config.use_cache else None

    # ---- 2. plan and run vision ----------------------------------------
    t0 = time.perf_counter()
    figures = {}
    if config.vision.describe_figures:
        figures = await asyncio.to_thread(detect_figures, pdf, config.vision.figures)
    plans = plan_pages(document, figures, config.vision)
    overrides = await _run_vision(pdf, document, plans, config, reporter, cache)
    timings["vision_s"] = time.perf_counter() - t0

    # ---- 3. assemble ----------------------------------------------------
    doc = assemble_document(
        document, overrides=overrides, source_path=str(pdf), source_sha256=digest
    )

    # ---- 4. chunk -------------------------------------------------------
    reporter.start(Stage.CHUNK, 1, "loading embedding model")
    t0 = time.perf_counter()
    chunker = SemanticChunkerAdapter(config.chunk)
    chunks = await asyncio.to_thread(chunker.chunk, doc)
    timings["chunk_s"] = time.perf_counter() - t0
    reporter.finish(Stage.CHUNK, f"{len(chunks)} chunks")

    if not chunks:
        raise HootieError(
            "no chunks were produced. The document may have no extractable text; "
            "run `hootie inspect` to see whether it needs OCR."
        )

    # ---- 5/6. generate and ground ---------------------------------------
    if config.generate.endpoint is None:
        raise HootieError(
            "no generation endpoint is configured. Add a [generate.endpoint] "
            "section to your config file."
        )

    gen_client = OpenAICompatClient(config.generate.endpoint)
    ground_client = (
        gen_client
        if not config.ground.enabled or config.ground.endpoint is config.generate.endpoint
        else OpenAICompatClient(config.ground.endpoint)
    )

    try:
        t0 = time.perf_counter()
        pairs = await _generate(chunks, gen_client, config, reporter)
        timings["generate_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        kept, rejected = await _ground(pairs, chunks, ground_client, config, reporter)
        timings["ground_s"] = time.perf_counter() - t0
        structured_mode = gen_client.mode.value
    finally:
        await gen_client.close()
        if ground_client is not gen_client:
            await ground_client.close()

    # ---- 7. write -------------------------------------------------------
    reporter.start(Stage.WRITE, 3)
    out = config.output.dir
    dataset_path = out / config.output.dataset_name
    write_dataset(
        kept,
        dataset_path,
        system_prompt=config.generate.system_prompt,
        include_provenance=config.output.include_provenance,
    )
    reporter.advance(Stage.WRITE, detail=str(dataset_path))
    write_rejected(rejected, out / config.output.rejected_name)
    reporter.advance(Stage.WRITE)

    counts = summarize_plan(plans)
    manifest = {
        "hootie_version": __version__,
        "source": {
            "path": str(pdf),
            "sha256": digest,
            "page_count": document.page_count,
            "pdf_type": document.pdf_type,
            "classification_confidence": round(document.confidence, 3),
            "complex_layout": document.is_complex,
        },
        "vision_calls": {
            task.value: counts.get(task, 0)
            for task in (PageTask.OCR, PageTask.TABLE_REREAD, PageTask.DESCRIBE)
        },
        "vision_calls_total": sum(
            counts.get(t, 0) for t in (PageTask.OCR, PageTask.TABLE_REREAD, PageTask.DESCRIBE)
        ),
        "figure_pages": sorted(figures),
        "chunks": len(chunks),
        "pairs": {
            "generated": len(pairs),
            "kept": len(kept),
            "rejected": len(rejected),
            "yield": round(len(kept) / len(pairs), 3) if pairs else 0.0,
        },
        "models": {
            "generate": config.generate.endpoint.model,
            "ground": config.ground.endpoint.model if config.ground.enabled else None,
            "vision": config.vision.endpoint.model if config.vision.endpoint else None,
            "embedding": config.chunk.embedding_model,
        },
        "structured_output_mode": structured_mode,
        "timings": {
            **{k: round(v, 2) for k, v in timings.items()},
            "total_s": round(time.perf_counter() - started, 2),
        },
    }
    write_manifest(manifest, out / config.output.manifest_name)
    reporter.finish(Stage.WRITE, str(out))

    return RunResult(document, doc, chunks, kept, rejected, manifest, dataset_path)


async def _run_vision(
    pdf: Path,
    document: ParsedDocument,
    plans: Sequence[PagePlan],
    config: Config,
    reporter: ProgressReporter,
    cache: _Cache | None,
) -> dict[int, str]:
    """Render and read every page the planner marked, returning page overrides."""
    billable = [p for p in plans if p.needs_vlm]
    reporter.start(Stage.VISION, len(billable))
    if not billable:
        reporter.finish(Stage.VISION, "no vision calls needed")
        return {}

    if config.vision.endpoint is None:
        raise HootieError(
            f"{len(billable)} page(s) need a vision model but no [vision.endpoint] "
            "is configured. Add one, or disable the optional work with "
            "--no-figures / --no-table-reread."
        )

    cached = cache.load("vision") if cache else {}
    overrides: dict[int, str] = {int(k): v for k, v in cached.items()}
    todo = [p for p in billable if p.page not in overrides]
    if overrides:
        reporter.advance(Stage.VISION, len(overrides), f"{len(overrides)} pages from cache")

    if not todo:
        reporter.finish(Stage.VISION, "all pages cached")
        return overrides

    page_text = {p.page: p.markdown for p in document.pages}
    client = OpenAICompatClient(config.vision.endpoint)
    engine = VlmVisionEngine(client)
    renderer = PageRenderer(pdf, config.vision.dpi, config.vision.max_image_edge)
    budget = _Budget(Stage.VISION, config.vision.failure_budget)
    semaphore = asyncio.Semaphore(config.vision.concurrency)

    async def handle(plan: PagePlan) -> tuple[int, str] | None:
        async with semaphore:
            # Render inside the worker so only `concurrency` pages are ever
            # resident; pre-rendering a long scan would exhaust memory.
            image = await asyncio.to_thread(renderer.render, plan.page)
            text = await engine.read_page(image, plan, page_text.get(plan.page, ""))
            if not text:
                return None
            if plan.task is PageTask.DESCRIBE:
                anchor = await asyncio.to_thread(anchor_text, pdf, plan.page, plan.figures)
                return plan.page, splice(
                    page_text.get(plan.page, ""), normalize_blocks(text, plan.page), anchor
                )
            # OCR and TABLE_REREAD both transcribe the whole page, so they replace it.
            return plan.page, text

    try:
        tasks = [asyncio.create_task(handle(p)) for p in todo]
        for coro, plan in zip(asyncio.as_completed(tasks), todo, strict=False):
            try:
                result = await coro
            except Exception as exc:
                budget.record(exc)
            else:
                if result:
                    overrides[result[0]] = result[1]
            reporter.advance(Stage.VISION, detail=f"{plan.task.value} pages")
    finally:
        renderer.close()
        await client.close()

    if cache:
        cache.save("vision", {str(k): v for k, v in overrides.items()})
    reporter.finish(Stage.VISION, f"{len(overrides)} pages enriched")
    return overrides


async def _generate(
    chunks: Sequence[SourceChunk],
    client: OpenAICompatClient,
    config: Config,
    reporter: ProgressReporter,
) -> list[QAPair]:
    reporter.start(Stage.GENERATE, len(chunks))
    generator = QAGenerator(client, config.generate)
    budget = _Budget(Stage.GENERATE, config.generate.failure_budget)
    semaphore = asyncio.Semaphore(config.generate.concurrency)
    pairs: list[QAPair] = []

    async def one(chunk: SourceChunk) -> list[QAPair]:
        async with semaphore:
            return await generator.generate(chunk)

    tasks = [asyncio.create_task(one(c)) for c in chunks]
    for coro in asyncio.as_completed(tasks):
        try:
            pairs.extend(await coro)
        except Exception as exc:
            budget.record(exc)
        reporter.advance(Stage.GENERATE, detail=f"{len(pairs)} pairs")

    reporter.finish(Stage.GENERATE, f"{len(pairs)} pairs from {len(chunks)} chunks")
    return pairs


async def _ground(
    pairs: Sequence[QAPair],
    chunks: Sequence[SourceChunk],
    client: OpenAICompatClient,
    config: Config,
    reporter: ProgressReporter,
) -> tuple[list[QAPair], list[QAPair]]:
    if not config.ground.enabled:
        return list(pairs), []

    reporter.start(Stage.GROUND, len(pairs))
    by_id = {c.id: c for c in chunks}
    checker = GroundingChecker(client, config.ground)
    budget = _Budget(Stage.GROUND, config.ground.failure_budget)
    semaphore = asyncio.Semaphore(config.ground.concurrency)

    async def one(pair: QAPair) -> QAPair:
        async with semaphore:
            return await checker.check(pair, by_id[pair.chunk_id])

    tasks = [asyncio.create_task(one(p)) for p in pairs]
    for coro in asyncio.as_completed(tasks):
        try:
            await coro
        except Exception as exc:
            budget.record(exc)
        reporter.advance(Stage.GROUND)

    kept = [p for p in pairs if p.grounded is not False]
    rejected = [p for p in pairs if p.grounded is False]
    reporter.finish(Stage.GROUND, f"{len(kept)} kept, {len(rejected)} rejected")
    return kept, rejected


class _Cache:
    """Caches expensive stage output against the document and relevant settings."""

    def __init__(self, config: Config, digest: str) -> None:
        # Only settings that change the output participate in the key, so
        # unrelated edits (output paths, concurrency) do not invalidate vision.
        material = json.dumps(
            {
                "vision": config.vision.model_dump(mode="json", exclude={"endpoint"}),
                "model": config.vision.endpoint.model if config.vision.endpoint else None,
            },
            sort_keys=True,
        )
        settings = hashlib.sha256(material.encode()).hexdigest()[:8]
        self.dir = Path(config.cache_dir) / f"{digest[:16]}-{settings}"

    def load(self, name: str) -> dict[str, Any]:
        path = self.dir / f"{name}.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug("ignoring unreadable cache file %s", path)
            return {}

    def save(self, name: str, payload: dict[str, Any]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{name}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:  # caching is an optimisation, never a hard failure
            logger.debug("could not write cache %s: %s", name, exc)


def run(
    pdf: Path, config: Config | None = None, *, progress: ProgressCallback | None = None
) -> RunResult:
    """Synchronous facade for callers who do not want an event loop."""
    return asyncio.run(run_async(pdf, config, progress=progress))
