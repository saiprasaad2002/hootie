"""Command-line interface.

Commands are ordered by how much they cost to run. `inspect` and `chunk` need no
credentials at all, so thresholds can be tuned against real documents before any
paid call is made.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .chunking import SemanticChunkerAdapter
from .config import Config
from .errors import PyqaError
from .parsing import PdfInspectorParser, assemble_document
from .parsing.assemble import sha256_file
from .parsing.figures import detect_figures
from .progress import ProgressEvent
from .types import PageTask, Stage
from .vision import plan_pages, summarize_plan

app = typer.Typer(
    name="pyqa",
    help="Turn PDFs into finetuning-ready QA datasets using your own OCR and LLM endpoints.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to a pyqa TOML config file."),
]

TASK_STYLE = {
    PageTask.OCR: "bold red",
    PageTask.TABLE_REREAD: "yellow",
    PageTask.DESCRIBE: "cyan",
    PageTask.NONE: "dim",
}


def _load_config(path: Path | None) -> Config:
    return Config.load(path) if path else Config()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    # These libraries log every HTTP request at INFO, which buries our progress.
    # Note `httpx2`/`httpcore2`: the openai 3.x SDK uses those, not `httpx`.
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "openai",
        "huggingface_hub",
        "urllib3",
        "filelock",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def inspect(
    pdf: Annotated[Path, typer.Argument(help="PDF to analyse.")],
    config_path: ConfigOption = None,
    no_figures: Annotated[
        bool, typer.Option("--no-figures", help="Skip figure detection.")
    ] = False,
    no_table_reread: Annotated[
        bool, typer.Option("--no-table-reread", help="Never re-read tables with the VLM.")
    ] = False,
    max_vision_pages: Annotated[
        int | None, typer.Option("--max-vision-pages", help="Cap billable vision pages.")
    ] = None,
) -> None:
    """Analyse a PDF and project what the vision stage would cost.

    Makes no network calls and needs no credentials. Use it to tune figure
    detection thresholds against your own documents before spending anything.
    """
    cfg = _load_config(config_path)
    vision = cfg.vision.model_copy(
        update={
            "describe_figures": cfg.vision.describe_figures and not no_figures,
            "table_reread": cfg.vision.table_reread and not no_table_reread,
            "max_vision_pages": max_vision_pages
            if max_vision_pages is not None
            else cfg.vision.max_vision_pages,
        }
    )

    document = PdfInspectorParser().parse(pdf)
    figures = detect_figures(pdf, vision.figures) if vision.describe_figures else {}
    plans = plan_pages(document, figures, vision)
    counts = summarize_plan(plans)

    console.print()
    console.print(f"[bold]{pdf}[/bold]")
    console.print(
        f"  {document.page_count} pages · classified [bold]{document.pdf_type}[/bold] "
        f"(confidence {document.confidence:.2f}) · "
        f"layout {'complex' if document.is_complex else 'simple'}"
    )
    console.print(f"  sha256 {sha256_file(pdf)[:16]}…")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Page", justify="right")
    table.add_column("Vision task")
    table.add_column("Figures", justify="right")
    table.add_column("Tables", justify="center")
    table.add_column("Cols", justify="center")
    table.add_column("Chars", justify="right")
    table.add_column("Why")

    by_page = {p.page: p for p in document.pages}
    for plan in plans:
        page = by_page[plan.page]
        style = TASK_STYLE[plan.task]
        table.add_row(
            str(plan.page),
            f"[{style}]{plan.task.value}[/{style}]",
            str(len(plan.figures)) if plan.figures else "-",
            "✓" if page.has_tables else "-",
            "✓" if page.has_columns else "-",
            str(len(page.markdown)),
            "; ".join(plan.reasons) if plan.reasons else "",
        )
    console.print(table)

    billable = sum(v for k, v in counts.items() if k is not PageTask.NONE)
    console.print()
    console.print("[bold]Projected vision calls[/bold]")
    for task in (PageTask.OCR, PageTask.TABLE_REREAD, PageTask.DESCRIBE):
        if counts.get(task):
            console.print(f"  {task.value:14s} {counts[task]:>4}")
    console.print(f"  [bold]{'total':14s}[/bold] {billable:>4} of {document.page_count} pages")
    if not billable:
        console.print("  [green]No vision calls needed — this document parses natively.[/green]")
    console.print()


@app.command()
def chunk(
    pdf: Annotated[Path, typer.Argument(help="PDF to chunk.")],
    config_path: ConfigOption = None,
    show_text: Annotated[
        bool, typer.Option("--show-text", help="Print each chunk's full text.")
    ] = False,
) -> None:
    """Parse and semantically chunk a PDF, showing page provenance.

    Needs no credentials, but does load a local embedding model on first use.
    Use it to sanity-check chunk sizes and boundaries before generating.
    """
    cfg = _load_config(config_path)
    document = PdfInspectorParser().parse(pdf)
    doc = assemble_document(document, source_path=str(pdf), source_sha256=sha256_file(pdf))

    with console.status("Loading embedding model and chunking…"):
        chunks = SemanticChunkerAdapter(cfg.chunk).chunk(doc)

    if not chunks:
        console.print("[yellow]No chunks produced — the document has no extractable text.[/yellow]")
        console.print("Run [bold]pyqa inspect[/bold] to see whether it needs OCR.")
        return

    console.print()
    console.print(f"[bold]{pdf}[/bold] · {document.page_count} pages · {len(chunks)} chunks")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Chunk")
    table.add_column("Pages")
    table.add_column("Chars", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Fig", justify="center")
    table.add_column("Preview" if not show_text else "Text")

    for c in chunks:
        preview = (
            c.text
            if show_text
            else c.text[:60].replace("\n", " ") + ("…" if len(c.text) > 60 else "")
        )
        table.add_row(
            c.id,
            c.page_label,
            str(len(c.text)),
            str(c.token_count),
            "✓" if c.contains_figure else "-",
            preview,
        )
    console.print(table)

    sizes = [len(c.text) for c in chunks]
    console.print()
    console.print(
        f"  chars: min {min(sizes)} · median {sorted(sizes)[len(sizes) // 2]} · max {max(sizes)}"
    )
    console.print()


STAGE_LABELS = {
    Stage.PARSE: "Parsing",
    Stage.VISION: "Vision",
    Stage.CHUNK: "Chunking",
    Stage.GENERATE: "Generating",
    Stage.GROUND: "Grounding",
    Stage.WRITE: "Writing",
}


class _RichProgress:
    """Renders pipeline events as one live bar per stage."""

    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.tasks: dict[Stage, int] = {}

    def __call__(self, event: ProgressEvent) -> None:
        label = STAGE_LABELS.get(event.stage, str(event.stage))
        if event.stage not in self.tasks:
            self.tasks[event.stage] = self.progress.add_task(label, total=event.total or None)
        task = self.tasks[event.stage]
        self.progress.update(
            task,
            completed=event.completed,
            total=event.total or None,
            description=f"{label} [dim]{event.detail}[/dim]" if event.detail else label,
        )


@app.command()
def check_connection(config_path: ConfigOption = None) -> None:
    """Verify the configured endpoints answer before starting a long run."""
    from .generation import OpenAICompatClient

    cfg = _load_config(config_path)
    targets = [
        ("generate", cfg.generate.endpoint),
        ("vision", cfg.vision.endpoint),
        ("ground", cfg.ground.endpoint if cfg.ground.enabled else None),
    ]

    async def probe(name: str, endpoint) -> tuple[str, bool, str]:
        client = OpenAICompatClient(endpoint)
        try:
            reply = await client.complete([{"role": "user", "content": "Reply with: ok"}])
            return name, True, f"{endpoint.model} -> {reply.strip()[:40]!r}"
        except Exception as exc:
            return name, False, str(exc)[:110]
        finally:
            await client.close()

    async def main_probe():
        return await asyncio.gather(*(probe(n, e) for n, e in targets if e is not None))

    configured = [n for n, e in targets if e is not None]
    if not configured:
        console.print("[yellow]No endpoints configured.[/yellow]")
        raise typer.Exit(1)

    with console.status("Contacting endpoints…"):
        results = asyncio.run(main_probe())

    ok = True
    console.print()
    for name, healthy, detail in results:
        mark = "[green]✓[/green]" if healthy else "[red]✗[/red]"
        console.print(f"  {mark} {name:9s} {detail}")
        ok &= healthy
    console.print()
    if not ok:
        raise typer.Exit(1)


@app.command()
def run(
    pdf: Annotated[Path, typer.Argument(help="PDF to convert into a QA dataset.")],
    config_path: ConfigOption = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory.")] = None,
    no_figures: Annotated[
        bool, typer.Option("--no-figures", help="Skip figure descriptions.")
    ] = False,
    no_table_reread: Annotated[
        bool, typer.Option("--no-table-reread", help="Skip complex-table re-reads.")
    ] = False,
    no_ground: Annotated[
        bool, typer.Option("--no-ground", help="Skip the grounding filter.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Ignore cached vision results.")
    ] = False,
    max_vision_pages: Annotated[
        int | None, typer.Option("--max-vision-pages", help="Cap billable vision pages.")
    ] = None,
) -> None:
    """Run the full pipeline: parse, vision, chunk, generate, ground, write."""
    from .pipeline import run_async

    cfg = _load_config(config_path)
    cfg.vision.describe_figures = cfg.vision.describe_figures and not no_figures
    cfg.vision.table_reread = cfg.vision.table_reread and not no_table_reread
    cfg.ground.enabled = cfg.ground.enabled and not no_ground
    cfg.use_cache = cfg.use_cache and not no_cache
    if max_vision_pages is not None:
        cfg.vision.max_vision_pages = max_vision_pages
    if out is not None:
        cfg.output.dir = out

    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console, transient=False) as progress:
        result = asyncio.run(run_async(pdf, cfg, progress=_RichProgress(progress)))

    m = result.manifest
    pairs = m["pairs"]
    console.print()
    console.print("[bold green]Done.[/bold green]")
    # Print the directory once; full paths on every line wrap badly in narrow terminals.
    console.print(f"  output dir  [bold]{cfg.output.dir}[/bold]")
    console.print(f"    {cfg.output.dataset_name:16s} [bold]{pairs['kept']}[/bold] pairs")
    console.print(f"    {cfg.output.rejected_name:16s} {pairs['rejected']} rejected")
    console.print(f"    {cfg.output.manifest_name:16s} run details")
    console.print()
    console.print(
        f"  {m['chunks']} chunks · {m['vision_calls_total']} vision calls · "
        f"yield {pairs['yield']:.0%} · {m['timings']['total_s']}s"
    )
    if pairs["rejected"]:
        console.print(
            f"  [dim]{pairs['rejected']} pair(s) dropped by the grounding filter; "
            f"see {cfg.output.rejected_name} for reasons.[/dim]"
        )
    console.print()


def run_cli() -> None:
    try:
        app()
    except PyqaError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    run_cli()
