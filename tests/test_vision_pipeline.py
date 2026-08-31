"""Vision stage end-to-end: routing, splicing, budgets, and cache reuse."""

from __future__ import annotations

import json

from hootie.pipeline import run_async


def _image_requests(state) -> list:
    return [
        req
        for req in state.requests
        if any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
            for m in req.get("messages", [])
        )
    ]


async def test_figure_pages_get_described_and_spliced(figures_pdf, stub_config, stub):
    state, _ = stub
    state.vision_reply = "```figure\nA flowchart routing applications by score band.\n```"

    result = await run_async(figures_pdf, stub_config)

    # Only the two figure-bearing pages cost a call; the logo pages do not.
    assert result.manifest["vision_calls"]["describe"] == 2
    assert result.manifest["figure_pages"] == [2, 3]
    assert len(_image_requests(state)) == 2

    # The description is spliced into the document, not appended in a pile.
    assert "```figure page=2" in result.doc.markdown
    assert "flowchart routing applications" in result.doc.markdown
    assert any(c.contains_figure for c in result.chunks)


async def test_description_lands_next_to_its_page_text(figures_pdf, stub_config, stub):
    state, _ = stub
    state.vision_reply = "```figure\nUNIQUEMARKER chart.\n```"

    result = await run_async(figures_pdf, stub_config)
    markdown = result.doc.markdown

    # It must sit inside page 2's span, not drift to the end of the document.
    span = next(s for s in result.doc.spans if s.page == 2)
    marker = markdown.index("UNIQUEMARKER")
    assert span.start <= marker < span.end


async def test_no_figures_flag_skips_all_vision_calls(figures_pdf, stub_config, stub):
    state, _ = stub
    stub_config.vision.describe_figures = False

    result = await run_async(figures_pdf, stub_config)
    assert result.manifest["vision_calls_total"] == 0
    assert _image_requests(state) == []


async def test_max_vision_pages_caps_spend(figures_pdf, stub_config, stub):
    state, _ = stub
    stub_config.vision.max_vision_pages = 1

    result = await run_async(figures_pdf, stub_config)
    assert result.manifest["vision_calls_total"] == 1
    assert len(_image_requests(state)) == 1


async def test_unfenced_vision_output_is_still_wrapped(figures_pdf, stub_config, stub):
    """Models ignore the fencing instruction sometimes; output must survive."""
    state, _ = stub
    state.vision_reply = "A plain unfenced description of the chart."

    result = await run_async(figures_pdf, stub_config)
    assert "```figure page=2" in result.doc.markdown
    assert "plain unfenced description" in result.doc.markdown


async def test_second_run_reuses_cached_vision_results(figures_pdf, stub_config, stub):
    """Vision is the expensive stage; a resumed run must not pay for it twice."""
    state, _ = stub

    first = await run_async(figures_pdf, stub_config)
    assert first.manifest["vision_calls_total"] == 2
    calls_after_first = len(_image_requests(state))
    assert calls_after_first == 2

    second = await run_async(figures_pdf, stub_config)
    assert len(_image_requests(state)) == calls_after_first, "cached run must send no images"
    assert "```figure page=2" in second.doc.markdown


async def test_no_cache_forces_fresh_vision_calls(figures_pdf, stub_config, stub):
    state, _ = stub
    await run_async(figures_pdf, stub_config)
    stub_config.use_cache = False
    await run_async(figures_pdf, stub_config)
    assert len(_image_requests(state)) == 4


async def test_vision_failures_within_budget_do_not_abort_the_run(figures_pdf, stub_config, stub):
    state, _ = stub
    state.fail_first_n = 1  # one page fails every retry
    stub_config.vision.endpoint = stub_config.vision.endpoint.model_copy(update={"max_retries": 0})
    stub_config.vision.failure_budget = 5

    result = await run_async(figures_pdf, stub_config)
    assert result.kept, "a single bad page must not sink the run"


async def test_manifest_records_vision_breakdown(figures_pdf, stub_config):
    await run_async(figures_pdf, stub_config)
    manifest_path = stub_config.output.dir / stub_config.output.manifest_name
    manifest = json.loads(manifest_path.read_text())

    assert set(manifest["vision_calls"]) == {"ocr", "table_reread", "describe"}
    assert manifest["vision_calls_total"] == sum(manifest["vision_calls"].values())
    assert manifest["models"]["vision"] == "stub-model"
