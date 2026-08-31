"""End-to-end pipeline runs against the stub server — no credentials, no network."""

from __future__ import annotations

import json

import pytest

from hootie.errors import HootieError
from hootie.pipeline import run_async
from hootie.types import Stage


async def test_full_run_produces_a_chat_jsonl_dataset(native_pdf, stub_config):
    result = await run_async(native_pdf, stub_config)

    dataset = stub_config.output.dir / stub_config.output.dataset_name
    assert dataset.is_file()

    lines = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert lines, "dataset must not be empty"
    for record in lines:
        roles = [m["role"] for m in record["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert all(m["content"] for m in record["messages"])
    assert len(lines) == len(result.kept)


async def test_manifest_counts_reconcile(native_pdf, stub_config):
    result = await run_async(native_pdf, stub_config)
    m = result.manifest
    pairs = m["pairs"]

    assert pairs["generated"] == pairs["kept"] + pairs["rejected"]
    assert m["chunks"] == len(result.chunks)
    assert m["source"]["page_count"] == 3
    assert m["source"]["sha256"]
    assert m["models"]["generate"] == "stub-model"
    assert m["structured_output_mode"] == "json_schema"


async def test_rejected_pairs_are_written_with_reasons(native_pdf, stub_config, stub):
    state, _ = stub
    state.ground_verdict = False  # reject everything

    result = await run_async(native_pdf, stub_config)
    assert result.kept == []
    assert result.rejected

    rejected = stub_config.output.dir / stub_config.output.rejected_name
    records = [json.loads(line) for line in rejected.read_text().splitlines()]
    assert records
    for record in records:
        assert record["reason"] == "not in passage"
        assert record["provenance"]["chunk_id"]
    assert result.manifest["pairs"]["yield"] == 0.0


async def test_grounding_can_be_disabled(native_pdf, stub_config, stub):
    state, _ = stub
    state.ground_verdict = False
    stub_config.ground.enabled = False

    result = await run_async(native_pdf, stub_config)
    assert result.rejected == []
    assert result.kept, "with grounding off, nothing should be filtered"


async def test_provenance_is_optional_in_the_training_file(native_pdf, stub_config):
    await run_async(native_pdf, stub_config)
    plain = json.loads(
        (stub_config.output.dir / stub_config.output.dataset_name).read_text().splitlines()[0]
    )
    assert "provenance" not in plain

    stub_config.output.include_provenance = True
    await run_async(native_pdf, stub_config)
    rich = json.loads(
        (stub_config.output.dir / stub_config.output.dataset_name).read_text().splitlines()[0]
    )
    assert rich["provenance"]["chunk_id"]
    assert rich["provenance"]["first_page"] >= 1


async def test_structured_output_downgrade_still_produces_a_dataset(native_pdf, stub_config, stub):
    """A server supporting neither schema mechanism must still yield training data."""
    state, _ = stub
    state.supports_json_schema = False
    state.supports_json_object = False

    result = await run_async(native_pdf, stub_config)
    assert result.kept
    assert result.manifest["structured_output_mode"] == "prompt"


async def test_progress_events_cover_every_stage(native_pdf, stub_config):
    seen: list = []
    await run_async(native_pdf, stub_config, progress=seen.append)

    stages = {e.stage for e in seen}
    assert {Stage.PARSE, Stage.CHUNK, Stage.GENERATE, Stage.GROUND, Stage.WRITE} <= stages
    # Counters must never run past their total, or the bar renders nonsense.
    assert all(e.total is None or e.completed <= e.total for e in seen)


async def test_native_pdf_makes_no_vision_calls(native_pdf, stub_config, stub):
    state, _ = stub
    result = await run_async(native_pdf, stub_config)

    assert result.manifest["vision_calls_total"] == 0
    assert not any(
        isinstance(m.get("content"), list)
        for req in state.requests
        for m in req.get("messages", [])
    ), "a natively-parsing PDF must never send an image"


async def test_missing_generation_endpoint_is_a_clear_error(native_pdf, stub_config):
    stub_config.generate.endpoint = None
    with pytest.raises(HootieError, match="no generation endpoint"):
        await run_async(native_pdf, stub_config)
