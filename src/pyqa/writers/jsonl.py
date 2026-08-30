"""Write the dataset, the rejects, and the run manifest to local disk.

Chat/messages JSONL is the format TRL, Axolotl, Unsloth and OpenAI finetuning
ingest directly, so it is the default. Provenance is kept out of the training
file by default (trainers choke on unexpected keys) but is available with
`include_provenance`, and always present in `rejected.jsonl`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..types import QAPair


def _chat_record(pair: QAPair, system_prompt: str, include_provenance: bool) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": pair.question})
    messages.append({"role": "assistant", "content": pair.answer})

    record: dict[str, Any] = {"messages": messages}
    if include_provenance:
        record["provenance"] = _provenance(pair)
    return record


def _provenance(pair: QAPair) -> dict[str, Any]:
    return {
        "chunk_id": pair.chunk_id,
        "first_page": pair.first_page,
        "last_page": pair.last_page,
        "contains_figure": pair.contains_figure,
    }


def _write_lines(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_dataset(
    pairs: Sequence[QAPair],
    path: Path,
    *,
    system_prompt: str = "",
    include_provenance: bool = False,
) -> int:
    """Write kept pairs as chat/messages JSONL. Returns the line count."""
    return _write_lines(
        path,
        (_chat_record(p, system_prompt, include_provenance) for p in pairs),
    )


def write_rejected(pairs: Sequence[QAPair], path: Path) -> int:
    """Write dropped pairs with their reasons, so yield can be audited."""
    return _write_lines(
        path,
        (
            {
                "question": p.question,
                "answer": p.answer,
                "reason": p.rejection_reason or "unsupported",
                "provenance": _provenance(p),
            }
            for p in pairs
        ),
    )


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write the run manifest, including the exact vision call counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), "utf-8")
