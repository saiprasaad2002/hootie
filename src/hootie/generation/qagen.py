"""Generate QA pairs from a chunk, with schema validation and corrective retry."""

from __future__ import annotations

import logging

from ..config import GenerateConfig
from ..errors import GenerationError
from ..types import QAPair, SourceChunk
from .json_utils import extract_json
from .prompts import QA_SCHEMA, QA_SYSTEM, qa_user_prompt

logger = logging.getLogger("hootie.generate")

# Even in strict json_schema mode a model can return an empty or malformed list,
# so the response is validated locally regardless of how it was requested.
_PARSE_ATTEMPTS = 3


class QAGenerator:
    """Turns one chunk into QA pairs."""

    def __init__(self, client, config: GenerateConfig | None = None) -> None:
        self.client = client
        self.config = config or GenerateConfig()

    async def generate(self, chunk: SourceChunk) -> list[QAPair]:
        messages = [
            {"role": "system", "content": QA_SYSTEM},
            {
                "role": "user",
                "content": qa_user_prompt(
                    chunk.text, self.config.pairs_per_chunk, chunk.page_label
                ),
            },
        ]

        last_error: str | None = None
        for attempt in range(_PARSE_ATTEMPTS):
            if last_error:
                # Feed the parse failure back; models usually fix it in one turn.
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply could not be parsed: {last_error}. "
                            "Reply again with only the JSON object."
                        ),
                    },
                ]

            raw = await self.client.complete(messages, schema=QA_SCHEMA, schema_name="qa_pairs")
            try:
                return self._parse(raw, chunk)
            except ValueError as exc:
                last_error = str(exc)
                logger.debug(
                    "chunk %s: unparseable QA response on attempt %s: %s",
                    chunk.id,
                    attempt + 1,
                    exc,
                )

        raise GenerationError(
            f"chunk {chunk.id}: model did not return valid QA pairs after "
            f"{_PARSE_ATTEMPTS} attempts ({last_error})"
        )

    def _parse(self, raw: str, chunk: SourceChunk) -> list[QAPair]:
        data = extract_json(raw)

        # Tolerate a bare list, which models return often enough to be worth handling.
        items = data.get("pairs") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("expected an object with a 'pairs' array")

        pairs: list[QAPair] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    chunk_id=chunk.id,
                    first_page=chunk.first_page,
                    last_page=chunk.last_page,
                    contains_figure=chunk.contains_figure,
                )
            )

        if not pairs:
            raise ValueError("response contained no usable question/answer pairs")
        return pairs[: self.config.pairs_per_chunk]
