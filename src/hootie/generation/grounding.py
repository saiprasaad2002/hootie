"""Check that each generated answer is actually supported by its source chunk.

This is the difference between a dataset that teaches a model the document and
one that teaches it to hallucinate confidently. Rejected pairs are not discarded
silently: they are written out with reasons so yield can be audited.
"""

from __future__ import annotations

import logging

from ..config import GroundConfig
from ..types import QAPair, SourceChunk
from .json_utils import extract_json
from .prompts import GROUNDING_SCHEMA, GROUNDING_SYSTEM, grounding_user_prompt

logger = logging.getLogger("hootie.ground")


class GroundingChecker:
    """Verdicts on whether an answer is supported by its passage."""

    def __init__(self, client, config: GroundConfig | None = None) -> None:
        self.client = client
        self.config = config or GroundConfig()

    async def check(self, pair: QAPair, chunk: SourceChunk) -> QAPair:
        """Annotate `pair` with a grounding verdict, in place, and return it."""
        messages = [
            {"role": "system", "content": GROUNDING_SYSTEM},
            {
                "role": "user",
                "content": grounding_user_prompt(chunk.text, pair.question, pair.answer),
            },
        ]

        try:
            raw = await self.client.complete(
                messages, schema=GROUNDING_SCHEMA, schema_name="grounding"
            )
            data = extract_json(raw)
            if not isinstance(data, dict) or "supported" not in data:
                raise ValueError("verdict missing a 'supported' field")
            pair.grounded = bool(data["supported"])
            reason = str(data.get("reason", "")).strip()
            pair.rejection_reason = None if pair.grounded else (reason or "unsupported")
        except Exception as exc:
            # A broken checker must not quietly delete good training data.
            logger.debug("grounding check failed for %s: %s", pair.chunk_id, exc)
            pair.grounded = self.config.keep_on_error
            pair.rejection_reason = None if pair.grounded else f"grounding check failed: {exc}"
        return pair
