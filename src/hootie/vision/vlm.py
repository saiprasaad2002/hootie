"""Vision engine backed by any OpenAI-compatible endpoint that accepts images.

Covers vLLM serving a VLM, plus hosted providers speaking the same protocol.
Images are sent inline as base64 data URIs, which every OpenAI-compatible vision
server accepts and which keeps the user's pages off any intermediate host.
"""

from __future__ import annotations

import base64
import logging

from ..errors import VisionError
from ..types import PagePlan, PageTask
from .prompts import SYSTEM_BY_TASK, user_prompt

logger = logging.getLogger("hootie.vision")


class VlmVisionEngine:
    """Default `VisionEngine`, driving a `ChatClient` that supports images."""

    def __init__(self, client) -> None:
        self.client = client

    async def read_page(self, image: bytes, plan: PagePlan, context: str) -> str:
        if plan.task is PageTask.NONE:
            return ""

        system = SYSTEM_BY_TASK.get(plan.task)
        if system is None:  # pragma: no cover - guarded by the enum
            raise VisionError(f"no prompt defined for task {plan.task}")

        data_uri = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt(plan.task, plan.page, context)},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]

        try:
            text = await self.client.complete(messages)
        except Exception as exc:
            raise VisionError(f"vision call failed for page {plan.page}: {exc}") from exc

        return (text or "").strip()
