"""A minimal OpenAI-compatible server for tests.

Lets the whole pipeline run in CI with no credentials and no network. It routes
on the system prompt, so one server serves generation, grounding, and vision.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class StubState:
    """Knobs the tests use to shape the stub's behaviour."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.supports_json_schema = True
        self.supports_json_object = True
        self.pairs_per_chunk = 2
        self.ground_verdict = True
        self.vision_reply = "```figure\nA bar chart of approval rates.\n```"
        self.fail_first_n = 0
        self._calls = 0
        self.lock = threading.Lock()

    def next_call(self) -> int:
        with self.lock:
            self._calls += 1
            return self._calls


def _classify(messages: list[dict[str, Any]]) -> str:
    system = " ".join(
        m.get("content", "")
        for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    )
    if "supported by a source passage" in system:
        return "ground"
    if "question-answer pairs" in system:
        return "qa"
    if any(
        isinstance(m.get("content"), list)
        and any(part.get("type") == "image_url" for part in m["content"])
        for m in messages
    ):
        return "vision"
    return "echo"


def make_server(state: StubState) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence stderr noise
            pass

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append(request)

            if state.fail_first_n and state.next_call() <= state.fail_first_n:
                self._error(503, "temporarily unavailable")
                return

            fmt = (request.get("response_format") or {}).get("type")
            if fmt == "json_schema" and not state.supports_json_schema:
                self._error(400, "response_format json_schema is not supported")
                return
            if fmt == "json_object" and not state.supports_json_object:
                self._error(400, "response_format is unsupported by this model")
                return

            kind = _classify(request.get("messages", []))
            if kind == "qa":
                content = json.dumps(
                    {
                        "pairs": [
                            {
                                "question": f"Stub question {i + 1}?",
                                "answer": f"Stub answer {i + 1}.",
                            }
                            for i in range(state.pairs_per_chunk)
                        ]
                    }
                )
            elif kind == "ground":
                content = json.dumps(
                    {
                        "supported": state.ground_verdict,
                        "reason": "supported" if state.ground_verdict else "not in passage",
                    }
                )
            elif kind == "vision":
                content = state.vision_reply
            else:
                content = "ok"

            self._json(
                200,
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "model": request.get("model", "stub"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)
