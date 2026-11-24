"""Reassemble a streamed (SSE) Responses-API reply into one full response object.

Codex streams model replies as Server-Sent Events over the Responses API. For the
log we want the complete response the way a non-streaming call would have returned
it.

The Responses API makes this easy: the backend assembles everything — including
text and tool-call arguments that arrived as `...delta` events — and emits the
finished object in a terminal `response.completed` event. So we parse the stream
and take that final object; we trust the backend's assembly, and the recorder
keeps the raw SSE text alongside it in the log as ground truth.
"""

from __future__ import annotations

import json
from typing import Any

# Terminal event types whose `response` holds the final, fully-assembled object.
TERMINAL = {"response.completed", "response.failed", "response.incomplete"}


def parse_sse(raw: str) -> list[dict[str, Any]]:
    """Parse SSE text into its ordered list of JSON ``data:`` events.

    Events are separated by a blank line. Non-JSON payloads (such as the ``[DONE]``
    sentinel) are skipped — every event we care about is a JSON object carrying its
    own ``type``.
    """
    events: list[dict[str, Any]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[len("data:"):].strip()
            for line in block.split("\n")
            if line.startswith("data:")
        )
        if not data:
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            pass  # e.g. the "[DONE]" sentinel — not a response object
    return events


def reassemble(raw_text: str) -> dict[str, Any]:
    """Return the full response object from a Responses-API SSE stream.

    Prefers the terminal ``response.completed`` / ``.failed`` / ``.incomplete``
    event, which carries the whole assembled object. Falls back to the last
    response snapshot seen if the stream was cut off before a terminal event, and
    to a marker if the stream carried no response object at all.
    """
    latest: dict[str, Any] | None = None
    for ev in parse_sse(raw_text):
        if isinstance(ev.get("response"), dict):
            latest = ev["response"]
            if ev.get("type") in TERMINAL:
                return latest
    return latest or {"object": "response", "_note": "no response object in stream"}
