"""Reassemble a streamed (SSE) LLM response into one full response object.

Codex streams responses as Server-Sent Events. For the log we want the complete
response the way a non-streaming call would have returned it. Reassembly runs
*once, after the stream closes*, over the full decoded text — never per chunk —
because a single SSE event can be split across TCP/stream chunk boundaries.

Two wire formats are handled, dispatched by request path:

* Chat Completions (``/chat/completions``): many ``chat.completion.chunk``
  events carrying ``choices[].delta``; we merge content and tool-call argument
  fragments back together.
* Responses API (``/responses``): the terminal ``response.completed`` event
  already carries the whole ``response`` object, so we just take it.
"""

from __future__ import annotations

import json
from typing import Any


def parse_sse(raw: str) -> list[dict[str, Any] | str]:
    """Parse SSE text into the ordered list of ``data:`` payloads.

    JSON payloads are returned as dicts; the sentinel ``[DONE]`` and any
    non-JSON payload are returned as the raw string. Event lines are ignored
    here — the data payloads carry their own ``type``/``object`` discriminator.
    """
    out: list[dict[str, Any] | str] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        data_lines = [
            line[len("data:"):].lstrip(" ")
            for line in block.split("\n")
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            out.append(payload)
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            out.append(payload)
    return out


def reassemble_chat(events: list[dict[str, Any] | str]) -> dict[str, Any]:
    """Merge Chat Completions streaming chunks into one completion object."""
    base: dict[str, Any] = {"object": "chat.completion", "choices": []}
    choices: dict[int, dict[str, Any]] = {}

    for ev in events:
        if not isinstance(ev, dict):
            continue
        for key in ("id", "model", "created", "system_fingerprint"):
            if key in ev:
                base[key] = ev[key]
        if ev.get("usage"):
            base["usage"] = ev["usage"]

        for ch in ev.get("choices", []) or []:
            idx = ch.get("index", 0)
            acc = choices.setdefault(
                idx,
                {"index": idx, "message": {"role": "assistant", "content": ""},
                 "finish_reason": None},
            )
            if ch.get("finish_reason") is not None:
                acc["finish_reason"] = ch["finish_reason"]
            delta = ch.get("delta", {}) or {}
            if delta.get("role"):
                acc["message"]["role"] = delta["role"]
            if delta.get("content"):
                acc["message"]["content"] += delta["content"]
            _merge_tool_calls(acc["message"], delta.get("tool_calls"))

    for acc in choices.values():
        msg = acc["message"]
        if not msg.get("content") and "tool_calls" in msg:
            msg["content"] = None
        if "tool_calls" in msg:
            msg["tool_calls"] = [msg["tool_calls"][i] for i in sorted(msg["tool_calls"])]
    base["choices"] = [choices[i] for i in sorted(choices)]
    return base


def _merge_tool_calls(message: dict[str, Any], tool_deltas: Any) -> None:
    """Accumulate streamed tool-call fragments, keyed by their ``index``."""
    if not tool_deltas:
        return
    store: dict[int, dict[str, Any]] = message.setdefault("tool_calls", {})
    for td in tool_deltas:
        i = td.get("index", 0)
        tc = store.setdefault(i, {"id": None, "type": "function",
                                  "function": {"name": "", "arguments": ""}})
        if td.get("id"):
            tc["id"] = td["id"]
        if td.get("type"):
            tc["type"] = td["type"]
        fn = td.get("function", {}) or {}
        if fn.get("name"):
            tc["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            tc["function"]["arguments"] += fn["arguments"]


def reassemble_responses(events: list[dict[str, Any] | str]) -> dict[str, Any]:
    """Pull the full response object out of a Responses-API event stream.

    The terminal ``response.completed`` / ``response.failed`` /
    ``response.incomplete`` event carries the whole ``response`` object, so we
    prefer it and fall back to the last event that embeds one.
    """
    terminal = {"response.completed", "response.failed", "response.incomplete"}
    last_with_response: dict[str, Any] | None = None
    for ev in events:
        if isinstance(ev, dict) and isinstance(ev.get("response"), dict):
            last_with_response = ev["response"]
            if ev.get("type") in terminal:
                return ev["response"]
    if last_with_response is not None:
        return last_with_response
    return {"object": "response", "_note": "no response object found in stream"}


def reassemble(raw_text: str, path: str) -> dict[str, Any]:
    """Reassemble ``raw_text`` into a full response object, dispatched by path."""
    events = parse_sse(raw_text)
    if "/responses" in path:
        return reassemble_responses(events)
    return reassemble_chat(events)
