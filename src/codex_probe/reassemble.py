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

# Enable modern annotation syntax on all supported Pythons.
from __future__ import annotations

import json  # to parse each SSE "data:" payload
from typing import Any  # loose typing for JSON-ish values


def parse_sse(raw: str) -> list[dict[str, Any] | str]:
    """Parse SSE text into the ordered list of ``data:`` payloads.

    JSON payloads are returned as dicts; the sentinel ``[DONE]`` and any
    non-JSON payload are returned as the raw string. Event lines are ignored
    here — the data payloads carry their own ``type``/``object`` discriminator.
    """
    out: list[dict[str, Any] | str] = []
    # SSE events are separated by a blank line. Normalize CRLF first, then split
    # on the blank-line boundary so each `block` is one complete event.
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        # One event may carry several "data:" lines; collect them in order.
        data_lines = [
            line[len("data:"):].lstrip(" ")  # strip the "data:" prefix and one space
            for line in block.split("\n")
            if line.startswith("data:")  # ignore "event:"/"id:"/comment lines
        ]
        if not data_lines:
            continue  # a block with no data (e.g. just an event: line) carries nothing
        # Per the SSE spec, multiple data lines are joined with newlines.
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            out.append(payload)  # OpenAI's end-of-stream sentinel; keep as marker
            continue
        try:
            out.append(json.loads(payload))  # the normal case: a JSON event object
        except json.JSONDecodeError:
            out.append(payload)  # keep non-JSON payloads rather than dropping them
    return out


def reassemble_chat(events: list[dict[str, Any] | str]) -> dict[str, Any]:
    """Merge Chat Completions streaming chunks into one completion object."""
    # The envelope fields (id/model/usage/...) that we copy from the chunks.
    base: dict[str, Any] = {"object": "chat.completion", "choices": []}
    # index -> accumulated choice state (a stream can carry multiple choices).
    choices: dict[int, dict[str, Any]] = {}

    for ev in events:
        if not isinstance(ev, dict):
            continue  # skip "[DONE]" and any non-JSON payloads
        # Carry over top-level envelope fields whenever a chunk includes them.
        for key in ("id", "model", "created", "system_fingerprint"):
            if key in ev:
                base[key] = ev[key]
        if ev.get("usage"):  # some servers emit a final usage-only chunk
            base["usage"] = ev["usage"]

        # Each chunk carries zero or more choice deltas; merge each by its index.
        for ch in ev.get("choices", []) or []:
            idx = ch.get("index", 0)
            # First time we see this index, seed an empty assistant message.
            acc = choices.setdefault(
                idx,
                {"index": idx, "message": {"role": "assistant", "content": ""},
                 "finish_reason": None},
            )
            # finish_reason arrives on the last delta for the choice.
            if ch.get("finish_reason") is not None:
                acc["finish_reason"] = ch["finish_reason"]
            delta = ch.get("delta", {}) or {}  # the incremental piece for this chunk
            if delta.get("role"):
                acc["message"]["role"] = delta["role"]  # usually only on the first delta
            if delta.get("content"):
                acc["message"]["content"] += delta["content"]  # concatenate text fragments
            # Tool calls also arrive as fragments; merge them separately.
            _merge_tool_calls(acc["message"], delta.get("tool_calls"))

    # Finalize each accumulated choice.
    for acc in choices.values():
        msg = acc["message"]
        # OpenAI uses content=None (not "") when the turn is purely tool calls.
        if not msg.get("content") and "tool_calls" in msg:
            msg["content"] = None  # OpenAI convention when only tool calls returned
        # Convert the index-keyed tool_calls dict back into an ordered list.
        if "tool_calls" in msg:
            msg["tool_calls"] = [msg["tool_calls"][i] for i in sorted(msg["tool_calls"])]
    # Emit choices in index order.
    base["choices"] = [choices[i] for i in sorted(choices)]
    return base


def _merge_tool_calls(message: dict[str, Any], tool_deltas: Any) -> None:
    """Accumulate streamed tool-call fragments, keyed by their ``index``."""
    if not tool_deltas:
        return  # nothing to merge for this delta
    # tool_calls accumulate in a dict keyed by index until finalized into a list.
    store: dict[int, dict[str, Any]] = message.setdefault("tool_calls", {})
    for td in tool_deltas:
        i = td.get("index", 0)  # which tool call this fragment belongs to
        # Seed a blank tool-call record the first time we see this index.
        tc = store.setdefault(i, {"id": None, "type": "function",
                                  "function": {"name": "", "arguments": ""}})
        if td.get("id"):
            tc["id"] = td["id"]        # the call id (arrives once)
        if td.get("type"):
            tc["type"] = td["type"]    # e.g. "function"
        fn = td.get("function", {}) or {}
        if fn.get("name"):
            tc["function"]["name"] += fn["name"]            # name may be fragmented
        if fn.get("arguments"):
            tc["function"]["arguments"] += fn["arguments"]  # JSON args stream in pieces


def reassemble_responses(events: list[dict[str, Any] | str]) -> dict[str, Any]:
    """Pull the full response object out of a Responses-API event stream.

    The terminal ``response.completed`` / ``response.failed`` /
    ``response.incomplete`` event carries the whole ``response`` object, so we
    prefer it and fall back to the last event that embeds one.
    """
    # Any of these terminal event types carries the final, complete response.
    terminal = {"response.completed", "response.failed", "response.incomplete"}
    last_with_response: dict[str, Any] | None = None  # fallback if no terminal event seen
    for ev in events:
        # Many Responses events embed a snapshot of the response object.
        if isinstance(ev, dict) and isinstance(ev.get("response"), dict):
            last_with_response = ev["response"]  # remember the most recent snapshot
            if ev.get("type") in terminal:
                return ev["response"]  # the authoritative final object
    if last_with_response is not None:
        return last_with_response  # best effort: latest snapshot we saw
    # Stream had no response object at all (malformed/empty) — return a marker.
    return {"object": "response", "_note": "no response object found in stream"}


def reassemble(raw_text: str, path: str) -> dict[str, Any]:
    """Reassemble ``raw_text`` into a full response object, dispatched by path."""
    events = parse_sse(raw_text)  # parse once, over the full text
    # Choose the reassembler by the request path (robust to backend/config drift).
    if "/responses" in path:
        return reassemble_responses(events)
    return reassemble_chat(events)
