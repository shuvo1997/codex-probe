"""Unit tests for SSE reassembly (Responses API).

The backend assembles text and tool-call arguments and emits the finished object
in a terminal `response.completed` event; we parse the stream and take that object.
We also check that an event split across chunk boundaries is fine, because we join
the full text before parsing.
"""

from codex_probe.reassemble import parse_sse, reassemble


def test_takes_terminal_response_object():
    raw = (
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"r1","model":"gpt","output":[{"type":"message"}]}}\n\n'
    )
    out = reassemble(raw)
    assert out["id"] == "r1"
    assert out["model"] == "gpt"
    assert out["output"][0]["type"] == "message"


def test_tool_call_arguments_present_in_terminal_object():
    # Arguments stream as deltas, but the terminal object carries them assembled.
    raw = (
        'event: response.function_call_arguments.delta\n'
        'data: {"type":"response.function_call_arguments.delta","delta":"{\\"ci"}\n\n'
        'event: response.function_call_arguments.delta\n'
        'data: {"type":"response.function_call_arguments.delta","delta":"ty\\": \\"SF\\"}"}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"r2","model":"m","output":[{"type":"function_call","name":"get_weather","arguments":"{\\"city\\": \\"SF\\"}"}]}}\n\n'
    )
    out = reassemble(raw)
    fc = out["output"][0]
    assert fc["type"] == "function_call"
    assert fc["name"] == "get_weather"
    assert fc["arguments"] == '{"city": "SF"}'


def test_falls_back_to_last_snapshot_if_no_terminal_event():
    raw = (
        'event: response.in_progress\n'
        'data: {"type":"response.in_progress","response":{"id":"r3","model":"m","output":[]}}\n\n'
    )
    out = reassemble(raw)
    assert out["id"] == "r3"


def test_event_split_across_chunk_boundary_is_fine():
    # We join the full text before parsing, so a mid-event boundary doesn't matter.
    a = 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r'
    b = '4","model":"m","output":[]}}\n\n'
    out = reassemble(a + b)
    assert out["id"] == "r4"


def test_empty_stream_returns_marker():
    out = reassemble("")
    assert out["_note"] == "no response object in stream"


def test_parse_sse_skips_non_json():
    events = parse_sse('data: {"type":"x"}\n\ndata: [DONE]\n\n')
    assert events == [{"type": "x"}]
