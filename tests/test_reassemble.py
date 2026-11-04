"""Unit tests for SSE reassembly — the trickiest correctness surface.

The important case: tool-call arguments arrive as many fragments, and SSE events
can be split across stream-chunk boundaries. We reassemble from the full text
after the stream ends, so these must merge back exactly.
"""

from codex_probe.reassemble import parse_sse, reassemble, reassemble_chat


def _sse(*payloads: str) -> str:
    # Build an SSE string from JSON payloads, ending with the [DONE] sentinel.
    return "".join(f"data: {p}\n\n" for p in payloads) + "data: [DONE]\n\n"


def test_chat_content_merges():
    # Two content deltas ("Hel" + "lo") plus a finish_reason should merge to "Hello".
    raw = _sse(
        '{"id":"x","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}',
        '{"choices":[{"index":0,"delta":{"content":"lo"}}]}',
        '{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    )
    out = reassemble(raw, "/v1/chat/completions")  # path => chat reassembler
    assert out["choices"][0]["message"]["content"] == "Hello"  # fragments concatenated
    assert out["choices"][0]["finish_reason"] == "stop"        # carried from last delta
    assert out["model"] == "m"                                 # envelope field preserved


def test_chat_tool_call_arguments_merge_across_fragments():
    # A tool call whose name and JSON arguments arrive in several fragments.
    raw = _sse(
        '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_"}}]}}]}',
        '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"weather","arguments":"{\\"ci"}}]}}]}',
        '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\\": \\"SF\\"}"}}]}}]}',
        '{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
    )
    out = reassemble(raw, "/v1/chat/completions")
    tc = out["choices"][0]["message"]["tool_calls"][0]  # the single merged tool call
    assert tc["id"] == "call_1"                          # id arrives once, preserved
    assert tc["function"]["name"] == "get_weather"       # "get_" + "weather"
    assert tc["function"]["arguments"] == '{"city": "SF"}'  # three arg fragments joined


def test_event_split_across_chunk_boundary_is_fine_because_we_join_first():
    # Two stream chunks that split one SSE event mid-JSON; after joining the raw
    # text, parse_sse sees one well-formed event.
    chunk_a = 'data: {"choices":[{"index":0,"delta":{"content":"ab'  # event starts here...
    chunk_b = 'c"}}]}\n\ndata: [DONE]\n\n'                            # ...and finishes here
    out = reassemble_chat(parse_sse(chunk_a + chunk_b))  # join before parsing
    assert out["choices"][0]["message"]["content"] == "abc"


def test_responses_api_takes_terminal_response_object():
    # For the Responses API, the terminal event carries the whole response object.
    raw = (
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"r1","model":"gpt","output":[{"type":"message"}]}}\n\n'
    )
    out = reassemble(raw, "/v1/responses")  # path => responses reassembler
    assert out["id"] == "r1"                # taken straight from response.completed
    assert out["model"] == "gpt"
