"""End-to-end passthrough tests against a mock OpenAI-compatible backend.

A real socket on both hops (client -> proxy -> mock backend) so we exercise the
actual streaming/framing code, not a stubbed client. Covers: non-streaming byte
round-trip unchanged, live SSE relay + reassembly, auth redaction in the log,
JSONL schema, and a 502 on an unreachable backend.
"""

import json        # build/parse request & response bodies
import threading   # run the mock backend in a thread; Event for the live-relay test
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # the mock backend

import httpx       # the client that stands in for Codex
import pytest      # fixtures + parametrization

from codex_probe import ProxyRecorder  # the system under test

# A canned non-streaming JSON response the mock returns for plain POSTs.
JSON_BODY = {"id": "chatcmpl-1", "model": "mock-1", "object": "chat.completion",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}

# A canned Chat Completions SSE stream: two content deltas ("Hel" + "lo") + stop.
SSE_BODY = (
    'data: {"id":"c2","model":"mock-stream","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: [DONE]\n\n'
)


# Responses-API SSE: the terminal event carries the whole response object.
RESPONSES_SSE = (
    'event: response.output_text.delta\n'
    'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
    'event: response.completed\n'
    'data: {"type":"response.completed","response":{"id":"r1","model":"mock-resp","output":[]}}\n\n'
)

# Shared record of what the backend actually received, for request-side assertions.
RECEIVED: dict = {}
# Set by the live-relay test; gates the backend's second chunk so the test can
# prove chunk 1 was relayed *before* chunk 2 was produced (i.e. no buffering).
RELEASE_SECOND_CHUNK = threading.Event()


class _MockBackend(BaseHTTPRequestHandler):
    # Silence default per-request stderr logging.
    def log_message(self, *_):
        pass

    def do_POST(self):  # noqa: N802
        # Read the request body and remember it + headers for request-side asserts.
        length = int(self.headers.get("Content-Length", 0))
        body_in = self.rfile.read(length)
        RECEIVED["body"] = body_in
        RECEIVED["headers"] = dict(self.headers)

        # Route by path suffix so one mock serves every test's needs.
        if self.path.endswith("/responses"):
            self._send_sse(RESPONSES_SSE.encode())   # Responses-API stream
        elif self.path.endswith("/drip"):
            self._drip()                             # gated two-chunk stream
        elif self.path.endswith("/stream"):
            self._send_sse(SSE_BODY.encode())        # Chat Completions stream
        else:
            # Default: a plain non-streaming JSON response with a Content-Length.
            body = json.dumps(JSON_BODY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _send_sse(self, body: bytes):
        # Send a complete SSE body at once (with a Content-Length).
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _drip(self):
        # Chunked (no Content-Length): emit chunk 1, then wait for the test to
        # confirm it saw chunk 1 before emitting chunk 2.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")  # we frame chunks by hand
        self.end_headers()
        first = b'data: {"choices":[{"index":0,"delta":{"content":"first"}}]}\n\n'
        self.wfile.write(b"%X\r\n%s\r\n" % (len(first), first))  # one HTTP/1.1 chunk
        self.wfile.flush()                                       # push it out now
        RELEASE_SECOND_CHUNK.wait(timeout=5)                     # block until test releases
        rest = b'data: {"choices":[{"index":0,"delta":{"content":"second"}}]}\n\ndata: [DONE]\n\n'
        self.wfile.write(b"%X\r\n%s\r\n0\r\n\r\n" % (len(rest), rest))  # chunk 2 + terminator
        self.wfile.flush()


@pytest.fixture
def backend():
    # Start the mock backend on a random free port; yield its base URL.
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockBackend)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}/v1"
    # Teardown: stop the server and release the socket.
    server.shutdown()
    server.server_close()


@pytest.fixture
def recorder(backend, tmp_path):
    # A recorder pointed at the mock backend, logging into a pytest temp dir.
    rec = ProxyRecorder(
        {"backend_base_url": backend, "listen_port": 0, "log_dir": str(tmp_path)}
    )
    endpoint = rec.start()
    yield rec, endpoint  # hand the test both the recorder and its local endpoint
    rec.stop()           # teardown


def test_non_streaming_round_trip_unchanged(recorder, backend):
    # Proves: request body + headers reach the backend unchanged, and the
    # response reaches the client unchanged, while the log redacts auth.
    rec, endpoint = recorder
    RECEIVED.clear()  # start from a clean record of what the backend saw
    req_payload = {"messages": [{"role": "user", "content": "x"}]}
    resp = httpx.post(f"{endpoint}/chat/completions", json=req_payload,
                      headers={"Authorization": "Bearer sk-secret"})
    assert resp.status_code == 200
    assert resp.json() == JSON_BODY  # response relayed byte-for-byte (decoded)

    # request side: backend received the body unchanged, auth unredacted, Host rewritten
    assert json.loads(RECEIVED["body"]) == req_payload
    assert RECEIVED["headers"]["Authorization"] == "Bearer sk-secret"
    assert RECEIVED["headers"]["Host"] == backend.split("//")[1].split("/")[0]

    call = rec.calls[0]
    assert len(rec.calls) == 1               # exactly one call recorded
    assert call["response_body"] == JSON_BODY
    assert call["response_model"] == "mock-1"
    assert call["backend_url"].endswith("/v1/chat/completions")
    # auth forwarded upstream but redacted in the *log*
    assert call["request_headers"]["Authorization"] == "<redacted>"


def test_responses_api_passthrough_and_reassembly(recorder):
    # Proves the /responses path is relayed and reassembled from the terminal event.
    rec, endpoint = recorder
    with httpx.stream("POST", f"{endpoint}/responses", json={"input": "hi"}) as r:
        relayed = "".join(r.iter_text())
    assert "response.completed" in relayed    # client saw the full event stream
    call = rec.calls[0]
    assert call["response_body"]["id"] == "r1"
    assert call["response_model"] == "mock-resp"  # reassembled from terminal event


def test_stream_is_relayed_live_not_buffered(recorder):
    # Proves the proxy forwards chunk 1 before the backend has produced chunk 2.
    rec, endpoint = recorder
    RELEASE_SECOND_CHUNK.clear()  # backend will block after chunk 1 until we set this
    with httpx.stream("POST", f"{endpoint}/chat/completions/drip", json={},
                      timeout=10) as r:
        it = r.iter_lines()
        first_line = next(l for l in it if l.strip())  # arrives before chunk 2 exists
        assert "first" in first_line
        RELEASE_SECOND_CHUNK.set()  # now let the backend finish
        rest = "".join(it)          # consume the remainder of the stream
    assert "second" in rest
    # Both chunks reassemble into one message content.
    assert rec.calls[0]["response_body"]["choices"][0]["message"]["content"] == "firstsecond"


def test_streaming_relayed_and_reassembled(recorder):
    # Proves a full SSE stream is relayed intact and reassembled; raw SSE is kept.
    rec, endpoint = recorder
    chunks = []
    with httpx.stream("POST", f"{endpoint}/chat/completions/stream", json={}) as r:
        for line in r.iter_lines():
            chunks.append(line)
    relayed = "\n".join(chunks)
    assert "[DONE]" in relayed  # the live stream reached the client intact

    call = rec.calls[0]
    assert call["streamed"] is True
    assert call["response_body"]["choices"][0]["message"]["content"] == "Hello"  # "Hel"+"lo"
    assert call["response_raw"] is not None  # full unabridged SSE kept


def test_log_schema_and_jsonl_file(recorder):
    # Proves the record has the expected keys and one JSONL line is written.
    rec, endpoint = recorder
    httpx.post(f"{endpoint}/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})

    # The stable schema a reviewer/replayer can rely on.
    required = {"index", "timestamp", "method", "path", "backend_url",
               "request_headers", "request_body", "response_status",
               "response_headers", "response_model", "streamed",
               "response_body", "response_raw", "latency_ms", "error", "seed"}
    assert required <= set(rec.calls[0])  # record has (at least) all required keys

    # JSONL file written and parseable, one line per call
    lines = rec.log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["index"] == 1


def test_unreachable_backend_returns_502(tmp_path):
    # Proves a connect failure returns 502 to the client and is still logged.
    rec = ProxyRecorder(
        {"backend_base_url": "http://127.0.0.1:1/v1", "listen_port": 0,  # port 1 = refused
         "log_dir": str(tmp_path), "connect_timeout": 1.0}
    )
    endpoint = rec.start()
    try:
        resp = httpx.post(f"{endpoint}/chat/completions", json={})
        assert resp.status_code == 502
        assert rec.calls[0]["error"] is not None  # the failure is captured
    finally:
        rec.stop()
