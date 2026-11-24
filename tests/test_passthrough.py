"""End-to-end passthrough tests against a mock OpenAI-compatible backend.

A real socket on both hops (client -> proxy -> mock backend) so we exercise the
actual streaming/framing code, not a stubbed client. Covers: non-streaming byte
round-trip unchanged, live SSE relay + reassembly, auth redaction in the log,
JSONL schema, and a 502 on an unreachable backend.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from codex_probe import ProxyRecorder

JSON_BODY = {"id": "chatcmpl-1", "model": "mock-1", "object": "chat.completion",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}

SSE_BODY = (
    'data: {"id":"c2","model":"mock-stream","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: [DONE]\n\n'
)


RESPONSES_SSE = (
    'event: response.output_text.delta\n'
    'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
    'event: response.completed\n'
    'data: {"type":"response.completed","response":{"id":"r1","model":"mock-resp","output":[]}}\n\n'
)

RECEIVED: dict = {}
RELEASE_SECOND_CHUNK = threading.Event()


class _MockBackend(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body_in = self.rfile.read(length)
        RECEIVED["body"] = body_in
        RECEIVED["headers"] = dict(self.headers)

        if self.path.endswith("/responses"):
            self._send_sse(RESPONSES_SSE.encode())
        elif self.path.endswith("/drip"):
            self._drip()
        elif self.path.endswith("/stream"):
            self._send_sse(SSE_BODY.encode())
        else:
            body = json.dumps(JSON_BODY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _send_sse(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _drip(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        first = b'data: {"choices":[{"index":0,"delta":{"content":"first"}}]}\n\n'
        self.wfile.write(b"%X\r\n%s\r\n" % (len(first), first))
        self.wfile.flush()
        RELEASE_SECOND_CHUNK.wait(timeout=5)
        rest = b'data: {"choices":[{"index":0,"delta":{"content":"second"}}]}\n\ndata: [DONE]\n\n'
        self.wfile.write(b"%X\r\n%s\r\n0\r\n\r\n" % (len(rest), rest))
        self.wfile.flush()


@pytest.fixture
def backend():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockBackend)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}/v1"
    server.shutdown()
    server.server_close()


@pytest.fixture
def recorder(backend, tmp_path):
    rec = ProxyRecorder(
        {"backend_base_url": backend, "listen_port": 0, "log_dir": str(tmp_path)}
    )
    endpoint = rec.start()
    yield rec, endpoint
    rec.stop()


def test_non_streaming_round_trip_unchanged(recorder, backend):
    rec, endpoint = recorder
    RECEIVED.clear()
    req_payload = {"messages": [{"role": "user", "content": "x"}]}
    resp = httpx.post(f"{endpoint}/chat/completions", json=req_payload,
                      headers={"Authorization": "Bearer sk-secret"})
    assert resp.status_code == 200
    assert resp.json() == JSON_BODY

    assert json.loads(RECEIVED["body"]) == req_payload
    assert RECEIVED["headers"]["Authorization"] == "Bearer sk-secret"
    assert RECEIVED["headers"]["Host"] == backend.split("//")[1].split("/")[0]

    call = rec.calls[0]
    assert len(rec.calls) == 1
    assert call["response_body"] == JSON_BODY
    assert call["response_model"] == "mock-1"
    assert call["backend_url"].endswith("/v1/chat/completions")
    assert call["request_headers"]["Authorization"] == "<redacted>"


def test_responses_api_passthrough_and_reassembly(recorder):
    rec, endpoint = recorder
    with httpx.stream("POST", f"{endpoint}/responses", json={"input": "hi"}) as r:
        relayed = "".join(r.iter_text())
    assert "response.completed" in relayed
    call = rec.calls[0]
    assert call["response_body"]["id"] == "r1"
    assert call["response_model"] == "mock-resp"


def test_stream_is_relayed_live_not_buffered(recorder):
    rec, endpoint = recorder
    RELEASE_SECOND_CHUNK.clear()
    with httpx.stream("POST", f"{endpoint}/chat/completions/drip", json={},
                      timeout=10) as r:
        it = r.iter_lines()
        first_line = next(l for l in it if l.strip())
        assert "first" in first_line
        RELEASE_SECOND_CHUNK.set()
        rest = "".join(it)
    assert "second" in rest
    assert rec.calls[0]["response_body"]["choices"][0]["message"]["content"] == "firstsecond"


def test_streaming_relayed_and_reassembled(recorder):
    rec, endpoint = recorder
    chunks = []
    with httpx.stream("POST", f"{endpoint}/chat/completions/stream", json={}) as r:
        for line in r.iter_lines():
            chunks.append(line)
    relayed = "\n".join(chunks)
    assert "[DONE]" in relayed

    call = rec.calls[0]
    assert call["streamed"] is True
    assert call["response_body"]["choices"][0]["message"]["content"] == "Hello"
    assert call["response_raw"] is not None


def test_log_schema_and_jsonl_file(recorder):
    rec, endpoint = recorder
    httpx.post(f"{endpoint}/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})

    required = {"index", "timestamp", "method", "path", "backend_url",
               "request_headers", "request_body", "response_status",
               "response_headers", "response_model", "streamed",
               "response_body", "response_raw", "latency_ms", "error", "seed"}
    assert required <= set(rec.calls[0])

    lines = rec.log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["index"] == 1


def test_unreachable_backend_returns_502(tmp_path):
    rec = ProxyRecorder(
        {"backend_base_url": "http://127.0.0.1:1/v1", "listen_port": 0,
         "log_dir": str(tmp_path), "connect_timeout": 1.0}
    )
    endpoint = rec.start()
    try:
        resp = httpx.post(f"{endpoint}/chat/completions", json={})
        assert resp.status_code == 502
        assert rec.calls[0]["error"] is not None
    finally:
        rec.stop()
