"""The recording reverse-proxy: :class:`ProxyRecorder`.

``ProxyRecorder`` listens locally on an OpenAI-compatible endpoint, forwards
every request to the configured real backend *unchanged*, relays the response
back to Codex *live* (streaming), and records the full request + reassembled
response of every call into an in-memory list and a per-session JSONL file.

Which backend is forwarded to is config only (``backend_base_url``), so the same
recorder captures both the default-OpenAI run and the swapped-in local-OSS run.
"""

# Postpone evaluation of type annotations so we can write modern syntax like
# `dict[str, Any] | None` even on older Pythons (they'd otherwise error at import).
from __future__ import annotations

import gzip          # to decompress a gzip-encoded response body for the log
import json          # to serialize each call record to a JSONL line / parse bodies
import sys           # sys.exc_info() to inspect the error in the server's handle_error
import threading     # the server runs in a background thread; a lock guards shared state
import time          # perf_counter() for per-call latency measurement
import zlib          # to decompress a deflate-encoded response body for the log
from datetime import datetime, timezone  # timestamps for log filenames and records
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # the stdlib HTTP server
from pathlib import Path  # filesystem paths for the log directory/file
from typing import Any    # loose typing for JSON-ish values

import httpx  # the HTTP client used for the upstream hop (proxy -> real backend)

# HOP_BY_HOP: header names that must not be copied across a proxy (see config.py).
from .config import HOP_BY_HOP, Config, load_config
# reassemble(): turns a streamed SSE response back into one full response object.
from .reassemble import reassemble


class ProxyRecorder:
    """Record every LLM call Codex makes, and swap its backend by config.

    Example
    -------
    >>> recorder = ProxyRecorder({"backend_base_url": "https://api.openai.com/v1"})
    >>> endpoint = recorder.start()      # -> "http://127.0.0.1:8135/v1"
    ...  # point Codex's model_providers.<name>.base_url at `endpoint`, run Codex
    >>> calls = recorder.stop()          # list[dict], one per LLM call, in order
    """

    def __init__(self, config: dict[str, Any] | Config):
        # Accept either a raw dict (validate it now) or an already-built Config.
        self.config: Config = config if isinstance(config, Config) else load_config(config)
        # All recorded calls, in request order; returned by stop() and appended in _record.
        self.calls: list[dict[str, Any]] = []
        # One lock guards the two pieces of shared state (counter + calls/log file),
        # because ThreadingHTTPServer handles each request in its own thread.
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None  # the listening server (None until start)
        self._thread: threading.Thread | None = None     # thread running serve_forever
        self._client: httpx.Client | None = None         # reused upstream HTTP client
        self._log_path: Path | None = None               # path of this session's JSONL file
        self._log_file = None                             # the open file handle for that JSONL
        self._counter = 0                                 # monotonic call index (1-based)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> str:
        """Start the proxy in a background thread and return the local endpoint.

        The returned URL (``http://host:port<endpoint_path>``) is what goes in
        Codex's ``model_providers.<name>.base_url``.
        """
        # Guard against starting twice (would leak a server/thread).
        if self._server is not None:
            raise RuntimeError("recorder already started")

        cfg = self.config  # local alias for brevity
        # Ensure the log directory exists (parents too); no error if already there.
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        # One JSONL file per session: use the configured name, else a timestamped one.
        name = cfg.session_name or datetime.now().strftime("session-%Y%m%d-%H%M%S")
        self._log_path = Path(cfg.log_dir) / f"{name}.jsonl"
        # Open in append mode so re-running with the same session_name keeps history.
        self._log_file = self._log_path.open("a", encoding="utf-8")

        # read timeout None => never time out a long/idle stream; bounded connect.
        self._client = httpx.Client(
            timeout=httpx.Timeout(None, connect=cfg.connect_timeout),
            follow_redirects=False,  # relay 3xx to the client unchanged, don't follow them
        )

        # Capture `self` so the nested Handler (instantiated per request by the
        # server) can call back into this recorder instance.
        recorder = self

        # The per-request handler. ThreadingHTTPServer creates one per connection.
        class Handler(BaseHTTPRequestHandler):
            # HTTP/1.1 is required so we can stream with chunked transfer-encoding.
            protocol_version = "HTTP/1.1"  # required for chunked streaming to Codex

            def log_message(self, *_args):  # silence default stderr access log
                pass

            # Codex only POSTs, but we proxy every verb so the tool is generic.
            def do_POST(self):  # noqa: N802 (http.server naming)
                recorder._handle(self)

            def do_GET(self):  # noqa: N802
                recorder._handle(self)

            def do_PUT(self):  # noqa: N802
                recorder._handle(self)

            def do_DELETE(self):  # noqa: N802
                recorder._handle(self)

        class _QuietServer(ThreadingHTTPServer):
            # Codex opens spare keep-alive sockets and resets them before sending
            # a request; the stdlib would print a full traceback for each. Swallow
            # just those connection-teardown errors; re-raise anything real.
            def handle_error(self, request, client_address):  # noqa: D102
                if not isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):
                    super().handle_error(request, client_address)

        # Bind the listening socket (port 0 => OS picks a free port).
        self._server = _QuietServer((cfg.listen_host, cfg.listen_port), Handler)
        # Read back the actually-bound host/port (needed when port was 0).
        bound_host, bound_port = self._server.server_address[:2]
        # Serve in a daemon thread so start() returns immediately and the process
        # can exit even if stop() is never called.
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        # The endpoint to paste into Codex's config, including the mount prefix.
        return f"http://{bound_host}:{bound_port}{cfg.endpoint_path}"

    def stop(self) -> list[dict[str, Any]]:
        """Shut the proxy down and return the ordered list of recorded calls."""
        # Stop accepting/serving requests, then release the socket.
        if self._server is not None:
            self._server.shutdown()       # unblocks serve_forever()
            self._server.server_close()   # closes the listening socket
            self._server = None
        # Wait for the serve thread to finish (bounded, so stop() can't hang forever).
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        # Close the upstream HTTP client (frees connections).
        if self._client is not None:
            self._client.close()
            self._client = None
        # Close the JSONL file so all buffered lines are flushed to disk.
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        return self.calls  # the in-memory record of every call this session

    # Context-manager sugar: `with ProxyRecorder(cfg) as rec: ...` starts/stops.
    def __enter__(self) -> "ProxyRecorder":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    @property
    def log_path(self) -> Path | None:
        """Path of the JSONL file for the current/last session."""
        return self._log_path

    # -- request handling --------------------------------------------------

    def _target_url(self, request_path: str) -> str:
        """Map an incoming path onto the real backend, stripping our mount prefix."""
        prefix = self.config.endpoint_path  # e.g. "/v1"
        # Drop our local mount prefix so we don't duplicate it onto backend_base_url
        # (which already ends in its own version path, e.g. ".../v1").
        suffix = request_path[len(prefix):] if request_path.startswith(prefix) else request_path
        # Guarantee a leading slash before concatenation.
        if suffix and not suffix.startswith("/"):
            suffix = "/" + suffix
        # e.g. "/v1/chat/completions" -> backend_base_url + "/chat/completions".
        return self.config.backend_base_url + suffix

    def _upstream_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        """Build the headers to send upstream: client's (minus hop-by-hop) + injected."""
        headers: dict[str, str] = {}
        # Forward the client's own headers (auth, user-agent, etc.) for faithful passthrough.
        if self.config.forward_headers:
            for k, v in handler.headers.items():
                # Skip hop-by-hop headers; httpx regenerates them for the new connection.
                if k.lower() not in HOP_BY_HOP:
                    headers[k] = v
        # Add/override with configured headers (e.g. a backend API key Codex won't send).
        headers.update(self.config.inject_headers)  # injected wins (e.g. backend key)
        return headers

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        # Read the full request body (Content-Length bytes; 0 if absent).
        length = int(handler.headers.get("Content-Length", 0) or 0)
        req_body = handler.rfile.read(length) if length else b""
        # Compute the real backend URL this request maps to.
        target = self._target_url(handler.path)
        # Assign the call index at request *arrival* so concurrent calls keep
        # their true request order, not completion order.
        with self._lock:
            self._counter += 1
            index = self._counter
        started = time.perf_counter()  # start the latency clock

        resp = None               # upstream response (stays None if connect fails)
        chunks: list[bytes] = []  # buffered raw response bytes, for the log
        headers_sent = False      # whether we've already written status+headers to the client
        try:
            # Open a streaming upstream request so we can relay as bytes arrive.
            with self._client.stream(
                handler.command, target, headers=self._upstream_headers(handler),
                content=req_body,
            ) as resp:
                # Status + response headers (strip hop-by-hop; we re-frame the body).
                handler.send_response_only(resp.status_code)  # status line, no default headers
                for k, v in resp.headers.items():
                    if k.lower() not in HOP_BY_HOP:
                        handler.send_header(k, v)  # relay upstream headers verbatim
                # We stream, so advertise chunked framing instead of a fixed length.
                handler.send_header("Transfer-Encoding", "chunked")
                handler.end_headers()
                headers_sent = True  # from here on we can no longer send a clean error status

                # Relay raw bytes live while buffering for the log. iter_raw keeps
                # any content-encoding intact, so the client decodes as upstream intended.
                for chunk in resp.iter_raw():
                    if not chunk:
                        continue  # skip keep-alive/empty reads
                    chunks.append(chunk)  # buffer for the log
                    # Write one HTTP/1.1 chunk: hex length, CRLF, data, CRLF.
                    handler.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                    handler.wfile.flush()  # flush so Codex sees tokens immediately (live)
                # Terminating zero-length chunk marks the end of the body.
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()

            # Success: measure latency and record the completed call.
            latency_ms = (time.perf_counter() - started) * 1000
            self._record(index, handler, target, req_body, resp,
                         b"".join(chunks), latency_ms, error=None)
        except Exception as exc:  # upstream unreachable / mid-stream failure / client gone
            latency_ms = (time.perf_counter() - started) * 1000
            # Keep whatever partial response we captured so a crash is still diagnosable.
            self._record(index, handler, target, req_body, resp,
                         b"".join(chunks), latency_ms, error=repr(exc))
            if headers_sent:
                # Status/headers already on the wire; we can't send a clean 502.
                # Drop the connection so the client sees a truncated (failed) stream.
                handler.close_connection = True
            else:
                # Nothing sent yet, so return a proper gateway error to the client.
                try:
                    handler.send_error(502, "Bad Gateway (upstream error)")
                except Exception:
                    pass  # client may already be gone; nothing more we can do

    # -- logging -----------------------------------------------------------

    def _record(self, index, handler, target, req_body, resp, raw_body,
                latency_ms, error):
        # Decode (and decompress) the buffered response bytes into text for the log.
        decoded = _decode_body(raw_body, resp.headers if resp is not None else {})
        # SSE responses are streamed; detect by content-type to choose reassembly.
        content_type = resp.headers.get("content-type", "") if resp is not None else ""
        streamed = "text/event-stream" in content_type

        response_body: Any  # the parsed/reassembled response we store
        if not decoded:
            response_body = None               # nothing captured (e.g. connect failure)
        elif streamed:
            response_body = reassemble(decoded, handler.path)  # merge SSE -> full object
        else:
            response_body = _maybe_json(decoded)               # plain JSON (or raw text)

        # One structured record per call; this dict is both appended and written as JSONL.
        call = {
            "index": index,                                      # request order (1-based)
            "timestamp": datetime.now(timezone.utc).isoformat(), # when recorded (UTC)
            "method": handler.command,                           # HTTP verb (POST, ...)
            "path": handler.path,                                # the local path Codex hit
            "backend_url": target,                               # where we forwarded it
            "request_headers": self._redact(dict(handler.headers)),  # Codex's headers (auth redacted)
            "request_body": _maybe_json(req_body.decode("utf-8", "replace")),  # full request payload
            "response_status": resp.status_code if resp is not None else None,  # upstream status
            "response_headers": dict(resp.headers) if resp is not None else {},  # upstream headers
            "response_model": _extract_model(response_body),     # served model id (proof of backend)
            "streamed": streamed,                                # was it an SSE stream?
            "response_body": response_body,                      # reassembled/parsed response
            "response_raw": decoded if streamed else None,       # unabridged SSE text (streams only)
            "latency_ms": round(latency_ms, 2),                  # wall-clock for the call
            "error": error,                                      # repr() of any exception, else None
            "seed": self.config.seed,                            # echoed for reproducibility bookkeeping
        }
        # Append to memory and persist the line immediately (flush => crash-safe).
        with self._lock:
            self.calls.append(call)
            if self._log_file is not None:
                # ensure_ascii=False keeps unicode readable in the JSONL.
                self._log_file.write(json.dumps(call, ensure_ascii=False) + "\n")
                self._log_file.flush()

    def _redact(self, headers: dict[str, str]) -> dict[str, str]:
        # If redaction is off, return the headers as-is.
        if not self.config.redact_auth:
            return headers
        names = self.config.redact_header_names  # lower-cased set of sensitive names
        # Replace sensitive values with a placeholder; keep everything else.
        return {k: ("<redacted>" if k.lower() in names else v) for k, v in headers.items()}


def _decode_body(raw: bytes, headers) -> str:
    """Decompress per content-encoding and decode to text for the log."""
    # Look at the upstream content-encoding to know whether to decompress.
    enc = (headers.get("content-encoding", "") if headers else "").lower()
    data = raw
    try:
        if enc == "gzip":
            data = gzip.decompress(raw)
        elif enc == "deflate":
            data = zlib.decompress(raw)
        # (br/zstd aren't handled; see README limitations — they'd log as garbage.)
    except Exception:
        data = raw  # log whatever we have rather than crash
    # "replace" so undecodable bytes become U+FFFD instead of raising.
    return data.decode("utf-8", "replace")


def _maybe_json(text: str) -> Any:
    # Parse JSON when possible; otherwise keep the raw text (never lose data).
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _extract_model(body: Any) -> str | None:
    """Pull the served model id out of a response body (proof of which backend)."""
    if isinstance(body, dict):
        # Chat Completions puts it at top level; Responses nests it under "response".
        return body.get("model") or (body.get("response") or {}).get("model")
    return None
