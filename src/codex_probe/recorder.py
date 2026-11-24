"""The recording reverse-proxy: :class:`ProxyRecorder`.

``ProxyRecorder`` listens locally on an OpenAI-compatible endpoint, forwards
every request to the configured real backend *unchanged*, relays the response
back to Codex *live* (streaming), and records the full request + reassembled
response of every call into an in-memory list and a per-session JSONL file.

Which backend is forwarded to is config only (``backend_base_url``), so the same
recorder captures both the default-OpenAI run and the swapped-in local-OSS run.
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
import time
import zlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from .config import HOP_BY_HOP, Config, load_config
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
        self.config: Config = config if isinstance(config, Config) else load_config(config)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._client: httpx.Client | None = None
        self._log_path: Path | None = None
        self._log_file = None
        self._counter = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> str:
        """Start the proxy in a background thread and return the local endpoint.

        The returned URL (``http://host:port<endpoint_path>``) is what goes in
        Codex's ``model_providers.<name>.base_url``.
        """
        if self._server is not None:
            raise RuntimeError("recorder already started")

        cfg = self.config
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        name = cfg.session_name or datetime.now().strftime("session-%Y%m%d-%H%M%S")
        self._log_path = Path(cfg.log_dir) / f"{name}.jsonl"
        self._log_file = self._log_path.open("a", encoding="utf-8")

        self._client = httpx.Client(
            timeout=httpx.Timeout(None, connect=cfg.connect_timeout),
            follow_redirects=False,
        )

        recorder = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                recorder._handle(self)

            def do_GET(self):
                recorder._handle(self)

            def do_PUT(self):
                recorder._handle(self)

            def do_DELETE(self):
                recorder._handle(self)

        class _QuietServer(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                if not isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):
                    super().handle_error(request, client_address)

        self._server = _QuietServer((cfg.listen_host, cfg.listen_port), Handler)
        bound_host, bound_port = self._server.server_address[:2]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{bound_host}:{bound_port}{cfg.endpoint_path}"

    def stop(self) -> list[dict[str, Any]]:
        """Shut the proxy down and return the ordered list of recorded calls."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        return self.calls

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
        prefix = self.config.endpoint_path
        suffix = request_path[len(prefix):] if request_path.startswith(prefix) else request_path
        if suffix and not suffix.startswith("/"):
            suffix = "/" + suffix
        return self.config.backend_base_url + suffix

    def _upstream_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        """Build the headers to send upstream: client's (minus hop-by-hop) + injected."""
        headers: dict[str, str] = {}
        if self.config.forward_headers:
            for k, v in handler.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    headers[k] = v
        headers.update(self.config.inject_headers)
        return headers

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", 0) or 0)
        req_body = handler.rfile.read(length) if length else b""
        target = self._target_url(handler.path)
        with self._lock:
            self._counter += 1
            index = self._counter
        started = time.perf_counter()

        resp = None
        chunks: list[bytes] = []
        headers_sent = False
        try:
            with self._client.stream(
                handler.command, target, headers=self._upstream_headers(handler),
                content=req_body,
            ) as resp:
                handler.send_response_only(resp.status_code)
                for k, v in resp.headers.items():
                    if k.lower() not in HOP_BY_HOP:
                        handler.send_header(k, v)
                handler.send_header("Transfer-Encoding", "chunked")
                handler.end_headers()
                headers_sent = True

                for chunk in resp.iter_raw():
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    handler.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                    handler.wfile.flush()
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()

            latency_ms = (time.perf_counter() - started) * 1000
            self._record(index, handler, target, req_body, resp,
                         b"".join(chunks), latency_ms, error=None)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            self._record(index, handler, target, req_body, resp,
                         b"".join(chunks), latency_ms, error=repr(exc))
            if headers_sent:
                handler.close_connection = True
            else:
                try:
                    handler.send_error(502, "Bad Gateway (upstream error)")
                except Exception:
                    pass

    # -- logging -----------------------------------------------------------

    def _record(self, index, handler, target, req_body, resp, raw_body,
                latency_ms, error):
        decoded = _decode_body(raw_body, resp.headers if resp is not None else {})
        content_type = resp.headers.get("content-type", "") if resp is not None else ""
        streamed = "text/event-stream" in content_type

        response_body: Any
        if not decoded:
            response_body = None
        elif streamed:
            response_body = reassemble(decoded)
        else:
            response_body = _maybe_json(decoded)

        call = {
            "index": index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": handler.command,
            "path": handler.path,
            "backend_url": target,
            "request_headers": self._redact(dict(handler.headers)),
            "request_body": _maybe_json(req_body.decode("utf-8", "replace")),
            "response_status": resp.status_code if resp is not None else None,
            "response_headers": dict(resp.headers) if resp is not None else {},
            "response_model": _extract_model(response_body),
            "streamed": streamed,
            "response_body": response_body,
            "response_raw": decoded if streamed else None,
            "latency_ms": round(latency_ms, 2),
            "error": error,
            "seed": self.config.seed,
        }
        with self._lock:
            self.calls.append(call)
            if self._log_file is not None:
                self._log_file.write(json.dumps(call, ensure_ascii=False) + "\n")
                self._log_file.flush()

    def _redact(self, headers: dict[str, str]) -> dict[str, str]:
        if not self.config.redact_auth:
            return headers
        names = self.config.redact_header_names
        return {k: ("<redacted>" if k.lower() in names else v) for k, v in headers.items()}


def _decode_body(raw: bytes, headers) -> str:
    """Decompress per content-encoding and decode to text for the log."""
    enc = (headers.get("content-encoding", "") if headers else "").lower()
    data = raw
    try:
        if enc == "gzip":
            data = gzip.decompress(raw)
        elif enc == "deflate":
            data = zlib.decompress(raw)
    except Exception:
        data = raw
    return data.decode("utf-8", "replace")


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _extract_model(body: Any) -> str | None:
    """Pull the served model id out of a response body (proof of which backend)."""
    if isinstance(body, dict):
        return body.get("model") or (body.get("response") or {}).get("model")
    return None
