"""Runnable quickstart for CodexProbe.

This runs with no external services: it stands up a tiny mock OpenAI-compatible
backend, puts the recorder in front of it, fires one request through (standing
in for Codex), and prints the captured call plus where the JSONL log landed.

For the real thing, you replace the mock backend URL with OpenAI or a local
Ollama/vLLM server and point Codex's `model_providers.<name>.base_url` at the
`endpoint` the recorder returns — see the README.

    python examples/quickstart.py
"""

import json        # pretty-print the captured call at the end
import threading    # run the mock backend in a background thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # the mock backend server

import httpx  # used here only to stand in for Codex making a request

from codex_probe import ProxyRecorder  # the thing we're demonstrating

# A canned SSE (Server-Sent Events) stream the mock backend returns: two content
# chunks then the [DONE] sentinel — the shape a real Chat Completions stream has.
_SSE = (
    'data: {"id":"c1","model":"demo-coder","choices":[{"index":0,'
    '"delta":{"role":"assistant","content":"print(\\"hi\\")"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: [DONE]\n\n'
)


class _MockBackend(BaseHTTPRequestHandler):
    # Silence the default per-request stderr logging.
    def log_message(self, *_):
        pass

    def do_POST(self):  # noqa: N802  (http.server requires this method name)
        # Drain the request body (we don't need it for the demo).
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = _SSE.encode()  # the canned stream as bytes
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")  # mark it as an SSE stream
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)  # send the whole canned stream


def main() -> None:
    # Stand up the mock backend on a random free port, in a background thread.
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _MockBackend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    host, port = backend.server_address[:2]  # read back the chosen port

    # --- the public API ---------------------------------------------------
    recorder = ProxyRecorder(
        {
            "backend_base_url": f"http://{host}:{port}/v1",  # swap for OpenAI / Ollama
            "listen_port": 0,  # pick any free port
            "log_dir": "./codex_probe_logs",
            "session_name": "quickstart",
        }
    )
    endpoint = recorder.start()  # start proxy; get the URL Codex would use
    print(f"Recorder endpoint (put this in Codex config.toml base_url): {endpoint}\n")

    # Stand in for Codex making a streamed chat-completions call.
    with httpx.stream("POST", f"{endpoint}/chat/completions",
                      json={"model": "demo-coder",
                            "messages": [{"role": "user", "content": "say hi in python"}],
                            "stream": True}) as r:
        streamed = "".join(r.iter_text())  # consume the live stream as the client would
    print("What Codex received (live stream), first line:")
    print("   " + streamed.splitlines()[0] + "\n")

    calls = recorder.stop()  # stop proxy; get the recorded calls
    backend.shutdown()       # stop the mock backend

    call = calls[0]  # the single call we made above
    print("What CodexProbe recorded:")
    print(f"   backend served : {call['response_model']}  via  {call['backend_url']}")
    print(f"   latency_ms     : {call['latency_ms']}")
    # The streamed chunks were reassembled into one message content.
    print(f"   reassembled    : "
          f"{call['response_body']['choices'][0]['message']['content']!r}")
    print(f"\nFull call as JSON written to: {recorder.log_path}")
    print(json.dumps(call, indent=2)[:600] + " ...")  # first 600 chars of the record


# Standard script entry point.
if __name__ == "__main__":
    main()
