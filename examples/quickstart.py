"""Runnable quickstart for CodexProbe — does all the runs from one file.

Pick what to run with --backend:

    # (0) OFFLINE DEMO (default) — no API key, no Codex, no model download.
    #     Stands up a mock backend and fires one streamed call through the
    #     recorder, to show the record -> live-relay -> reassemble machinery.
    python examples/quickstart.py

    # (a) REAL run against the default OpenAI backend (needs OPENAI_API_KEY).
    python examples/quickstart.py --backend openai \
        "Read src/codex_probe/recorder.py and explain what ProxyRecorder does."

    # (b) REAL run against a swapped-in local Qwen via Ollama (config-only swap;
    #     needs `ollama serve` + `ollama pull qwen3:8b`). Same recorder, same logs.
    python examples/quickstart.py --backend ollama \
        "Read src/codex_probe/recorder.py and explain what ProxyRecorder does."

For (a)/(b) you need the `codex` CLI on PATH (`npm i -g @openai/codex` or
`brew install codex`). Each run writes an ordered JSONL log under
./codex_probe_logs/. A pre-captured real run is committed at
examples/sample_capture_codex_qwen3.jsonl. See the README for the full walkthrough.
"""

import argparse     # parse --backend / --cd / --sandbox / prompt
import json         # pretty-print the captured call in the demo
import os           # build the child env (CODEX_HOME) and join paths
import shutil       # shutil.which() to check the codex CLI is installed
import subprocess   # run the real `codex exec` process
import sys          # propagate the exit code
import tempfile     # throwaway CODEX_HOME so the user's real config is untouched
import threading    # run the mock backend in a background thread (demo only)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # mock backend

import httpx        # stands in for Codex in the offline demo

from codex_probe import ProxyRecorder  # the thing we're demonstrating

# Real-backend presets. Only backend_base_url/inject_headers/model differ —
# everything else about the recorder (and the logs) is identical across backends.
BACKENDS = {
    "openai": {
        "backend_base_url": "https://api.openai.com/v1",
        "inject_headers": {"Authorization": "Bearer ${OPENAI_API_KEY}"},
        "model": "gpt-6.1-sol",
    },
    "ollama": {
        "backend_base_url": "http://127.0.0.1:11434/v1",
        "inject_headers": {},
        "model": "qwen3:8b",
    },
}

# A self-demonstrating default task for the real backends: Codex reads this very
# repo and explains the recorder, so the run exercises a real tool-calling loop.
DEFAULT_PROMPT = (
    "Read src/codex_probe/recorder.py and explain in 5 bullet points what "
    "ProxyRecorder does and how it streams responses."
)


# --------------------------------------------------------------------------- #
# (0) Offline demo: a canned mock backend + one streamed call through the proxy #
# --------------------------------------------------------------------------- #

# A canned SSE stream the mock returns: one content chunk then [DONE] — the shape
# of a real Chat Completions stream.
_SSE = (
    'data: {"id":"c1","model":"demo-coder","choices":[{"index":0,'
    '"delta":{"role":"assistant","content":"print(\\"hi\\")"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: [DONE]\n\n'
)


class _MockBackend(BaseHTTPRequestHandler):
    def log_message(self, *_):  # silence per-request stderr logging
        pass

    def do_POST(self):  # noqa: N802  (http.server requires this method name)
        self.rfile.read(int(self.headers.get("Content-Length", 0)))  # drain body
        body = _SSE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")  # it's an SSE stream
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_demo() -> int:
    """Offline: record -> live-relay -> reassemble, no external services."""
    # Stand up the mock backend on a random free port, in a background thread.
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _MockBackend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    host, port = backend.server_address[:2]

    recorder = ProxyRecorder(
        {
            "backend_base_url": f"http://{host}:{port}/v1",  # swap for OpenAI/Ollama
            "listen_port": 0,
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
        streamed = "".join(r.iter_text())  # consume the live stream as a client would
    print("What Codex received (live stream), first line:")
    print("   " + streamed.splitlines()[0] + "\n")

    calls = recorder.stop()  # stop proxy; get the recorded calls
    backend.shutdown()       # stop the mock backend

    call = calls[0]
    print("What CodexProbe recorded:")
    print(f"   backend served : {call['response_model']}  via  {call['backend_url']}")
    print(f"   latency_ms     : {call['latency_ms']}")
    print(f"   reassembled    : "  # streamed chunks reassembled into one message
          f"{call['response_body']['choices'][0]['message']['content']!r}")
    print(f"\nFull call as JSON written to: {recorder.log_path}")
    print(json.dumps(call, indent=2)[:600] + " ...")
    return 0


# --------------------------------------------------------------------------- #
# (a)/(b) Real run: drive the actual `codex` CLI through the recorder           #
# --------------------------------------------------------------------------- #

def run_codex(backend: str, cd: str, sandbox: str, prompt: str) -> int:
    """Drive the real Codex CLI against `backend`, recording every LLM call."""
    if not shutil.which("codex"):
        print("codex CLI not found. Install: npm i -g @openai/codex  (or brew install codex)")
        return 1

    preset = BACKENDS[backend]
    recorder = ProxyRecorder(
        {
            "backend_base_url": preset["backend_base_url"],  # where to forward
            "inject_headers": preset["inject_headers"],      # e.g. OpenAI key (none for Ollama)
            "listen_port": 0,                                # OS picks a free port
            "log_dir": "./codex_probe_logs",
            "session_name": f"codex-{backend}",              # one file per backend
        }
    )
    endpoint = recorder.start()
    print(f"[probe] recording {backend} backend; endpoint = {endpoint}")

    # Throwaway CODEX_HOME so we never touch the user's real ~/.codex config.
    codex_home = tempfile.mkdtemp(prefix="codex_home_")
    with open(os.path.join(codex_home, "config.toml"), "w") as f:
        f.write(
            f'model_provider = "probe"\n'            # make "probe" the active provider
            f'model = "{preset["model"]}"\n\n'       # the model id to request
            f'[model_providers.probe]\n'
            f'name = "CodexProbe"\n'
            f'base_url = "{endpoint}"\n'             # <-- Codex -> recorder
            f'wire_api = "responses"\n'             # current Codex only supports Responses
        )

    try:
        subprocess.run(
            ["codex", "exec", "--skip-git-repo-check",       # allow running outside a git repo
             "-s", sandbox, "-C", cd, prompt],              # sandbox, workdir, task
            env={**os.environ, "CODEX_HOME": codex_home},   # point Codex at the throwaway config
            check=False,
        )
    finally:
        calls = recorder.stop()  # always stop the recorder, even if codex errored

    print(f"\n[probe] recorded {len(calls)} LLM call(s) -> {recorder.log_path}")
    for c in calls:
        print(f"  #{c['index']} {c['response_status']} "
              f"model={c['response_model']} via {c['backend_url']} ({c['latency_ms']} ms)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CodexProbe quickstart — runs the offline demo "
                                             "or drives the real Codex CLI against a backend.")
    ap.add_argument("--backend", choices=["demo", *BACKENDS], default="demo",
                    help="demo (offline, default), or openai/ollama to drive the real Codex CLI")
    ap.add_argument("--cd", default=".", help="working root for the Codex task (real runs)")
    ap.add_argument("--sandbox", default="workspace-write",
                    choices=["read-only", "workspace-write", "danger-full-access"])
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT,
                    help="task prompt for the real Codex run (ignored by the demo)")
    args = ap.parse_args()

    if args.backend == "demo":
        return run_demo()
    return run_codex(args.backend, args.cd, args.sandbox, args.prompt)


if __name__ == "__main__":
    sys.exit(main())
