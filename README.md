# CodexProbe

A small, local **recording reverse-proxy** for OpenAI's [Codex CLI](https://github.com/openai/codex).

It sits between Codex and its real LLM backend at the `model_providers.<name>.base_url`
extension point, and:

1. **Records every LLM call** Codex makes — the full request (system/developer
   instructions, the entire running message + tool history, tool schemas) and
   the full response (message content and/or tool calls, with streamed responses
   reassembled) — into an ordered in-memory list and a per-session JSONL file.
2. **Swaps the backend by config alone.** Point `backend_base_url` at OpenAI,
   or at a self-hosted OSS model (Ollama / vLLM serving e.g. `Qwen2.5-Coder`),
   with no code change. The same recorder captures the "before" and "after" runs.

It works as a proxy at the `base_url` extension point, so it needs **no patch to
Codex's Rust source**.

## Try it in 2 minutes (for reviewers)

No API key, no Codex install, no model download needed for the first two steps —
they run entirely offline against a built-in mock backend.

```bash
# 1. clone and install (Python 3.9+)
git clone https://github.com/shuvo1997/codex-probe.git
cd codex-probe
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. run the test suite — proves correctness (19 tests, ~4s, offline)
pytest -q

# 3. see the full record -> live-relay -> reassemble flow end to end (offline)
python examples/quickstart.py
```

Step 3 starts the proxy, sends a streaming call through it (a mock backend stands
in for Codex's LLM), and prints the captured JSONL record — so you can see exactly
what the recorder logs, and that the stream was relayed live then reassembled,
without any external service.

**To see it in front of the real Codex CLI** (optional — needs the `codex` binary
and either an `OPENAI_API_KEY` or a local Ollama):

```bash
# against OpenAI — export your real key first (get one at
# https://platform.openai.com/api-keys). Never hard-code or commit it.
export OPENAI_API_KEY="sk-your-real-key-here"
python examples/quickstart.py --backend openai \
  --cd /tmp/work "Create reverse.py with reverse(s) and print reverse('hello')."

# against a self-hosted model (config-only swap — same recorder, same logs; no key)
python examples/quickstart.py --backend ollama \
  --cd /tmp/work "Create reverse.py with reverse(s) and print reverse('hello')."
```

Either way it writes `codex_probe_logs/codex-<backend>.jsonl` — one ordered line
per LLM call. **If you'd rather just read a real result without installing
anything, open `examples/sample_capture_codex_qwen3.jsonl`** — the two real LLM
calls from Codex driving local `qwen3:8b` end to end (a working tool call, then
the final answer). The full captured results (including the OSS tool-calling
findings) are in [End-to-end run](#end-to-end-run-captured-results-local-qwen).

## Install

```bash
pip install -e .
# for the tests:
pip install -e ".[dev]"
```

Runtime dependency: `httpx` only. The proxy server itself is Python stdlib
(`http.server`).

## Usage

```python
from codex_probe import ProxyRecorder

recorder = ProxyRecorder(config)
endpoint = recorder.start()   # e.g. "http://127.0.0.1:8135/v1"
# point model_providers.<name>.base_url in Codex's config.toml at `endpoint`,
# then run Codex normally (e.g. `codex exec "<task prompt>"`)
calls = recorder.stop()       # list[dict], one entry per LLM call, in order
```

`ProxyRecorder` is also a context manager (`with ProxyRecorder(cfg) as rec: ...`).

Examples included — note which are a *demo* vs a *real captured result*:

- `examples/quickstart.py` — one script, three modes via `--backend`:
  - *(default, no args)* **offline demo** — a built-in mock backend + one call
    through the recorder, to show the record → relay → reassemble flow with no
    external service. Not a real LLM result.
  - `--backend openai` / `--backend ollama` — drive the **real** `codex` CLI
    through the recorder against that backend (the config-only swap); writes a
    throwaway `CODEX_HOME` so your own Codex config is untouched. **Real results.**
- `examples/sample_capture_codex_qwen3.jsonl` — a **real captured run**: the two
  LLM calls from Codex driving local `qwen3:8b` on the task *"Explain what
  ProxyRecorder does."* Call 1 is a working `exec_command` tool call; call 2 is
  Codex's final answer after reading the file. Read this to see a genuine
  agentic loop end to end, no setup required.
- `examples/sample_capture.jsonl` — a single real captured call from the older
  `qwen2.5-coder:7b` run (shows the malformed tool-call divergence; see below).

Both `.jsonl` samples are auth-free (Ollama needs no key).

```bash
# offline demo (mock backend)
python examples/quickstart.py

# real run against local Qwen — reads an actual file in this repo and explains it.
# --cd defaults to this repo root. Logs land in codex_probe_logs/codex-ollama.jsonl.
python examples/quickstart.py --backend ollama \
  "Read src/codex_probe/recorder.py and explain in 5 bullet points what ProxyRecorder does and how it streams responses."

# same task, OpenAI backend — the config-only swap. Export your real key first
# (https://platform.openai.com/api-keys); never hard-code or commit it.
export OPENAI_API_KEY="sk-your-real-key-here"
python examples/quickstart.py --backend openai \
  "Read src/codex_probe/recorder.py and explain in 5 bullet points what ProxyRecorder does and how it streams responses."
```

## What a captured call looks like

Each entry in `calls` (and each JSONL line) has this shape:

```json
{
  "index": 1,
  "timestamp": "2026-11-03T22:19:06+00:00",
  "method": "POST",
  "path": "/v1/chat/completions",
  "backend_url": "http://127.0.0.1:11434/v1/chat/completions",
  "request_headers": {"Authorization": "<redacted>", "...": "..."},
  "request_body": { "model": "...", "messages": [...], "tools": [...] },
  "response_status": 200,
  "response_headers": {"content-type": "text/event-stream"},
  "response_model": "qwen2.5-coder:7b",
  "streamed": true,
  "response_body": { "choices": [{"message": {"role": "assistant", "content": "...", "tool_calls": [...]}, "finish_reason": "stop"}] },
  "response_raw": "data: {...}\n\ndata: [DONE]\n\n",
  "latency_ms": 812.4,
  "error": null,
  "seed": 1234
}
```

- `request_body` is the **complete** payload Codex sent (full history + tool schema).
- `response_body` is the response reassembled into the non-streaming shape.
- `response_raw` keeps the **unabridged** SSE text when the call was streamed.
- `response_model` and `backend_url` are your proof of *which* backend served the
  call (local Qwen vs. OpenAI).

## Codex config: pointing Codex at the recorder

Put provider settings in the **user-level** `~/.codex/config.toml` (project-local
`.codex/config.toml` ignores `model_provider`/`model_providers`). To avoid
touching your real config, use a throwaway `CODEX_HOME`:

```bash
export CODEX_HOME="$(mktemp -d)"
cat > "$CODEX_HOME/config.toml" <<'TOML'
model_provider = "probe"
model = "qwen3:8b"

[model_providers.probe]
name = "CodexProbe"
base_url = "http://127.0.0.1:8135/v1"   # the recorder's endpoint
wire_api = "responses"                   # the only value current Codex accepts
TOML

codex exec "write a Python function that reverses a string"
```

> **Version note (verified against Codex `main`, 2026-11-04).** Codex **removed
> the Chat Completions wire API**: the `WireApi` enum in
> `codex-rs/model-provider-info/src/lib.rs` now has a single variant,
> `Responses` (the default), and `wire_api = "chat"` is rejected with an explicit
> "chat wire API removed" error. So current Codex always POSTs to
> `/v1/responses`. Older builds (and the task text) also supported
> `wire_api = "chat"` → `/v1/chat/completions`. **CodexProbe handles both**: it
> dispatches reassembly by request *path*, so whichever your Codex build speaks,
> the recorder relays and reassembles it correctly. Your OSS backend must expose
> the matching endpoint — see below.

### Responses vs. Chat Completions wire formats

- **Responses API** (`POST /responses`): OpenAI's newer stateful protocol. Its
  SSE stream is a sequence of typed events, and the terminal
  `response.completed` event already carries the full `response` object — so
  CodexProbe takes it directly.
- **Chat Completions** (`POST /chat/completions`): the long-standing protocol
  that local OSS servers (Ollama, vLLM) speak. Its SSE stream is many
  `chat.completion.chunk` events with `choices[].delta`; CodexProbe merges the
  content and tool-call argument fragments back together.

## Serving a local Qwen model (the backend swap)

### Ollama (recommended on macOS / Apple Silicon)

vLLM needs an NVIDIA GPU, so on a Mac use Ollama. Current Ollama exposes
`POST /v1/responses` (non-stateful), which is exactly the endpoint current Codex
uses — so the swap is a true passthrough, no translation needed.

```bash
# 1. install Ollama (macOS). Needs 0.40.14+ — see the note below.
brew install ollama            # or download from https://ollama.com/download
ollama --version               # confirm >= 0.40.14

# 2. start the server (leave this running in its own terminal)
OLLAMA_CONTEXT_LENGTH=32768 ollama serve   # server on :11434, large context (see note)

# 3. pull the model (in another terminal)
ollama pull qwen3:8b           # ~5.2 GB; drives Codex end-to-end
```

You also need the Codex CLI itself for the real run: `npm i -g @openai/codex`
(or `brew install codex`).

> **Use Ollama 0.40.14+ and `qwen3:8b` for working tool-calling.** The
> `qwen2.5-coder` variants emit text-embedded tool calls Codex can't act on, and
> Ollama < 0.40 sent `qwen3` function calls with an empty `name` (rejected by
> Codex). On 0.40.14+ with `qwen3:8b`, Codex runs end-to-end. See the captured
> runs below for all three cases.

Then set `backend_base_url` to `http://127.0.0.1:11434/v1` and run the recorder.
Codex talks to the recorder; the recorder forwards to Ollama.

> **Expected divergence.** Ollama's Responses support is **non-stateful** (no
> `previous_response_id` / `conversation`). If Codex relies on stateful Responses
> features, those calls will fail at Ollama — and the failure will be visible in
> your recorded logs (non-200 `response_status` + `error`/body). Per the task,
> documenting exactly where the open model diverges is an accepted outcome, and
> these logs are the instrument for it.

**Approximate hardware:** `qwen3:8b` needs ~5.2 GB and runs on a 16 GB
Apple-Silicon Mac (tight but usable alongside Codex). For more headroom a
smaller variant trades tool-calling reliability; 14b/32b want >16 GB.

> **Context-window gotcha.** Ollama's default context is small (2–4k tokens).
> Codex's instructions + tool schemas easily exceed that and get **silently
> truncated**, which makes the model look broken. Raise it on the *server*
> (`OLLAMA_CONTEXT_LENGTH`, or a Modelfile `num_ctx`) — not in the request, which
> would break faithful passthrough.

### vLLM (Linux + NVIDIA GPU)

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 8000
# backend_base_url = http://127.0.0.1:8000/v1
```

### Verifying the swap actually happened

After a run, check the log:

```bash
python - <<'PY'
import json
for line in open("codex_probe_logs/<session>.jsonl"):
    c = json.loads(line)
    print(c["index"], c["backend_url"], c["response_model"])
PY
```

`backend_url` pointing at `:11434` and `response_model` reading `qwen2.5-coder:7b`
are your proof the calls hit local Qwen and not OpenAI.

## How streaming works without changing Codex's behavior

For every response the proxy:

1. Sends the upstream status and headers straight back, dropping **hop-by-hop**
   headers (`Connection`, `Transfer-Encoding`, `Content-Length`, …) since those
   describe a single connection hop and must not be copied across the proxy.
2. Relays the body with HTTP/1.1 **chunked** framing, writing and flushing each
   upstream chunk the instant it arrives — so the Codex CLI stays responsive and
   sees tokens live, exactly as if it were talking to the backend directly.
3. Relays the raw bytes with `iter_raw()`, leaving any `content-encoding` intact
   so the client decodes the stream exactly as the backend intended.
4. **Reassembles once, after the stream closes**, over the full buffered text —
   never per chunk — because a single SSE event can be split across chunk
   boundaries. The request is never modified; `seed` is logged, not injected.

Upstream 4xx/5xx are relayed unchanged and logged; an unreachable backend returns
`502` and is still logged with the error.

## Tests

```bash
pytest -q
```

Covers: config validation; a real-socket passthrough round-trip against a mock
OpenAI-compatible backend (non-streaming bytes unchanged, auth redacted in the
log); live SSE relay + reassembly; tool-call arguments split across fragments and
events split across chunk boundaries; log-schema + JSONL file; and the 502 path.

## End-to-end run: captured results (local Qwen)

Run on an Apple M2 Pro (16 GB), Codex CLI 0.166.0, model store on an external SSD.
The recorder made the OSS tool-calling path fully visible — and the captured logs
walked us through three distinct outcomes as we changed model and Ollama version.

**1. `qwen2.5-coder:7b` / `14b` on Ollama 0.16.3 — malformed tool call.**
The coder variants did *not* return a Responses `function_call` item. They emitted
the tool call as **plain-text JSON inside the assistant message**:

```json
{"name": "exec_command", "arguments": {"cmd": "echo '...' > reverse.py"}}
```

Codex only acts on structured `function_call` items, so nothing ran. Reproduced on
both 7b and 14b (model-variant limitation, not the proxy). Codex also logged
`Model metadata for qwen2.5-coder:7b not found`.

**2. `qwen3:8b` on Ollama 0.16.3 — right shape, empty name.**
qwen3 *does* emit a structured `function_call` — but the old Ollama `/v1/responses`
shim sent it with an **empty `name`** (`"name": ""`) while the arguments were
correct. Codex's router rejected every call with `error=unsupported call:` (7–8
times per run) and fell back to guessing. Confirmed in `response_raw` that Ollama,
not the proxy, produced the empty name (a known class of bug, cf. LiteLLM #3333).

**3. `qwen3:8b` on Ollama 0.40.14 — works end to end.** ✅
Upgrading Ollama (`brew upgrade ollama`, 0.16.3 → 0.40.14) fixed the empty-name
bug. qwen3:8b now emits a well-formed call that Codex accepts and executes:

```
$ python examples/quickstart.py --backend ollama --cd /tmp/work \
    "Explain what ProxyRecorder does."
[probe] recorded 2 LLM call(s) -> codex_probe_logs/codex-ollama.jsonl
  #1 200 model=qwen3:8b via http://127.0.0.1:11434/v1/responses (42598 ms)
  #2 200 model=qwen3:8b via http://127.0.0.1:11434/v1/responses (34393 ms)
```

Reassembled log:
- call #1 → `reasoning` + `function_call name='exec_command' call_id='call_o8y4y9ob'`
  with valid args — **accepted, no `unsupported call` error.**
- call #2 → final `message`, grounded in the file contents Codex fed back.

What every captured call contains (the normally-opaque part):
- `backend_url = http://127.0.0.1:11434/v1/responses`, `response_model = qwen3:8b`
  — **proof the call hit local Qwen, not OpenAI.**
- `request_body` holds Codex's full outgoing payload: `instructions`, the growing
  `input` list, and **9 tool schemas** (`exec_command`, `write_stdin`,
  `request_user_input`, …), plus `tool_choice`, `store`, `prompt_cache_key`, etc.
- `response_raw` keeps the unabridged SSE stream; `response_body` is it reassembled.

**Takeaways.**
- Two different "OSS tool-calling diverges" cases, both diagnosed *only* because
  the recorder preserves the raw wire: a model-side one (coder variant emits text
  JSON) and a server-side one (old Ollama drops the function name).
- qwen3:8b over-reasons on trivially-guessable prompts (it answered "what does
  slugify do" from general knowledge without reading the file). Give it a prompt
  that genuinely needs the file and it calls the tool correctly. Bumping reasoning
  effort or a larger variant (14b/32b) reduces this.
- The config-only backend swap (OpenAI ↔ self-hosted Qwen) is proven working in
  both directions, each captured as ordered JSONL.

## Honest limitations

- OSS tool-calling is the rough edge (see the captured runs above): it depends on
  both the model variant and the Ollama version. `qwen2.5-coder` emits text-embedded
  tool calls; old Ollama dropped the function name; `qwen3:8b` on Ollama 0.40.14+
  works. The logs are the diagnostic for exactly where and how any of these break.
- The live run isn't exercised in CI (it needs a running Ollama and a `codex`
  binary); CI covers the proxy via the mock-backend tests.
- The proxy assumes an OpenAI-compatible backend (Responses or Chat Completions).
- Loopback only by design; it forwards real auth upstream, so don't expose it.
