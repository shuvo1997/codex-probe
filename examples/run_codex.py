"""Drive the *real* Codex CLI through CodexProbe, for either backend.

This delivers the end-to-end walkthrough: start the recorder, write a throwaway
``CODEX_HOME/config.toml`` whose provider ``base_url`` points at the recorder,
run ``codex exec <prompt>``, then stop and print the per-session JSONL log.

The backend is config only — pick it with ``--backend``:

    # (a) default OpenAI backend (needs OPENAI_API_KEY in your env)
    python examples/run_codex.py --backend openai "add a docstring to utils.py"

    # (b) swapped-in local Qwen via Ollama (needs `ollama serve` + the model)
    python examples/run_codex.py --backend ollama "add a docstring to utils.py"

Requires the `codex` CLI on PATH (`brew install codex` or
`npm i -g @openai/codex`). If it's absent, the script tells you and exits.
"""

import argparse     # parse --backend / --cd / --sandbox / prompt
import os            # build the child env (CODEX_HOME) and join paths
import shutil        # shutil.which() to check the codex CLI is installed
import subprocess    # run the real `codex exec` process
import sys           # propagate the exit code
import tempfile      # create a throwaway CODEX_HOME so the user's config is untouched

from codex_probe import ProxyRecorder  # the recorder we put in front of the backend

# Backend presets. Only `backend_base_url`/`inject_headers` differ — everything
# else about the recorder (and the resulting logs) is identical across backends.
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


def main() -> int:
    # --- parse CLI args ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="ollama")  # which preset to forward to
    ap.add_argument("--cd", default=".", help="working root for the Codex task")  # codex -C
    ap.add_argument("--sandbox", default="workspace-write",  # codex sandbox policy
                    choices=["read-only", "workspace-write", "danger-full-access"])
    ap.add_argument("prompt", help="task prompt for `codex exec`")  # the coding task
    args = ap.parse_args()

    # Fail fast with a helpful message if the codex binary isn't installed.
    if not shutil.which("codex"):
        print("codex CLI not found. Install: brew install codex  (or npm i -g @openai/codex)")
        return 1

    preset = BACKENDS[args.backend]  # look up base_url/headers/model for the chosen backend
    # Start the recorder in front of the real backend (backend differs only by preset).
    recorder = ProxyRecorder(
        {
            "backend_base_url": preset["backend_base_url"],  # where to forward
            "inject_headers": preset["inject_headers"],      # e.g. OpenAI key (none for Ollama)
            "listen_port": 0,                                # let the OS pick a free port
            "log_dir": "./codex_probe_logs",                 # where the JSONL lands
            "session_name": f"codex-{args.backend}",         # one file per backend run
        }
    )
    endpoint = recorder.start()  # the local URL Codex will talk to
    print(f"[probe] recording {args.backend} backend; endpoint = {endpoint}")

    # Write a throwaway CODEX_HOME so we never touch the user's real ~/.codex config.
    codex_home = tempfile.mkdtemp(prefix="codex_home_")
    with open(os.path.join(codex_home, "config.toml"), "w") as f:
        # A minimal Codex config: a custom provider whose base_url is our recorder.
        f.write(
            f'model_provider = "probe"\n'            # make "probe" the active provider
            f'model = "{preset["model"]}"\n\n'       # the model id to request
            f'[model_providers.probe]\n'
            f'name = "CodexProbe"\n'
            f'base_url = "{endpoint}"\n'             # <-- Codex -> recorder
            f'wire_api = "responses"\n'             # current Codex only supports Responses
        )

    try:
        # Run the real Codex non-interactively against our provider.
        subprocess.run(
            ["codex", "exec", "--skip-git-repo-check",   # allow running outside a git repo
             "-s", args.sandbox, "-C", args.cd, args.prompt],  # sandbox, workdir, task
            env={**os.environ, "CODEX_HOME": codex_home},  # point Codex at the throwaway config
            check=False,  # don't raise on non-zero; we still want to stop + show the log
        )
    finally:
        calls = recorder.stop()  # always stop the recorder, even if codex errored

    # Summarize the captured calls (proof of backend + timing).
    print(f"\n[probe] recorded {len(calls)} LLM call(s) -> {recorder.log_path}")
    for c in calls:
        print(f"  #{c['index']} {c['response_status']} "
              f"model={c['response_model']} via {c['backend_url']} ({c['latency_ms']} ms)")
    return 0


# Standard entry point: run main() and exit with its return code.
if __name__ == "__main__":
    sys.exit(main())
