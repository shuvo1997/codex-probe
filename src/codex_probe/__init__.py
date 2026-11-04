"""CodexProbe: a recording reverse-proxy + backend swap for the Codex CLI.

Sits at a ``model_providers.<name>.base_url`` extension point, forwards every
LLM call unchanged to a real backend, records the full request/response, and —
by config alone — lets you swap that backend to a self-hosted OSS model.
"""

# Re-export the public API so callers write `from codex_probe import ...`
# instead of reaching into submodules.
from .config import Config, load_config      # config dataclass + validator
from .recorder import ProxyRecorder          # the main class (start/stop/calls)

# Names considered public (what `from codex_probe import *` would expose, and a
# signal to readers of the supported surface).
__all__ = ["ProxyRecorder", "Config", "load_config"]
# Package version (kept in sync with pyproject.toml).
__version__ = "0.1.0"
