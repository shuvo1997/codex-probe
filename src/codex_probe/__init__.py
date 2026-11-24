"""CodexProbe: a recording reverse-proxy + backend swap for the Codex CLI.

Sits at a ``model_providers.<name>.base_url`` extension point, forwards every
LLM call unchanged to a real backend, records the full request/response, and —
by config alone — lets you swap that backend to a self-hosted OSS model.
"""

from .config import Config, load_config
from .recorder import ProxyRecorder

__all__ = ["ProxyRecorder", "Config", "load_config"]
__version__ = "0.1.0"
