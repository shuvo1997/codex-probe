"""Config schema, defaults, and validation for :class:`codex_probe.ProxyRecorder`.

The whole toolbox is driven by a single JSON-serializable ``dict``. This module
is the one place that knows the field names, their defaults, and their rules, so
validation errors are raised early with a clear message instead of blowing up
deep inside the proxy loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

HOP_BY_HOP = frozenset(
    h.lower()
    for h in (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    )
)

DEFAULT_REDACT = ("authorization", "api-key", "x-api-key", "openai-api-key")


@dataclass
class Config:
    """Validated, normalized configuration.

    Prefer :func:`load_config`; construct directly only if you already hold
    trusted values.
    """

    # --- the one required field ---
    backend_base_url: str

    # --- networking / where the proxy listens ---
    listen_host: str = "127.0.0.1"
    listen_port: int = 8135
    endpoint_path: str = "/v1"

    # --- logging ---
    log_dir: str = "./codex_probe_logs"
    session_name: str | None = None

    # --- header handling ---
    forward_headers: bool = True
    inject_headers: dict[str, str] = field(default_factory=dict)
    redact_auth: bool = True
    redact_header_names: tuple[str, ...] = DEFAULT_REDACT

    # --- misc ---
    seed: int | None = None
    connect_timeout: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable echo of the config (for the logs)."""
        return asdict(self)


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` / ``$VAR`` references so secrets stay in the env."""
    return os.path.expandvars(value)


def load_config(config: dict[str, Any]) -> Config:
    """Validate ``config`` and return a normalized :class:`Config`.

    Raises
    ------
    ValueError
        On any missing required field or out-of-range / wrong-typed value, with
        a message naming the offending field.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config).__name__}")

    unknown = set(config) - {f for f in Config.__dataclass_fields__}
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")

    # --- backend_base_url: required, must be a real http(s) URL ---
    url = config.get("backend_base_url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("backend_base_url is required and must be a non-empty string")
    url = _expand_env(url).rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("backend_base_url must start with http:// or https://")

    # --- listen_port: int in valid TCP range (bool excluded; bool is an int subclass) ---
    port = config.get("listen_port", 8135)
    if not isinstance(port, int) or isinstance(port, bool) or not (0 <= port <= 65535):
        raise ValueError("listen_port must be an int in [0, 65535]")

    # --- endpoint_path: must start with "/"; normalize to "/x" or "" ---
    path = config.get("endpoint_path", "/v1")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("endpoint_path must be a string beginning with '/'")
    path = "/" + path.strip("/") if path.strip("/") else ""

    # --- inject_headers: must be a flat dict[str, str]; expand env in values ---
    inject = config.get("inject_headers", {}) or {}
    if not isinstance(inject, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in inject.items()
    ):
        raise ValueError("inject_headers must be a dict[str, str]")
    inject = {k: _expand_env(v) for k, v in inject.items()}

    # --- seed: optional int (bool excluded) ---
    seed = config.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed must be an int or null")

    # --- redact_header_names: list of strings ---
    redact_names = config.get("redact_header_names", DEFAULT_REDACT)
    if not all(isinstance(n, str) for n in redact_names):
        raise ValueError("redact_header_names must be a list of strings")

    return Config(
        backend_base_url=url,
        listen_host=str(config.get("listen_host", "127.0.0.1")),
        listen_port=port,
        endpoint_path=path,
        log_dir=str(config.get("log_dir", "./codex_probe_logs")),
        session_name=config.get("session_name"),
        forward_headers=bool(config.get("forward_headers", True)),
        inject_headers=inject,
        redact_auth=bool(config.get("redact_auth", True)),
        redact_header_names=tuple(n.lower() for n in redact_names),
        seed=seed,
        connect_timeout=float(config.get("connect_timeout", 30.0)),
    )
