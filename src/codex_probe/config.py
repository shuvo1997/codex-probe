"""Config schema, defaults, and validation for :class:`codex_probe.ProxyRecorder`.

The whole toolbox is driven by a single JSON-serializable ``dict``. This module
is the one place that knows the field names, their defaults, and their rules, so
validation errors are raised early with a clear message instead of blowing up
deep inside the proxy loop.
"""

# Enable modern annotation syntax (e.g. `str | None`) regardless of Python version.
from __future__ import annotations

import os  # for environment-variable expansion of ${VARS} in config values
from dataclasses import dataclass, field, asdict  # the typed config container + helpers
from typing import Any  # loose typing for the incoming raw dict

# Hop-by-hop headers must never be forwarded between the two connections
# (client<->proxy and proxy<->backend); they describe a single hop only.
# https://www.rfc-editor.org/rfc/rfc9110#section-7.6.1
HOP_BY_HOP = frozenset(
    h.lower()  # store lower-cased so lookups are case-insensitive
    for h in (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",            # must be set from the *upstream* URL, not copied
        "content-length",  # recomputed per hop (and we stream with chunked framing)
    )
)

# Default header names whose *values* are blanked in the logs (but still
# forwarded upstream unchanged). Keeps bearer tokens out of the JSONL on disk.
DEFAULT_REDACT = ("authorization", "api-key", "x-api-key", "openai-api-key")


@dataclass
class Config:
    """Validated, normalized configuration.

    Prefer :func:`load_config`; construct directly only if you already hold
    trusted values.
    """

    # --- the one required field ---
    backend_base_url: str  # real backend to forward to, incl. version path (".../v1")

    # --- networking / where the proxy listens ---
    listen_host: str = "127.0.0.1"  # loopback by default (don't expose the proxy)
    listen_port: int = 8135  # 0 => OS picks a free port, read back after start()
    endpoint_path: str = "/v1"  # local mount prefix mapped onto backend_base_url

    # --- logging ---
    log_dir: str = "./codex_probe_logs"  # directory for the per-session JSONL file
    session_name: str | None = None  # label for the per-session JSONL file (else timestamp)

    # --- header handling ---
    forward_headers: bool = True  # forward the client's own headers upstream
    inject_headers: dict[str, str] = field(default_factory=dict)  # extra upstream headers
    redact_auth: bool = True  # blank sensitive header values in the log
    redact_header_names: tuple[str, ...] = DEFAULT_REDACT  # which names to blank

    # --- misc ---
    seed: int | None = None  # echoed into logs only; never injected into requests
    connect_timeout: float = 30.0  # upstream connect timeout (read timeout is unbounded)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable echo of the config (for the logs)."""
        return asdict(self)  # dataclass -> plain dict


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` / ``$VAR`` references so secrets stay in the env."""
    return os.path.expandvars(value)  # leaves unknown vars untouched


def load_config(config: dict[str, Any]) -> Config:
    """Validate ``config`` and return a normalized :class:`Config`.

    Raises
    ------
    ValueError
        On any missing required field or out-of-range / wrong-typed value, with
        a message naming the offending field.
    """
    # The config must be a dict (JSON object), not a list/str/etc.
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config).__name__}")

    # Reject typos/unknown keys early instead of silently ignoring them.
    unknown = set(config) - {f for f in Config.__dataclass_fields__}
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")

    # --- backend_base_url: required, must be a real http(s) URL ---
    url = config.get("backend_base_url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("backend_base_url is required and must be a non-empty string")
    url = _expand_env(url).rstrip("/")  # expand env, drop trailing slash to avoid "//"
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
    path = "/" + path.strip("/") if path.strip("/") else ""  # collapse "/", strip extra slashes

    # --- inject_headers: must be a flat dict[str, str]; expand env in values ---
    inject = config.get("inject_headers", {}) or {}
    if not isinstance(inject, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in inject.items()
    ):
        raise ValueError("inject_headers must be a dict[str, str]")
    inject = {k: _expand_env(v) for k, v in inject.items()}  # e.g. "Bearer ${OPENAI_API_KEY}"

    # --- seed: optional int (bool excluded) ---
    seed = config.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed must be an int or null")

    # --- redact_header_names: list of strings ---
    redact_names = config.get("redact_header_names", DEFAULT_REDACT)
    if not all(isinstance(n, str) for n in redact_names):
        raise ValueError("redact_header_names must be a list of strings")

    # Build the normalized Config (all values validated/coerced above).
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
        redact_header_names=tuple(n.lower() for n in redact_names),  # lower-cased for matching
        seed=seed,
        connect_timeout=float(config.get("connect_timeout", 30.0)),
    )
