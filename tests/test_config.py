"""Config validation tests."""

import pytest

from codex_probe.config import load_config


def test_minimal_config_defaults():
    cfg = load_config({"backend_base_url": "https://api.openai.com/v1/"})
    assert cfg.backend_base_url == "https://api.openai.com/v1"
    assert cfg.listen_host == "127.0.0.1"
    assert cfg.endpoint_path == "/v1"
    assert cfg.redact_auth is True


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({}, "backend_base_url is required"),
        ({"backend_base_url": "ftp://x"}, "http://"),
        ({"backend_base_url": "http://x", "listen_port": 70000}, "listen_port"),
        ({"backend_base_url": "http://x", "endpoint_path": "v1"}, "endpoint_path"),
        ({"backend_base_url": "http://x", "seed": "nope"}, "seed"),
        ({"backend_base_url": "http://x", "bogus": 1}, "unknown config field"),
        ({"backend_base_url": "http://x", "inject_headers": {"k": 1}}, "inject_headers"),
    ],
)
def test_invalid_config_raises(bad, msg):
    with pytest.raises(ValueError, match=msg):
        load_config(bad)


def test_env_expansion_in_inject_headers(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-secret")
    cfg = load_config(
        {"backend_base_url": "http://x", "inject_headers": {"Authorization": "Bearer ${MY_KEY}"}}
    )
    assert cfg.inject_headers["Authorization"] == "Bearer sk-secret"
