"""Config validation tests."""

import pytest  # for raises() and parametrize()

from codex_probe.config import load_config


def test_minimal_config_defaults():
    # Only backend_base_url is required; everything else takes its default.
    cfg = load_config({"backend_base_url": "https://api.openai.com/v1/"})
    assert cfg.backend_base_url == "https://api.openai.com/v1"  # trailing slash trimmed
    assert cfg.listen_host == "127.0.0.1"   # loopback default
    assert cfg.endpoint_path == "/v1"       # default mount prefix
    assert cfg.redact_auth is True          # redaction on by default


@pytest.mark.parametrize(
    # Each case: a bad config dict, and a substring expected in the error message.
    "bad, msg",
    [
        ({}, "backend_base_url is required"),                               # missing required
        ({"backend_base_url": "ftp://x"}, "http://"),                       # wrong scheme
        ({"backend_base_url": "http://x", "listen_port": 70000}, "listen_port"),  # port out of range
        ({"backend_base_url": "http://x", "endpoint_path": "v1"}, "endpoint_path"),  # no leading /
        ({"backend_base_url": "http://x", "seed": "nope"}, "seed"),         # seed not an int
        ({"backend_base_url": "http://x", "bogus": 1}, "unknown config field"),  # typo/unknown key
        ({"backend_base_url": "http://x", "inject_headers": {"k": 1}}, "inject_headers"),  # non-str value
    ],
)
def test_invalid_config_raises(bad, msg):
    # Every invalid config should raise ValueError naming the offending field.
    with pytest.raises(ValueError, match=msg):
        load_config(bad)


def test_env_expansion_in_inject_headers(monkeypatch):
    # ${VAR} in inject_headers values is expanded from the environment, so the
    # raw secret lives in the env, not in the config dict.
    monkeypatch.setenv("MY_KEY", "sk-secret")
    cfg = load_config(
        {"backend_base_url": "http://x", "inject_headers": {"Authorization": "Bearer ${MY_KEY}"}}
    )
    assert cfg.inject_headers["Authorization"] == "Bearer sk-secret"
