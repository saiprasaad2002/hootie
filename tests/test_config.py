"""Config loading, env interpolation, and credential hygiene."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pyqa.config import Config, EndpointConfig, expand_env
from pyqa.errors import ConfigError


def test_env_var_is_expanded(monkeypatch):
    monkeypatch.setenv("PYQA_TEST_KEY", "sk-abc")
    assert expand_env("${PYQA_TEST_KEY}") == "sk-abc"


def test_default_is_used_when_unset(monkeypatch):
    monkeypatch.delenv("PYQA_ABSENT", raising=False)
    assert expand_env("${PYQA_ABSENT:-fallback}") == "fallback"


def test_missing_var_without_default_fails_loudly(monkeypatch):
    """Better to fail at load than to send an empty key and get a puzzling 401."""
    monkeypatch.delenv("PYQA_ABSENT", raising=False)
    with pytest.raises(ConfigError, match="not set"):
        expand_env("${PYQA_ABSENT}")


def test_api_key_is_not_exposed_in_repr():
    endpoint = EndpointConfig(model="m", api_key="super-secret")
    assert "super-secret" not in repr(endpoint)
    assert endpoint.api_key.get_secret_value() == "super-secret"


def test_threshold_must_be_strictly_between_zero_and_one():
    """Chonkie rejects the endpoints with an unhelpful message; we catch it first."""
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValidationError):
            Config.model_validate({"chunk": {"threshold": bad}})
    assert Config.model_validate({"chunk": {"threshold": 0.5}}).chunk.threshold == 0.5


def test_grounding_falls_back_to_the_generation_endpoint():
    cfg = Config.model_validate(
        {"generate": {"endpoint": {"model": "gen"}}, "ground": {"enabled": True}}
    )
    assert cfg.ground.endpoint is not None
    assert cfg.ground.endpoint.model == "gen"


def test_unknown_keys_are_rejected():
    """A typo in a config file should be reported, not silently ignored."""
    with pytest.raises(ValidationError):
        Config.model_validate({"chunk": {"chunk_sizeee": 512}})


def test_example_config_loads(monkeypatch, tmp_path):
    monkeypatch.setenv("PYQA_API_KEY", "k")
    monkeypatch.setenv("PYQA_VISION_KEY", "k")
    cfg = Config.load("pyqa.example.toml")
    assert cfg.vision.endpoint.model
    assert cfg.generate.pairs_per_chunk == 5
    assert 0 < cfg.chunk.threshold < 1


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "absent.toml")
