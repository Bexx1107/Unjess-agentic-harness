"""Tests for Kimi (Moonshot AI) and Qwen (Alibaba DashScope) provider integration."""

import pytest
from unjess.config import Settings, _load_env_keys
from unjess.llm.router import _detect_provider, ProviderRouter


def test_detect_provider_kimi() -> None:
    assert _detect_provider("kimi-k3") == "kimi"
    assert _detect_provider("kimi-k2.7-code") == "kimi"
    assert _detect_provider("moonshot-v1-128k") == "kimi"


def test_detect_provider_qwen() -> None:
    assert _detect_provider("qwen-max") == "qwen"
    assert _detect_provider("qwen-plus") == "qwen"
    assert _detect_provider("qwen2.5-coder-32b-instruct") == "qwen"
    assert _detect_provider("qwen3-coder") == "qwen"


def test_env_key_loading_kimi_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")

    settings = Settings()
    _load_env_keys(settings)

    assert settings.api_keys.get("kimi") == "test-moonshot-key"
    assert settings.api_keys.get("qwen") == "test-dashscope-key"


def test_router_initialization_kimi_qwen() -> None:
    settings = Settings(
        api_keys={
            "kimi": "sk-kimi-dummy",
            "qwen": "sk-qwen-dummy",
        }
    )
    router = ProviderRouter(settings)

    assert "kimi" in router._providers
    assert "qwen" in router._providers

    assert router._providers["kimi"].provider_name == "kimi"
    assert router._providers["qwen"].provider_name == "qwen"
