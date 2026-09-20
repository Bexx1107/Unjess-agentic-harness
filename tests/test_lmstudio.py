"""Unit tests for LM Studio provider integration."""

import pytest
from unjess.config import Settings
from unjess.llm.router import ProviderRouter


def test_lmstudio_settings_default():
    settings = Settings()
    assert settings.lmstudio_base_url == "http://localhost:1234"


def test_lmstudio_router_initialization():
    settings = Settings(lmstudio_base_url="http://localhost:1234")
    router = ProviderRouter(settings)
    assert "lmstudio" in router.available_providers
    provider, model = router.get_provider("local-model", explicit_provider="lmstudio")
    assert provider is not None
    assert provider.provider_name == "lmstudio"
