"""Tests for unjess.first_run — first-run setup wizard and helpers."""

import os
from unittest.mock import MagicMock, patch

import pytest

from unjess.config import Settings
from unjess.first_run import (
    _PROVIDER_DISPLAY,
    _PROVIDER_MODELS,
    _ENV_KEY_NAMES,
    _detect_available_providers,
    _get_api_key,
    _fetch_live_models,
)


# ---------------------------------------------------------------------------
# _detect_available_providers
# ---------------------------------------------------------------------------


class TestDetectAvailableProviders:
    """Tests for provider availability detection."""

    def test_no_keys_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)
        settings = Settings()
        result = _detect_available_providers(settings)
        # Only openai-oauth and ollama are always available
        assert result["openai-oauth"] is True
        assert result["ollama"] is True
        assert result["openai"] is False
        assert result["anthropic"] is False
        assert result["google"] is False
        assert result["groq"] is False
        assert result["mistral"] is False

    def test_detects_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        result = _detect_available_providers(Settings())
        assert result["google"] is True
        assert result["openai"] is False

    def test_detects_saved_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)
        settings = Settings(api_keys={"anthropic": "sk-test"})
        result = _detect_available_providers(settings)
        assert result["anthropic"] is True

    def test_env_overrides_empty_saved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        settings = Settings(api_keys={})
        result = _detect_available_providers(settings)
        assert result["openai"] is True

    def test_settings_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)
        result = _detect_available_providers(None)
        assert result["ollama"] is True
        assert result["openai-oauth"] is True

    def test_all_providers_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for provider, env_var in _ENV_KEY_NAMES.items():
            monkeypatch.setenv(env_var, f"key-for-{provider}")
        result = _detect_available_providers(Settings())
        for provider in _ENV_KEY_NAMES:
            assert result[provider] is True

    def test_xai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-key")
        result = _detect_available_providers(Settings())
        assert result["xai"] is True

    def test_cerebras_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CEREBRAS_API_KEY", "cb-key")
        result = _detect_available_providers(Settings())
        assert result["cerebras"] is True

    def test_openrouter_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        result = _detect_available_providers(Settings())
        assert result["openrouter"] is True


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


class TestGetApiKey:
    """Tests for API key retrieval from env/settings."""

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
        result = _get_api_key("google", "GOOGLE_API_KEY")
        assert result == "env-google-key"

    def test_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        settings = Settings(api_keys={"google": "settings-key"})
        result = _get_api_key("google", "GOOGLE_API_KEY", settings)
        assert result == "settings-key"

    def test_env_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        settings = Settings(api_keys={"google": "settings-key"})
        result = _get_api_key("google", "GOOGLE_API_KEY", settings)
        assert result == "env-key"

    def test_no_key_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        result = _get_api_key("google", "GOOGLE_API_KEY")
        assert result == ""

    def test_no_key_no_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = _get_api_key("openai", "OPENAI_API_KEY", None)
        assert result == ""


# ---------------------------------------------------------------------------
# _fetch_live_models
# ---------------------------------------------------------------------------


class TestFetchLiveModels:
    """Tests for live model fetching (network calls mocked)."""

    def test_unknown_provider_returns_empty(self) -> None:
        result = _fetch_live_models("nonexistent_provider")
        assert result == []

    @patch("unjess.first_run._fetch_ollama_models")
    def test_ollama_delegation(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = [("llama3.1", "8GB")]
        result = _fetch_live_models("ollama")
        assert result == [("llama3.1", "8GB")]
        mock_fetch.assert_called_once()

    @patch("unjess.first_run._fetch_google_models")
    def test_google_delegation(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = [("gemini-3.1-flash", "Fast")]
        result = _fetch_live_models("google", Settings())
        mock_fetch.assert_called_once()

    @patch("unjess.first_run._fetch_groq_models")
    def test_groq_delegation(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = []
        result = _fetch_live_models("groq", Settings())
        mock_fetch.assert_called_once()

    @patch("unjess.first_run._fetch_mistral_models")
    def test_mistral_delegation(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = []
        result = _fetch_live_models("mistral", Settings())
        mock_fetch.assert_called_once()

    @patch("unjess.first_run._fetch_openai_models")
    def test_openai_delegation(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = []
        result = _fetch_live_models("openai", Settings())
        mock_fetch.assert_called_once()

    def test_fetch_catches_exceptions(self) -> None:
        with patch("unjess.first_run._fetch_ollama_models", side_effect=Exception("network")):
            result = _fetch_live_models("ollama")
            assert result == []


# ---------------------------------------------------------------------------
# Constants / data integrity
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_provider_display_has_all_known(self) -> None:
        expected = {
            "openai", "openai-oauth", "anthropic", "google",
            "groq", "mistral", "xai", "openrouter", "cerebras", "ollama",
        }
        assert expected.issubset(set(_PROVIDER_DISPLAY.keys()))

    def test_provider_models_has_entries(self) -> None:
        assert len(_PROVIDER_MODELS) > 0

    def test_each_provider_model_is_tuple(self) -> None:
        for provider, models in _PROVIDER_MODELS.items():
            for entry in models:
                assert isinstance(entry, tuple)
                assert len(entry) == 2

    def test_env_key_names_mapping(self) -> None:
        assert _ENV_KEY_NAMES["openai"] == "OPENAI_API_KEY"
        assert _ENV_KEY_NAMES["google"] == "GOOGLE_API_KEY"
        assert _ENV_KEY_NAMES["cerebras"] == "CEREBRAS_API_KEY"

    def test_provider_models_openai(self) -> None:
        models = _PROVIDER_MODELS["openai"]
        model_names = [m[0] for m in models]
        assert "gpt-5.4" in model_names

    def test_provider_models_google(self) -> None:
        models = _PROVIDER_MODELS["google"]
        model_names = [m[0] for m in models]
        assert "gemini-3.1-flash" in model_names

    def test_provider_models_ollama(self) -> None:
        models = _PROVIDER_MODELS["ollama"]
        model_names = [m[0] for m in models]
        assert "llama3.5" in model_names


# ---------------------------------------------------------------------------
# run_first_setup (smoke / integration with mocks)
# ---------------------------------------------------------------------------


class TestRunFirstSetup:
    """Tests for the interactive setup wizard with mocked console."""

    @patch("unjess.first_run.save_config")
    @patch("unjess.first_run._fetch_live_models", return_value=[])
    @patch("unjess.first_run.Prompt.ask")
    @patch("unjess.first_run.Confirm.ask", return_value=False)
    def test_free_mode_selection(
        self,
        mock_confirm: MagicMock,
        mock_prompt: MagicMock,
        mock_fetch: MagicMock,
        mock_save: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from unjess.first_run import run_first_setup

        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)

        # Choice "1" selects Free Mode; remaining prompts provide keys or skip
        mock_prompt.side_effect = ["1", "", "", "", "", ""]
        console = MagicMock()

        settings = Settings()
        result = run_first_setup(settings, console)
        assert result.free_mode_enabled is True
        mock_save.assert_called_once()

    @patch("unjess.first_run.save_config")
    @patch("unjess.first_run._fetch_live_models", return_value=[])
    @patch("unjess.first_run.Prompt.ask")
    @patch("unjess.first_run.Confirm.ask", return_value=False)
    def test_ollama_provider_selection(
        self,
        mock_confirm: MagicMock,
        mock_prompt: MagicMock,
        mock_fetch: MagicMock,
        mock_save: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from unjess.first_run import run_first_setup

        for var in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(var, raising=False)

        # Choice "11" = ollama, then pick model "1"
        mock_prompt.side_effect = ["11", "1"]
        console = MagicMock()

        settings = Settings()
        result = run_first_setup(settings, console)
        assert result.provider == "ollama"
        assert result.free_mode_enabled is False
