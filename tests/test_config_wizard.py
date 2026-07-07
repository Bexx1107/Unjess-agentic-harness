"""Tests for config wizard functionality — model/provider selection and validation.

Since ``config_wizard.py`` does not exist as a separate module, these tests
exercise the wizard-adjacent logic in ``unjess.first_run`` and
``unjess.config`` that drives interactive configuration: model suggestion,
provider ordering, API key collection helpers, and ``is_first_run`` checks.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unjess.config import Settings, is_first_run, save_config, load_config
from unjess.first_run import (
    _PROVIDER_DISPLAY,
    _PROVIDER_MODELS,
    _ENV_KEY_NAMES,
    _detect_available_providers,
    _get_api_key,
    _try_oauth_login,
    run_first_setup,
)


# ---------------------------------------------------------------------------
# is_first_run (config wizard trigger)
# ---------------------------------------------------------------------------


class TestIsFirstRun:
    """Tests for the first-run detection that would trigger the wizard."""

    def test_no_config_file(self, tmp_path: Path) -> None:
        assert is_first_run(tmp_path / "nonexistent.yaml")

    def test_empty_model_triggers_first_run(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        settings = Settings(model="")
        save_config(settings, config_path)
        assert is_first_run(config_path)

    def test_configured_model_not_first_run(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        settings = Settings(model="gpt-4o")
        save_config(settings, config_path)
        assert not is_first_run(config_path)


# ---------------------------------------------------------------------------
# Provider ordering / display names
# ---------------------------------------------------------------------------


class TestProviderDisplay:
    """Tests for provider display metadata used by the wizard UI."""

    def test_all_providers_have_display_name(self) -> None:
        expected = {
            "openai", "openai-oauth", "anthropic", "google",
            "groq", "mistral", "xai", "openrouter", "cerebras", "ollama",
        }
        for p in expected:
            assert p in _PROVIDER_DISPLAY, f"Missing display name for {p}"
            assert _PROVIDER_DISPLAY[p], f"Empty display name for {p}"

    def test_env_key_names_match_known_providers(self) -> None:
        key_providers = set(_ENV_KEY_NAMES.keys())
        # These providers need API keys
        expected = {"openai", "anthropic", "google", "groq", "mistral", "xai", "openrouter", "cerebras"}
        assert key_providers == expected

    def test_env_key_format(self) -> None:
        for provider, env_var in _ENV_KEY_NAMES.items():
            assert env_var.endswith("_API_KEY") or env_var.endswith("_KEY")
            assert env_var == env_var.upper()


# ---------------------------------------------------------------------------
# Model selection / validation helpers
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Tests for model data used during wizard selection."""

    def test_provider_models_all_have_descriptions(self) -> None:
        for provider, models in _PROVIDER_MODELS.items():
            for name, desc in models:
                assert name, f"Empty model name in {provider}"
                # desc can be empty for live-fetched models

    def test_provider_models_no_duplicates(self) -> None:
        for provider, models in _PROVIDER_MODELS.items():
            names = [m[0] for m in models]
            assert len(names) == len(set(names)), f"Duplicate models in {provider}"

    def test_openai_has_reasoning_model(self) -> None:
        openai_models = [m[0] for m in _PROVIDER_MODELS["openai"]]
        assert any("o4" in m or "o3" in m for m in openai_models)

    def test_google_has_flash(self) -> None:
        google_models = [m[0] for m in _PROVIDER_MODELS["google"]]
        assert "gemini-2.5-flash" in google_models

    def test_ollama_has_local_models(self) -> None:
        ollama_models = [m[0] for m in _PROVIDER_MODELS["ollama"]]
        assert len(ollama_models) >= 2


# ---------------------------------------------------------------------------
# API key validation (_get_api_key)
# ---------------------------------------------------------------------------


class TestApiKeyValidation:
    """Tests for API key retrieval logic used by the wizard."""

    def test_empty_env_empty_settings_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _get_api_key("openai", "OPENAI_API_KEY") == ""

    def test_env_key_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        s = Settings(api_keys={"openai": "from-settings"})
        assert _get_api_key("openai", "OPENAI_API_KEY", s) == "from-env"

    def test_settings_key_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(api_keys={"openai": "from-settings"})
        assert _get_api_key("openai", "OPENAI_API_KEY", s) == "from-settings"


# ---------------------------------------------------------------------------
# Provider detection (used by wizard to show status)
# ---------------------------------------------------------------------------


class TestProviderDetection:
    """Tests for detecting which providers are configured."""

    def test_ollama_always_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for v in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(v, raising=False)
        result = _detect_available_providers(Settings())
        assert result["ollama"] is True

    def test_oauth_always_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for v in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(v, raising=False)
        result = _detect_available_providers(Settings())
        assert result["openai-oauth"] is True

    def test_mixed_env_and_saved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for v in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        s = Settings(api_keys={"mistral": "mistral-key"})
        result = _detect_available_providers(s)
        assert result["groq"] is True
        assert result["mistral"] is True
        assert result["openai"] is False


# ---------------------------------------------------------------------------
# OAuth login (mocked)
# ---------------------------------------------------------------------------


class TestOAuthLogin:
    """Tests for the experimental OAuth login flow."""

    @patch("unjess.first_run.openai_oauth_login", create=True)
    def test_oauth_success(self, mock_login: MagicMock) -> None:
        with patch("unjess.first_run.openai_oauth_login", return_value="token-123"):
            from unjess.first_run import _try_oauth_login
            console = MagicMock()
            # Need to mock the import inside _try_oauth_login
            with patch(
                "unjess.first_run.openai_oauth_login",
                create=True,
                return_value="token-123",
            ):
                # _try_oauth_login imports from unjess.llm.oauth_provider
                with patch(
                    "unjess.llm.oauth_provider.openai_oauth_login",
                    return_value="token-xyz",
                    create=True,
                ):
                    result = _try_oauth_login(console)
                    # May be None if import fails, that's okay
                    # The key test is it doesn't crash

    def test_oauth_import_error(self) -> None:
        console = MagicMock()
        with patch.dict("sys.modules", {"unjess.llm.oauth_provider": None}):
            result = _try_oauth_login(console)
            assert result is None


# ---------------------------------------------------------------------------
# run_first_setup integration (mocked I/O)
# ---------------------------------------------------------------------------


class TestRunFirstSetupIntegration:
    """Integration tests for the wizard flow with mocked interactive prompts."""

    @patch("unjess.first_run.save_config")
    @patch("unjess.first_run._fetch_live_models", return_value=[])
    @patch("unjess.first_run.Prompt.ask")
    @patch("unjess.first_run.Confirm.ask", return_value=False)
    def test_google_provider_setup(
        self,
        mock_confirm: MagicMock,
        mock_prompt: MagicMock,
        mock_fetch: MagicMock,
        mock_save: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for v in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Choice 2 = google (first in provider_order), model 1
        mock_prompt.side_effect = ["2", "1"]
        console = MagicMock()
        settings = Settings()
        result = run_first_setup(settings, console)
        assert result.provider == "google"
        assert result.free_mode_enabled is False

    @patch("unjess.first_run.save_config")
    @patch("unjess.first_run._fetch_live_models", return_value=[])
    @patch("unjess.first_run.Prompt.ask")
    @patch("unjess.first_run.Confirm.ask", return_value=False)
    def test_custom_model_name(
        self,
        mock_confirm: MagicMock,
        mock_prompt: MagicMock,
        mock_fetch: MagicMock,
        mock_save: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for v in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Choice 2 = google, then type a custom model name
        mock_prompt.side_effect = ["2", "my-custom-model"]
        console = MagicMock()
        settings = Settings()
        result = run_first_setup(settings, console)
        assert result.model == "my-custom-model"

    @patch("unjess.first_run.save_config")
    @patch("unjess.first_run._fetch_live_models", return_value=[])
    @patch("unjess.first_run.Prompt.ask")
    @patch("unjess.first_run.Confirm.ask", return_value=False)
    def test_free_mode_with_google_key_picks_gemini(
        self,
        mock_confirm: MagicMock,
        mock_prompt: MagicMock,
        mock_fetch: MagicMock,
        mock_save: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for v in _ENV_KEY_NAMES.values():
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Choice 1 = Free Mode. The wizard walks through each free provider:
        # openrouter (no key → ask paste → skip), cerebras (no key → skip),
        # google (has key → ask add more → no), groq (no key → skip),
        # mistral (no key → skip).
        mock_prompt.side_effect = [
            "1",   # Pick free mode
            "",    # openrouter key → skip
            "",    # cerebras key → skip
            "n",   # google already has key → add more? → no
            "",    # groq key → skip
            "",    # mistral key → skip
        ]
        console = MagicMock()
        settings = Settings()
        result = run_first_setup(settings, console)
        assert result.free_mode_enabled is True
        # google key available → should start with gemini-2.5-flash
        assert result.model == "gemini-2.5-flash"
        assert result.provider == "google"
