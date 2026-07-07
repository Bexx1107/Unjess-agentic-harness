"""Tests for unjess.llm.router — provider detection and routing."""

import pytest

from unjess.llm.router import _match_prefix, _detect_provider


# ---------------------------------------------------------------------------
# _match_prefix
# ---------------------------------------------------------------------------

class TestMatchPrefix:
    """Tests for the _match_prefix helper."""

    def test_exact_match(self) -> None:
        assert _match_prefix("o1", "o1") is True

    def test_dash_suffix_match(self) -> None:
        assert _match_prefix("o1-mini", "o1") is True

    def test_no_match_longer_prefix(self) -> None:
        assert _match_prefix("o100", "o1") is False

    def test_no_match_different_string(self) -> None:
        assert _match_prefix("gpt-4", "o1") is False

    def test_empty_prefix_no_match(self) -> None:
        assert _match_prefix("anything", "") is False

    def test_empty_model(self) -> None:
        assert _match_prefix("", "o1") is False


# ---------------------------------------------------------------------------
# _detect_provider — OpenAI
# ---------------------------------------------------------------------------

class TestDetectProviderOpenAI:
    """Tests for OpenAI model detection."""

    def test_gpt4(self) -> None:
        assert _detect_provider("gpt-4") == "openai"

    def test_gpt4o(self) -> None:
        assert _detect_provider("gpt-4o") == "openai"

    def test_gpt4o_mini(self) -> None:
        assert _detect_provider("gpt-4o-mini") == "openai"

    def test_gpt35_turbo(self) -> None:
        assert _detect_provider("gpt-3.5-turbo") == "openai"

    def test_chatgpt(self) -> None:
        assert _detect_provider("chatgpt-4o-latest") == "openai"

    def test_o1(self) -> None:
        assert _detect_provider("o1") == "openai"

    def test_o1_mini(self) -> None:
        assert _detect_provider("o1-mini") == "openai"

    def test_o3(self) -> None:
        assert _detect_provider("o3") == "openai"

    def test_o3_mini(self) -> None:
        assert _detect_provider("o3-mini") == "openai"

    def test_o4(self) -> None:
        assert _detect_provider("o4") == "openai"

    def test_o4_mini(self) -> None:
        assert _detect_provider("o4-mini") == "openai"

    def test_case_insensitive(self) -> None:
        assert _detect_provider("GPT-4o") == "openai"


# ---------------------------------------------------------------------------
# _detect_provider — Anthropic
# ---------------------------------------------------------------------------

class TestDetectProviderAnthropic:
    """Tests for Anthropic model detection."""

    def test_claude_sonnet(self) -> None:
        assert _detect_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_claude_opus(self) -> None:
        assert _detect_provider("claude-opus-4-20250514") == "anthropic"

    def test_claude_haiku(self) -> None:
        assert _detect_provider("claude-3-5-haiku-latest") == "anthropic"

    def test_claude_generic(self) -> None:
        assert _detect_provider("claude-3") == "anthropic"

    def test_case_insensitive(self) -> None:
        assert _detect_provider("Claude-Sonnet-4") == "anthropic"


# ---------------------------------------------------------------------------
# _detect_provider — Google
# ---------------------------------------------------------------------------

class TestDetectProviderGoogle:
    """Tests for Google model detection."""

    def test_gemini_pro(self) -> None:
        assert _detect_provider("gemini-pro") == "google"

    def test_gemini_2_flash(self) -> None:
        assert _detect_provider("gemini-2.5-flash") == "google"

    def test_gemini_1_5_pro(self) -> None:
        assert _detect_provider("gemini-1.5-pro-latest") == "google"


# ---------------------------------------------------------------------------
# _detect_provider — Other providers
# ---------------------------------------------------------------------------

class TestDetectProviderOther:
    """Tests for Groq, Mistral, XAI, OpenRouter, Cerebras, and Ollama."""

    # Groq
    def test_llama(self) -> None:
        assert _detect_provider("llama-3.1-70b-versatile") == "groq"

    def test_mixtral(self) -> None:
        assert _detect_provider("mixtral-8x7b-32768") == "groq"

    def test_gemma(self) -> None:
        assert _detect_provider("gemma-7b-it") == "groq"

    # Mistral
    def test_mistral(self) -> None:
        assert _detect_provider("mistral-large-latest") == "mistral"

    def test_codestral(self) -> None:
        assert _detect_provider("codestral-latest") == "mistral"

    def test_pixtral(self) -> None:
        assert _detect_provider("pixtral-large-2411") == "mistral"

    def test_open_mistral(self) -> None:
        assert _detect_provider("open-mistral-nemo") == "mistral"

    # XAI
    def test_grok(self) -> None:
        assert _detect_provider("grok-3") == "xai"

    # OpenRouter
    def test_openrouter_prefix(self) -> None:
        assert _detect_provider("openrouter/google/gemini-pro") == "openrouter"

    def test_openrouter_free_suffix(self) -> None:
        assert _detect_provider("meta-llama/llama-3-8b-instruct:free") == "openrouter"

    # Cerebras
    def test_cerebras_glm(self) -> None:
        assert _detect_provider("glm-4-9b") == "cerebras"

    def test_cerebras_zai(self) -> None:
        assert _detect_provider("zai-1b") == "cerebras"

    # Ollama (has colon but not :free)
    def test_ollama_colon(self) -> None:
        assert _detect_provider("llama3.1:8b") == "groq"  # llama prefix wins first

    def test_ollama_custom_model(self) -> None:
        assert _detect_provider("my-custom-model:latest") == "ollama"

    # Default fallback
    def test_unknown_model_defaults_to_openai(self) -> None:
        assert _detect_provider("some-unknown-model") == "openai"

    def test_empty_model_defaults_to_openai(self) -> None:
        assert _detect_provider("") == "openai"
