"""Provider router — auto-detect provider from model name, manage fallback chain."""

import logging
import os
from typing import Any, Generator

from unjess.config import Settings
from unjess.llm.base import LLMProvider, LLMResponse, StreamChunk, Usage
from unjess.llm.openai_compat import OpenAICompatibleProvider
from unjess.llm.anthropic_provider import AnthropicProvider
from unjess.llm.google_provider import GoogleProvider

logger = logging.getLogger(__name__)


def _match_prefix(model: str, prefix: str) -> bool:
    """Check if model matches a prefix exactly or with a dash suffix.

    This avoids false positives like 'o100' matching 'o1'.
    """
    return model == prefix or model.startswith(prefix + "-")


def _detect_provider(model: str) -> str:
    """Auto-detect provider name from model string.

    Rules:
        gpt-* / o1(-*) / o3(-*) / o4(-*) / chatgpt-* → openai
        claude-*                                       → anthropic
        gemini-*                                       → google
        llama* / mixtral* / gemma*                     → groq
        mistral-* / codestral* / pixtral*              → mistral
        contains ":"                                   → ollama
    """
    m = model.lower()

    if m.startswith(("gpt-", "chatgpt-")) or _match_prefix(m, "o1") or _match_prefix(m, "o3") or _match_prefix(m, "o4"):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("gemini-"):
        return "google"
    if m.startswith(("llama", "mixtral", "gemma")):
        return "groq"
    if m.startswith(("mistral-", "codestral", "pixtral", "open-mistral")):
        return "mistral"
    if m.startswith("grok-"):
        return "xai"
    if m.startswith(("kimi-", "moonshot-")):
        return "kimi"
    if m.startswith(("qwen-", "qwen2", "qwen3")):
        return "qwen"
    if m.startswith("openrouter/") or ":free" in m:
        return "openrouter"
    if m.startswith("glm-") or m.startswith("zai-"):
        return "cerebras"
    if m.startswith("llamacpp") or m == "llamacpp":
        return "llamacpp"
    if m.startswith("ollama-api/"):
        return "ollama-api"
    if m.startswith("ollama/"):
        return "ollama"
    if ":" in m:
        return "ollama"

    return "openai"  # default fallback


class ProviderRouter:
    """Routes model requests to the correct LLM provider.

    Handles provider instantiation, model → provider mapping,
    and fallback chains.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[str, LLMProvider] = {}
        self._init_providers()

    def reload_providers(self) -> None:
        """Re-initialize providers from current settings.

        Call this after API keys change (e.g. after the setup wizard)
        to pick up newly configured providers.
        """
        self._providers.clear()
        self._init_providers()

    def _init_providers(self) -> None:
        """Instantiate providers based on available API keys."""
        keys = self._settings.api_keys
        llm_timeout = float(getattr(self._settings, "llm_timeout", 300.0))

        # OpenAI
        if keys.get("openai"):
            self._providers["openai"] = OpenAICompatibleProvider(
                api_key=keys["openai"],
                default_model=self._settings.model,
                name="openai",
                timeout=llm_timeout,
            )

        # Anthropic
        if keys.get("anthropic"):
            self._providers["anthropic"] = AnthropicProvider(
                api_key=keys["anthropic"],
                timeout=llm_timeout,
            )

        # Google
        if keys.get("google"):
            self._providers["google"] = GoogleProvider(
                api_key=keys["google"],
            )

        # Groq (OpenAI-compatible, free tier available)
        if keys.get("groq"):
            self._providers["groq"] = OpenAICompatibleProvider(
                api_key=keys["groq"],
                base_url="https://api.groq.com/openai/v1",
                default_model="llama-3.3-70b-versatile",
                name="groq",
                timeout=llm_timeout,
            )

        # Mistral (OpenAI-compatible, free tier available)
        if keys.get("mistral"):
            self._providers["mistral"] = OpenAICompatibleProvider(
                api_key=keys["mistral"],
                base_url="https://api.mistral.ai/v1",
                default_model="codestral-latest",
                name="mistral",
                timeout=llm_timeout,
            )

        # xAI / Grok (OpenAI-compatible)
        if keys.get("xai"):
            self._providers["xai"] = OpenAICompatibleProvider(
                api_key=keys["xai"],
                base_url="https://api.x.ai/v1",
                default_model="grok-4.1-fast",
                name="xai",
                timeout=llm_timeout,
            )

        # OpenRouter (OpenAI-compatible, 50+ free models)
        if keys.get("openrouter"):
            self._providers["openrouter"] = OpenAICompatibleProvider(
                api_key=keys["openrouter"],
                base_url="https://openrouter.ai/api/v1",
                default_model="openrouter/free",
                name="openrouter",
                timeout=llm_timeout,
            )

        # Cerebras (OpenAI-compatible, ultra-fast free tier)
        if keys.get("cerebras"):
            self._providers["cerebras"] = OpenAICompatibleProvider(
                api_key=keys["cerebras"],
                base_url="https://api.cerebras.ai/v1",
                default_model="zai-glm-4.7",
                name="cerebras",
                timeout=llm_timeout,
            )

        # Kimi / Moonshot AI (OpenAI-compatible)
        if keys.get("kimi") or keys.get("moonshot"):
            self._providers["kimi"] = OpenAICompatibleProvider(
                api_key=keys.get("kimi") or keys.get("moonshot", ""),
                base_url=getattr(self._settings, "kimi_base_url", "https://api.moonshot.ai/v1"),
                default_model="kimi-k3",
                name="kimi",
                timeout=llm_timeout,
            )

        # Qwen / DashScope (OpenAI-compatible)
        if keys.get("qwen") or keys.get("dashscope"):
            self._providers["qwen"] = OpenAICompatibleProvider(
                api_key=keys.get("qwen") or keys.get("dashscope", ""),
                base_url=getattr(self._settings, "qwen_base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                default_model="qwen-max",
                name="qwen",
                timeout=llm_timeout,
            )

        # Ollama (Local — runs locally without API key)
        try:
            ollama_url = getattr(self._settings, "ollama_base_url", "http://localhost:11434").rstrip("/")
            if not ollama_url.endswith("/v1"):
                ollama_url = f"{ollama_url}/v1"
            self._providers["ollama"] = OpenAICompatibleProvider(
                api_key="ollama",
                base_url=ollama_url,
                default_model="llama3.1",
                name="ollama",
                timeout=llm_timeout,
            )
        except Exception:
            pass  # Ollama local not available — skip silently

        # Ollama API (Cloud — requires Ollama API key / subscription)
        ollama_api_key = (
            keys.get("ollama-api")
            or keys.get("ollama")
            or os.environ.get("OLLAMA_API_KEY", "")
        )
        if ollama_api_key and ollama_api_key != "ollama":
            try:
                ollama_api_url = getattr(self._settings, "ollama_api_base_url", "https://ollama.com").rstrip("/")
                if "api.ollama.com" in ollama_api_url:
                    ollama_api_url = ollama_api_url.replace("api.ollama.com", "ollama.com")
                if not ollama_api_url.endswith("/v1"):
                    ollama_api_url = f"{ollama_api_url}/v1"
                self._providers["ollama-api"] = OpenAICompatibleProvider(
                    api_key=ollama_api_key,
                    base_url=ollama_api_url,
                    default_model="llama3.3",
                    name="ollama-api",
                    timeout=llm_timeout,
                )
            except Exception:
                pass  # Ollama API not available — skip silently

        # llama.cpp (always available — no key needed)
        try:
            self._providers["llamacpp"] = OpenAICompatibleProvider(
                api_key="llamacpp",
                base_url=f"{self._settings.llamacpp_base_url}/v1",
                default_model="llamacpp",
                name="llamacpp",
                timeout=llm_timeout,
            )
        except Exception:
            pass  # llama.cpp not available — skip silently

        # LM Studio (always available — no key needed, defaults to http://localhost:1234)
        try:
            lm_url = getattr(self._settings, "lmstudio_base_url", "http://localhost:1234").rstrip("/")
            if not lm_url.endswith("/v1"):
                lm_url = f"{lm_url}/v1"
            self._providers["lmstudio"] = OpenAICompatibleProvider(
                api_key="lmstudio",
                base_url=lm_url,
                default_model="local-model",
                name="lmstudio",
                timeout=llm_timeout,
            )
        except Exception:
            pass  # LM Studio not available — skip silently

    def get_provider(
        self,
        model: str = "",
        explicit_provider: str = "",
    ) -> tuple[LLMProvider, str]:
        """Resolve the provider for a given model.

        Args:
            model: Model name (used for auto-detection if no explicit provider).
            explicit_provider: Force a specific provider.

        Returns:
            (provider_instance, resolved_model_name)

        Raises:
            ValueError: If no suitable provider is available.
        """
        model = model or self._settings.model
        provider_name = explicit_provider or self._settings.provider or _detect_provider(model)

        if provider_name in self._providers:
            return self._providers[provider_name], model

        # Provider not available — try to explain why
        if provider_name == "ollama":
            raise ValueError(
                f"Provider 'ollama' is not available. Ensure Ollama is running at {self._settings.ollama_base_url}."
            )
        if provider_name == "ollama-api":
            raise ValueError(
                "Provider 'ollama-api' requires an API key. Set OLLAMA_API_KEY or configure it in Settings."
            )

        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "xai": "XAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "ollama-api": "OLLAMA_API_KEY",
        }
        if provider_name in env_vars:
            env_var = env_vars[provider_name]
            raise ValueError(
                f"Provider '{provider_name}' requires an API key. "
                f"Set {env_var} or add it to ~/.unjess/config.yaml"
            )

        raise ValueError(f"Unknown provider: '{provider_name}'")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> LLMResponse:
        """Route a chat request through the provider, with fallback chain."""
        return self._with_fallback("chat", messages=messages, tools=tools, model=model)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> Generator[StreamChunk, None, None]:
        """Route a streaming chat request through the provider, with fallback chain.

        Wraps the returned generator to catch rate-limit errors that occur
        during iteration (not during generator creation).
        """
        return self._stream_with_rotation(messages=messages, tools=tools, model=model)

    def abort(self) -> None:
        """Abort any ongoing requests or streams across active providers."""
        for provider in self._providers.values():
            if hasattr(provider, "abort"):
                try:
                    provider.abort()
                except Exception as exc:
                    logger.debug("Provider abort error: %s", exc)

    def _is_rate_limit(self, exc: Exception) -> bool:
        """Check if an exception is a rate limit / quota exhausted error."""
        msg = str(exc).lower()
        return any(indicator in msg for indicator in [
            "429", "rate_limit", "rate limit", "resource_exhausted",
            "quota", "too many requests",
        ])

    def _stream_with_rotation(self, **kwargs: Any) -> Generator[StreamChunk, None, None]:
        """Wrap streaming to catch rate-limit errors during iteration.

        For streaming calls, the API request may not actually fire until
        the first chunk is consumed.  This wrapper catches rate-limit
        errors on iteration and retries with a rotated API key.
        """
        model = kwargs.get("model", "") or self._settings.model
        prov_name = self._settings.provider or _detect_provider(model)
        pool_size = len(self._settings.api_key_pool.get(prov_name, []))
        max_attempts = max(1, pool_size)

        for attempt in range(max_attempts):
            try:
                gen = self._with_fallback("chat_stream", **kwargs)
                # Yield all chunks — this is where rate-limit errors surface
                yield from gen
                return  # completed successfully
            except Exception as exc:
                if self._is_rate_limit(exc) and attempt < max_attempts - 1:
                    rotated = self._rotate_key(prov_name)
                    if rotated:
                        logger.info(
                            "🔄 Stream rate-limited, rotated %s key (attempt %d/%d)",
                            prov_name,
                            attempt + 1,
                            max_attempts,
                        )
                        continue
                raise  # no more keys or not a rate limit

    def _with_fallback(self, method: str, **kwargs: Any) -> Any:
        """Try the primary provider, then each fallback in the chain.

        On rate limit errors, tries rotating API keys first before
        moving to the next fallback provider.
        """
        model = kwargs.get("model", "") or self._settings.model
        errors: list[str] = []

        # Build attempt order: primary first, then fallback chain
        attempts: list[tuple[str, str]] = []

        # Primary
        provider_name = self._settings.provider or _detect_provider(model)
        attempts.append((provider_name, model))

        # Fallback chain entries (format: "provider/model")
        fallback = self._settings.fallback_chain
        if not fallback:
            # Auto-build fallback from all configured providers
            fallback = self._auto_fallback_chain(provider_name)

        for entry in fallback:
            if "/" in entry:
                fb_provider, fb_model = entry.split("/", 1)
            else:
                fb_provider = _detect_provider(entry)
                fb_model = entry
            attempts.append((fb_provider, fb_model))

        for prov_name, attempt_model in attempts:
            if prov_name not in self._providers:
                errors.append(f"{prov_name}: not configured")
                continue

            provider = self._providers[prov_name]

            # Try with key rotation on rate limits
            max_key_rotations = len(
                self._settings.api_key_pool.get(prov_name, [])
            )
            for key_attempt in range(max(1, max_key_rotations)):
                try:
                    fn = getattr(provider, method)
                    attempt_kwargs = kwargs.copy()
                    attempt_kwargs["model"] = attempt_model
                    return fn(**attempt_kwargs)
                except Exception as exc:
                    if self._is_rate_limit(exc) and key_attempt < max_key_rotations - 1:
                        # Try rotating to next API key
                        rotated = self._rotate_key(prov_name)
                        if rotated:
                            provider = self._providers[prov_name]
                            logger.info("Rotated %s key, retrying...", prov_name)
                            continue
                    error_msg = f"{prov_name}/{attempt_model}: {exc}"
                    errors.append(error_msg)
                    logger.warning("Provider failed: %s", error_msg)
                    break  # Move to next provider

        raise ConnectionError(
            "All providers failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    def _rotate_key(self, provider: str) -> bool:
        """Rotate to the next API key for a provider.

        Returns True if a new key was activated.
        """
        pool = self._settings.api_key_pool.get(provider, [])
        if len(pool) <= 1:
            return False

        # Find current key index
        current_key = self._settings.api_keys.get(provider, "")
        try:
            current_idx = pool.index(current_key)
        except ValueError:
            current_idx = 0

        next_idx = (current_idx + 1) % len(pool)
        new_key = pool[next_idx]
        self._settings.api_keys[provider] = new_key

        # Rebuild the provider with the new key
        self._init_single_provider(provider, new_key)

        logger.info("🔄 Rotated %s API key (%d/%d)", provider, next_idx + 1, len(pool))
        return True

    def _init_single_provider(self, provider: str, api_key: str) -> None:
        """Re-initialize a single provider with a new API key."""
        llm_timeout = float(getattr(self._settings, "llm_timeout", 300.0))
        if provider == "google":
            self._providers["google"] = GoogleProvider(api_key=api_key)
        elif provider == "anthropic":
            self._providers["anthropic"] = AnthropicProvider(api_key=api_key, timeout=llm_timeout)
        elif provider == "openai":
            self._providers["openai"] = OpenAICompatibleProvider(
                api_key=api_key,
                default_model=self._settings.model,
                name="openai",
                timeout=llm_timeout,
            )
        elif provider == "groq":
            self._providers["groq"] = OpenAICompatibleProvider(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                default_model="llama-3.3-70b-versatile",
                name="groq",
                timeout=llm_timeout,
            )
        elif provider == "mistral":
            self._providers["mistral"] = OpenAICompatibleProvider(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
                default_model="codestral-latest",
                name="mistral",
                timeout=llm_timeout,
            )
        elif provider == "xai":
            self._providers["xai"] = OpenAICompatibleProvider(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                default_model="grok-4.1-fast",
                name="xai",
                timeout=llm_timeout,
            )
        elif provider == "openrouter":
            self._providers["openrouter"] = OpenAICompatibleProvider(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_model="openrouter/free",
                name="openrouter",
                timeout=llm_timeout,
            )
        elif provider == "cerebras":
            self._providers["cerebras"] = OpenAICompatibleProvider(
                api_key=api_key,
                base_url="https://api.cerebras.ai/v1",
                default_model="zai-glm-4.7",
                name="cerebras",
                timeout=llm_timeout,
            )
        elif provider == "ollama":
            ollama_url = getattr(self._settings, "ollama_base_url", "http://localhost:11434").rstrip("/")
            if not ollama_url.endswith("/v1"):
                ollama_url = f"{ollama_url}/v1"
            self._providers["ollama"] = OpenAICompatibleProvider(
                api_key=api_key or "ollama",
                base_url=ollama_url,
                default_model="llama3.1",
                name="ollama",
                timeout=llm_timeout,
            )
        elif provider == "ollama-api":
            ollama_api_url = getattr(self._settings, "ollama_api_base_url", "https://ollama.com").rstrip("/")
            if "api.ollama.com" in ollama_api_url:
                ollama_api_url = ollama_api_url.replace("api.ollama.com", "ollama.com")
            if not ollama_api_url.endswith("/v1"):
                ollama_api_url = f"{ollama_api_url}/v1"
            self._providers["ollama-api"] = OpenAICompatibleProvider(
                api_key=api_key,
                base_url=ollama_api_url,
                default_model="llama3.3",
                name="ollama-api",
                timeout=llm_timeout,
            )
        else:
            # Unknown provider — full reload
            self.reload_providers()

    def _auto_fallback_chain(self, primary: str) -> list[str]:
        """Build a fallback chain from all configured providers, excluding primary.

        Returns entries like 'cerebras/zai-glm-4.7' for auto-fallback.
        """
        _DEFAULT_MODELS = {
            "cerebras": "zai-glm-4.7",
            "groq": "llama-3.3-70b-versatile",
            "mistral": "codestral-latest",
            "google": "gemini-3.1-flash",
            "openrouter": "google/gemini-3.1-flash:free",
            "ollama": "qwen3.5:9b",
            "ollama-api": "llama3.3",
        }
        chain: list[str] = []
        for prov_name in self._providers:
            if prov_name == primary:
                continue
            default = _DEFAULT_MODELS.get(prov_name, "")
            if default:
                chain.append(f"{prov_name}/{default}")
        return chain

    @property
    def available_providers(self) -> list[str]:
        """List of configured provider names."""
        return list(self._providers.keys())

    @property
    def active_provider_name(self) -> str:
        """Name of the currently selected provider."""
        return self._settings.provider or _detect_provider(self._settings.model)

    @property
    def active_model(self) -> str:
        """Currently selected model."""
        return self._settings.model

    def list_models(self, provider_name: str = "") -> list[str]:
        """List available models for a specific provider.

        Args:
            provider_name: Provider to query. Defaults to active provider.

        Returns:
            List of model ID strings.
        """
        name = provider_name or self.active_provider_name
        provider = self._providers.get(name)
        if provider is None:
            return []
        return provider.list_models()

    def list_all_models(self) -> dict[str, list[str]]:
        """List available models from all configured providers.

        Returns:
            Dict mapping provider name to sorted list of model IDs.
        """
        result: dict[str, list[str]] = {}
        for name, provider in self._providers.items():
            models = provider.list_models()
            if models:
                result[name] = models
        return result
