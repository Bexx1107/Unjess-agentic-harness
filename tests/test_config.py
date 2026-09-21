"""Tests for unjess.config — Settings dataclass, YAML persistence, env vars, CLI overrides."""

import platform
from pathlib import Path
from unittest.mock import patch

import yaml
import pytest

from unjess.config import (
    Settings,
    load_config,
    save_config,
    save_keys,
    ensure_config,
    is_first_run,
    apply_cli_overrides,
    load_workspace_config,
    merge_workspace_config,
    resolve_workspace,
    get_os_info,
    get_shell,
    _settings_from_dict,
    _load_env_keys,
    _ENV_KEY_MAP,
    IGNORE_PATTERNS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_LLM_TIMEOUT,
)


# ======================================================================
# Settings dataclass defaults
# ======================================================================

class TestSettingsDefaults:
    """Verify that a bare Settings() has the correct default values."""

    def test_model_default_is_empty(self) -> None:
        s = Settings()
        assert s.model == ""

    def test_provider_default_is_empty(self) -> None:
        s = Settings()
        assert s.provider == ""

    def test_api_keys_default_is_empty_dict(self) -> None:
        s = Settings()
        assert s.api_keys == {}

    def test_api_key_pool_default_is_empty_dict(self) -> None:
        s = Settings()
        assert s.api_key_pool == {}

    def test_ollama_base_url_default(self) -> None:
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_workspace_default(self) -> None:
        s = Settings()
        assert s.workspace == "."

    def test_fallback_chain_default(self) -> None:
        s = Settings()
        assert s.fallback_chain == []

    def test_max_iterations_default(self) -> None:
        s = Settings()
        assert s.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_subagent_max_turns_default(self) -> None:
        s = Settings()
        assert s.subagent_max_turns == 10

    def test_confirm_commands_default(self) -> None:
        s = Settings()
        assert s.confirm_commands is True

    def test_command_timeout_default(self) -> None:
        s = Settings()
        assert s.command_timeout == DEFAULT_COMMAND_TIMEOUT

    def test_verbosity_default(self) -> None:
        s = Settings()
        assert s.verbosity == "normal"

    def test_show_diffs_default(self) -> None:
        s = Settings()
        assert s.show_diffs is True

    def test_show_cost_default(self) -> None:
        s = Settings()
        assert s.show_cost is True

    def test_show_stats_default(self) -> None:
        s = Settings()
        assert s.show_stats is True

    def test_accent_color_default(self) -> None:
        s = Settings()
        assert s.accent_color == "#666666"

    def test_max_cost_per_session_default(self) -> None:
        s = Settings()
        assert s.max_cost_per_session == 0.0

    def test_max_tokens_per_message_default(self) -> None:
        s = Settings()
        assert s.max_tokens_per_message == 0

    def test_free_mode_enabled_default(self) -> None:
        s = Settings()
        assert s.free_mode_enabled is False

    def test_enable_rag_default(self) -> None:
        s = Settings()
        assert s.enable_rag is False

    def test_ignore_patterns_default_matches_constant(self) -> None:
        s = Settings()
        assert s.ignore_patterns == IGNORE_PATTERNS

    def test_ignore_patterns_are_independent_copies(self) -> None:
        s1 = Settings()
        s2 = Settings()
        s1.ignore_patterns.append("custom_pattern")
        assert "custom_pattern" not in s2.ignore_patterns

    def test_api_keys_are_independent_copies(self) -> None:
        s1 = Settings()
        s2 = Settings()
        s1.api_keys["test"] = "key"
        assert "test" not in s2.api_keys

    def test_llm_timeout_default(self) -> None:
        s = Settings()
        assert s.llm_timeout == DEFAULT_LLM_TIMEOUT
        assert s.llm_timeout == 300.0


# ======================================================================
# _settings_from_dict
# ======================================================================

class TestSettingsFromDict:
    """Verify dict-to-Settings conversion and unknown-key filtering."""

    def test_known_keys_applied(self) -> None:
        data = {"model": "gpt-4", "provider": "openai", "max_iterations": 100}
        s = _settings_from_dict(data)
        assert s.model == "gpt-4"
        assert s.provider == "openai"
        assert s.max_iterations == 100

    def test_unknown_keys_ignored(self) -> None:
        data = {"model": "gpt-4", "totally_bogus_key": True, "another_fake": 42}
        s = _settings_from_dict(data)
        assert s.model == "gpt-4"
        assert not hasattr(s, "totally_bogus_key")

    def test_empty_dict_returns_defaults(self) -> None:
        s = _settings_from_dict({})
        assert s.model == ""
        assert s.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_partial_override(self) -> None:
        s = _settings_from_dict({"verbosity": "quiet"})
        assert s.verbosity == "quiet"
        assert s.model == ""  # unspecified → default


# ======================================================================
# _load_env_keys
# ======================================================================

class TestLoadEnvKeys:
    """Verify that _load_env_keys merges environment API keys correctly."""

    def test_env_keys_loaded_when_no_existing_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
        s = Settings()
        _load_env_keys(s)
        assert s.api_keys["openai"] == "sk-test-openai"
        assert s.api_keys["anthropic"] == "sk-test-anthropic"

    def test_env_keys_do_not_overwrite_existing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        s = Settings(api_keys={"openai": "from-config"})
        _load_env_keys(s)
        assert s.api_keys["openai"] == "from-config"

    def test_empty_env_var_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        s = Settings()
        _load_env_keys(s)
        assert "google" not in s.api_keys

    def test_missing_env_var_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        s = Settings()
        _load_env_keys(s)
        assert s.api_keys == {}

    def test_env_key_map_covers_expected_providers(self) -> None:
        expected = {
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "GROQ_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY",
            "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "MOONSHOT_API_KEY",
            "KIMI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY",
            "OLLAMA_API_KEY",
        }
        assert set(_ENV_KEY_MAP.keys()) == expected

    def test_all_env_keys_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var, provider in _ENV_KEY_MAP.items():
            monkeypatch.setenv(env_var, f"key-{provider}")
        s = Settings()
        _load_env_keys(s)
        for provider in _ENV_KEY_MAP.values():
            assert s.api_keys[provider] == f"key-{provider}"


# ======================================================================
# save_keys / key persistence
# ======================================================================

class TestSaveKeys:
    """Verify API key persistence to a separate YAML file."""

    def test_save_and_reload_keys(self, tmp_path: Path) -> None:
        keys_path = tmp_path / "keys.yaml"
        save_keys({"openai": "sk-123", "anthropic": "sk-456"}, path=keys_path)

        with open(keys_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["openai"] == "sk-123"
        assert data["anthropic"] == "sk-456"

    def test_empty_keys_not_saved(self, tmp_path: Path) -> None:
        keys_path = tmp_path / "keys.yaml"
        save_keys({"openai": "", "anthropic": ""}, path=keys_path)
        assert not keys_path.exists()

    def test_key_pool_saved_under_key_pool_key(self, tmp_path: Path) -> None:
        keys_path = tmp_path / "keys.yaml"
        pool = {"openai": ["sk-1", "sk-2", "sk-3"]}
        save_keys({"openai": "sk-1"}, api_key_pool=pool, path=keys_path)

        with open(keys_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "key_pool" in data
        assert data["key_pool"]["openai"] == ["sk-1", "sk-2", "sk-3"]

    def test_empty_pool_not_saved(self, tmp_path: Path) -> None:
        keys_path = tmp_path / "keys.yaml"
        save_keys({"openai": "sk-x"}, api_key_pool={}, path=keys_path)

        with open(keys_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "key_pool" not in data

    def test_save_keys_creates_parent_dirs(self, tmp_path: Path) -> None:
        keys_path = tmp_path / "deep" / "nested" / "keys.yaml"
        save_keys({"google": "gk-abc"}, path=keys_path)
        assert keys_path.exists()

    def test_all_empty_values_skips_write(self, tmp_path: Path) -> None:
        keys_path = tmp_path / "keys.yaml"
        save_keys({}, api_key_pool={}, path=keys_path)
        assert not keys_path.exists()

    def test_pool_with_empty_lists_not_saved(self, tmp_path: Path) -> None:
        keys_path = tmp_path / "keys.yaml"
        save_keys({"openai": "sk-x"}, api_key_pool={"empty_provider": []}, path=keys_path)

        with open(keys_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "key_pool" not in data


# ======================================================================
# load_config / save_config roundtrip
# ======================================================================

class TestLoadSaveConfig:
    """Verify YAML config load/save roundtrip behavior."""

    def test_save_then_load_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear env keys to avoid pollution
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        original = Settings(model="gpt-4o", provider="openai", max_iterations=25)
        save_config(original, config_path)

        # load_config also tries _load_saved_keys with global path — we patch it
        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            loaded = load_config(config_path)

        assert loaded.model == "gpt-4o"
        assert loaded.provider == "openai"
        assert loaded.max_iterations == 25

    def test_load_config_missing_file_returns_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "nonexistent.yaml"
        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            s = load_config(config_path)

        assert s.model == ""
        assert s.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_api_keys_not_in_config_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        s = Settings(api_keys={"openai": "sk-secret"})
        save_config(s, config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "api_keys" not in data

    def test_workspace_not_persisted(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        s = Settings(workspace="/some/path")
        save_config(s, config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "workspace" not in data

    def test_api_key_pool_not_in_config_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        s = Settings(api_key_pool={"openai": ["k1", "k2"]})
        save_config(s, config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "api_key_pool" not in data

    def test_save_config_creates_parent_dirs(self, tmp_path: Path) -> None:
        config_path = tmp_path / "a" / "b" / "config.yaml"
        save_config(Settings(), config_path)
        assert config_path.exists()

    def test_save_config_returns_path(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        result = save_config(Settings(), config_path)
        assert result == config_path

    def test_load_config_merges_saved_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        save_config(Settings(model="test-model"), config_path)

        saved_keys = {"openai": "sk-from-keyfile"}
        saved_pool = {"openai": ["sk-1", "sk-2"]}
        with patch("unjess.config._load_saved_keys", return_value=(saved_keys, saved_pool)):
            loaded = load_config(config_path)

        assert loaded.api_keys["openai"] == "sk-from-keyfile"
        assert loaded.api_key_pool["openai"] == ["sk-1", "sk-2"]

    def test_load_config_env_keys_overlay(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        save_config(Settings(model="m"), config_path)

        monkeypatch.setenv("GROQ_API_KEY", "gk-from-env")
        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            loaded = load_config(config_path)

        assert loaded.api_keys["groq"] == "gk-from-env"

    def test_load_priority_saved_keys_over_config_but_not_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saved keys don't override config keys; env doesn't override either."""
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        # Write config with an openai key already embedded in the raw data
        raw = {"model": "m", "api_keys": {"openai": "from-config"}}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f)

        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        with patch("unjess.config._load_saved_keys", return_value=({"openai": "from-keyfile"}, {})):
            loaded = load_config(config_path)

        # config.yaml sets api_keys.openai, saved_keys won't overwrite, env won't overwrite
        assert loaded.api_keys["openai"] == "from-config"

    def test_load_config_with_unknown_keys_in_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        raw = {"model": "test", "completely_unknown_field": True}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f)

        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            loaded = load_config(config_path)

        assert loaded.model == "test"
        assert not hasattr(loaded, "completely_unknown_field")


# ======================================================================
# ensure_config
# ======================================================================

class TestEnsureConfig:
    """Verify ensure_config creates defaults when missing and is idempotent."""

    def test_creates_config_when_missing(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        result = ensure_config(config_path)
        assert result == config_path
        assert config_path.exists()

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: my-existing-model\n", encoding="utf-8")
        ensure_config(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "my-existing-model" in content

    def test_returns_path(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        result = ensure_config(config_path)
        assert isinstance(result, Path)
        assert result == config_path

    def test_created_config_is_valid_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        ensure_config(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)


# ======================================================================
# is_first_run
# ======================================================================

class TestIsFirstRun:
    """Verify first-run detection logic."""

    def test_first_run_when_no_config_file(self, tmp_path: Path) -> None:
        assert is_first_run(tmp_path / "nope.yaml") is True

    def test_first_run_when_model_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        save_config(Settings(model=""), config_path)

        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            assert is_first_run(config_path) is True

    def test_not_first_run_when_keys_configured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        settings = Settings(model="gpt-4o")
        settings.api_keys["openai"] = "some-key"
        save_config(settings, config_path)

        with patch("unjess.config._load_saved_keys", return_value=({"openai": "some-key"}, {})):
            assert is_first_run(config_path) is False

    def test_not_first_run_when_ollama(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        settings = Settings(provider="ollama", model="qwen3.5:9b")
        save_config(settings, config_path)

        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            assert is_first_run(config_path) is False


# ======================================================================
# apply_cli_overrides
# ======================================================================

class TestApplyCLIOverrides:
    """Verify CLI override application logic."""

    def test_override_model(self) -> None:
        s = Settings()
        result = apply_cli_overrides(s, model="claude-3")
        assert result.model == "claude-3"

    def test_override_multiple_fields(self) -> None:
        s = Settings()
        result = apply_cli_overrides(s, model="test", provider="anthropic", verbosity="quiet")
        assert result.model == "test"
        assert result.provider == "anthropic"
        assert result.verbosity == "quiet"

    def test_none_values_ignored(self) -> None:
        s = Settings(model="original")
        result = apply_cli_overrides(s, model=None, provider=None)
        assert result.model == "original"
        assert result.provider == ""

    def test_unknown_keys_ignored(self) -> None:
        s = Settings()
        result = apply_cli_overrides(s, nonexistent_field="value")
        assert not hasattr(result, "nonexistent_field")

    def test_returns_same_instance(self) -> None:
        s = Settings()
        result = apply_cli_overrides(s, model="m")
        assert result is s

    def test_override_bool_fields(self) -> None:
        s = Settings()
        apply_cli_overrides(s, confirm_commands=False, show_diffs=False)
        assert s.confirm_commands is False
        assert s.show_diffs is False

    def test_override_numeric_fields(self) -> None:
        s = Settings()
        apply_cli_overrides(s, max_iterations=999, command_timeout=120)
        assert s.max_iterations == 999
        assert s.command_timeout == 120

    def test_no_overrides_returns_unchanged(self) -> None:
        s = Settings(model="keep")
        apply_cli_overrides(s)
        assert s.model == "keep"


# ======================================================================
# Workspace config
# ======================================================================

class TestWorkspaceConfig:
    """Verify workspace-scoped settings loading and merging."""

    def test_load_workspace_config_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        result = load_workspace_config(tmp_path)
        assert result == {}

    def test_load_workspace_config_reads_settings_yaml(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        ws_config = agents_dir / "settings.yaml"
        ws_config.write_text("model: ws-model\nverbosity: verbose\n", encoding="utf-8")

        result = load_workspace_config(tmp_path)
        assert result["model"] == "ws-model"
        assert result["verbosity"] == "verbose"

    def test_load_workspace_config_returns_empty_on_invalid_yaml(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        ws_config = agents_dir / "settings.yaml"
        ws_config.write_text(": :\n  {broken: [yaml", encoding="utf-8")

        result = load_workspace_config(tmp_path)
        assert result == {}

    def test_merge_workspace_config_overrides_global(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        ws_config = agents_dir / "settings.yaml"
        ws_config.write_text("model: workspace-model\nmax_iterations: 10\n", encoding="utf-8")

        s = Settings(model="global-model", max_iterations=50)
        result = merge_workspace_config(s, tmp_path)
        assert result.model == "workspace-model"
        assert result.max_iterations == 10

    def test_merge_workspace_config_ignores_unknown_keys(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        ws_config = agents_dir / "settings.yaml"
        ws_config.write_text("unknown_ws_key: 42\n", encoding="utf-8")

        s = Settings()
        result = merge_workspace_config(s, tmp_path)
        assert not hasattr(result, "unknown_ws_key")

    def test_merge_workspace_config_returns_same_instance(self, tmp_path: Path) -> None:
        s = Settings()
        result = merge_workspace_config(s, tmp_path)
        assert result is s

    def test_merge_workspace_config_noop_when_no_ws_config(self, tmp_path: Path) -> None:
        s = Settings(model="keep-this")
        merge_workspace_config(s, tmp_path)
        assert s.model == "keep-this"


# ======================================================================
# resolve_workspace
# ======================================================================

class TestResolveWorkspace:
    """Verify workspace path resolution."""

    def test_relative_dot_resolves_to_cwd(self) -> None:
        s = Settings(workspace=".")
        result = resolve_workspace(s)
        assert result == Path(".").resolve()
        assert result.is_absolute()

    def test_absolute_path_unchanged(self, tmp_path: Path) -> None:
        s = Settings(workspace=str(tmp_path))
        result = resolve_workspace(s)
        assert result == tmp_path.resolve()

    def test_returns_path_object(self) -> None:
        s = Settings(workspace=".")
        result = resolve_workspace(s)
        assert isinstance(result, Path)

    def test_relative_path_resolves(self) -> None:
        s = Settings(workspace="some/relative/path")
        result = resolve_workspace(s)
        assert result.is_absolute()
        assert result == Path("some/relative/path").resolve()


# ======================================================================
# get_os_info / get_shell
# ======================================================================

class TestOSHelpers:
    """Verify OS info and shell detection helpers."""

    def test_get_os_info_returns_string(self) -> None:
        result = get_os_info()
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("unjess.config.platform.system", return_value="Windows")
    def test_get_os_info_windows(self, mock_sys: object) -> None:
        assert get_os_info() == "windows"

    @patch("unjess.config.platform.system", return_value="Darwin")
    def test_get_os_info_macos(self, mock_sys: object) -> None:
        assert get_os_info() == "macOS"

    @patch("unjess.config.platform.system", return_value="Linux")
    def test_get_os_info_linux(self, mock_sys: object) -> None:
        assert get_os_info() == "linux"

    @patch("unjess.config.platform.system", return_value="Windows")
    def test_get_shell_windows(self, mock_sys: object) -> None:
        assert get_shell() == "powershell"

    @patch("unjess.config.platform.system", return_value="Linux")
    def test_get_shell_linux_from_env(self, mock_sys: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/zsh")
        assert get_shell() == "zsh"

    @patch("unjess.config.platform.system", return_value="Linux")
    def test_get_shell_linux_default_bash(self, mock_sys: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SHELL", raising=False)
        assert get_shell() == "bash"

    @patch("unjess.config.platform.system", return_value="Darwin")
    def test_get_shell_macos_from_env(self, mock_sys: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/fish")
        assert get_shell() == "fish"


# ======================================================================
# Edge cases and integration
# ======================================================================

class TestEdgeCases:
    """Integration-level and edge-case tests."""

    def test_full_roundtrip_with_all_fields(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        original = Settings(
            model="gpt-4o",
            provider="openai",
            max_iterations=100,
            subagent_max_turns=20,
            confirm_commands=False,
            command_timeout=60,
            verbosity="verbose",
            show_diffs=False,
            show_cost=False,
            show_stats=False,
            accent_color="#ff0000",
            max_cost_per_session=5.0,
            max_tokens_per_message=4096,
            free_mode_enabled=True,
            enable_rag=True,
            ollama_base_url="http://custom:1234",
            fallback_chain=["anthropic/claude-3", "google/gemini"],
            ignore_patterns=[".git", "node_modules"],
        )
        save_config(original, config_path)

        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            loaded = load_config(config_path)

        assert loaded.model == original.model
        assert loaded.provider == original.provider
        assert loaded.max_iterations == original.max_iterations
        assert loaded.subagent_max_turns == original.subagent_max_turns
        assert loaded.confirm_commands == original.confirm_commands
        assert loaded.command_timeout == original.command_timeout
        assert loaded.verbosity == original.verbosity
        assert loaded.show_diffs == original.show_diffs
        assert loaded.show_cost == original.show_cost
        assert loaded.show_stats == original.show_stats
        assert loaded.accent_color == original.accent_color
        assert loaded.max_cost_per_session == original.max_cost_per_session
        assert loaded.max_tokens_per_message == original.max_tokens_per_message
        assert loaded.free_mode_enabled == original.free_mode_enabled
        assert loaded.enable_rag == original.enable_rag
        assert loaded.ollama_base_url == original.ollama_base_url
        assert loaded.fallback_chain == original.fallback_chain
        assert loaded.ignore_patterns == original.ignore_patterns

    def test_config_yaml_with_empty_content(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            loaded = load_config(config_path)

        assert loaded.model == ""  # defaults

    def test_config_yaml_with_null_content(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in _ENV_KEY_MAP:
            monkeypatch.delenv(env_var, raising=False)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("null\n", encoding="utf-8")

        with patch("unjess.config._load_saved_keys", return_value=({}, {})):
            loaded = load_config(config_path)

        assert loaded.model == ""

    def test_save_config_with_keys_calls_save_keys(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        s = Settings(api_keys={"openai": "sk-test"}, api_key_pool={"openai": ["sk-1"]})

        with patch("unjess.config.save_keys") as mock_sk:
            save_config(s, config_path)
            mock_sk.assert_called_once_with(
                {"openai": "sk-test"},
                api_key_pool={"openai": ["sk-1"]},
            )

    def test_save_config_no_keys_does_not_call_save_keys(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        s = Settings()

        with patch("unjess.config.save_keys") as mock_sk:
            save_config(s, config_path)
            mock_sk.assert_not_called()

    def test_settings_from_dict_with_list_fields(self) -> None:
        data = {"fallback_chain": ["a/b", "c/d"], "ignore_patterns": [".git"]}
        s = _settings_from_dict(data)
        assert s.fallback_chain == ["a/b", "c/d"]
        assert s.ignore_patterns == [".git"]

    def test_settings_from_dict_with_nested_dict_fields(self) -> None:
        data = {"api_keys": {"openai": "sk"}, "api_key_pool": {"openai": ["k1"]}}
        s = _settings_from_dict(data)
        assert s.api_keys == {"openai": "sk"}
        assert s.api_key_pool == {"openai": ["k1"]}
