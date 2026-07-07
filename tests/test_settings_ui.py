"""Tests for unjess.settings_ui — settings display, formatting, and editing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from unjess.config import Settings
from unjess.settings_ui import (
    _all_field_names,
    _format_value,
    _get_field_type,
    _CATEGORIES,
    _VALID_VALUES,
    edit_setting,
    save_and_notify,
    show_settings,
)


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------

class TestFormatValue:
    """Tests for the _format_value display formatter."""

    def test_bool_true_shows_checkmark(self) -> None:
        result = _format_value(True, "bool")
        assert "✓" in result

    def test_bool_false_shows_cross(self) -> None:
        result = _format_value(False, "bool")
        assert "✗" in result

    def test_float_zero_shows_unlimited(self) -> None:
        result = _format_value(0.0, "float")
        assert "unlimited" in result

    def test_float_with_cost_in_name_shows_dollar(self) -> None:
        result = _format_value(5.50, "float", field_name="max_cost_per_session")
        assert "$5.50" in result

    def test_float_without_cost_shows_plain(self) -> None:
        result = _format_value(3.14, "float", field_name="some_ratio")
        assert "3.14" in result

    def test_int_zero_shows_unlimited(self) -> None:
        result = _format_value(0, "int")
        assert "unlimited" in result

    def test_int_nonzero_shows_value(self) -> None:
        result = _format_value(42, "int")
        assert "42" in result

    def test_empty_string_shows_auto(self) -> None:
        result = _format_value("", "str")
        assert "auto" in result

    def test_nonempty_string_shows_value(self) -> None:
        result = _format_value("gpt-4o", "str")
        assert "gpt-4o" in result

    def test_bool_takes_priority_over_int_zero(self) -> None:
        # False is also 0, but isinstance(False, bool) is True and should
        # be checked before isinstance(False, int).
        result = _format_value(False, "bool")
        assert "✗" in result
        assert "unlimited" not in result


# ---------------------------------------------------------------------------
# _all_field_names / _get_field_type
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for helper functions that query the category registry."""

    def test_all_field_names_returns_known_fields(self) -> None:
        names = _all_field_names()
        assert "model" in names
        assert "provider" in names
        assert "max_iterations" in names
        assert "verbosity" in names

    def test_all_field_names_no_duplicates(self) -> None:
        names = _all_field_names()
        assert len(names) == len(set(names))

    def test_get_field_type_known(self) -> None:
        assert _get_field_type("model") == "str"
        assert _get_field_type("max_iterations") == "int"
        assert _get_field_type("confirm_commands") == "bool"
        assert _get_field_type("max_cost_per_session") == "float"

    def test_get_field_type_unknown_defaults_to_str(self) -> None:
        assert _get_field_type("nonexistent_field_xyz") == "str"


# ---------------------------------------------------------------------------
# edit_setting — happy paths
# ---------------------------------------------------------------------------

class TestEditSettingHappy:
    """Tests for successful edit_setting calls."""

    def test_edit_string_setting(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "model", "gpt-4o")
        assert ok is True
        assert s.model == "gpt-4o"
        assert "→" in msg

    def test_edit_int_setting(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "max_iterations", "100")
        assert ok is True
        assert s.max_iterations == 100

    def test_edit_bool_setting_true_variants(self) -> None:
        for val in ("true", "1", "yes", "on", "True", "YES"):
            s = Settings(confirm_commands=False)
            ok, _ = edit_setting(s, "confirm_commands", val)
            assert ok is True
            assert s.confirm_commands is True

    def test_edit_bool_setting_false_variants(self) -> None:
        for val in ("false", "0", "no", "off"):
            s = Settings(confirm_commands=True)
            ok, _ = edit_setting(s, "confirm_commands", val)
            assert ok is True
            assert s.confirm_commands is False

    def test_edit_float_setting(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "max_cost_per_session", "9.99")
        assert ok is True
        assert s.max_cost_per_session == pytest.approx(9.99)

    def test_edit_verbosity_valid(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "verbosity", "verbose")
        assert ok is True
        assert s.verbosity == "verbose"

    def test_edit_verbosity_case_insensitive(self) -> None:
        s = Settings()
        ok, _ = edit_setting(s, "verbosity", "QUIET")
        assert ok is True
        assert s.verbosity == "quiet"

    def test_message_shows_old_and_new_value(self) -> None:
        s = Settings(model="old-model")
        ok, msg = edit_setting(s, "model", "new-model")
        assert ok is True
        assert "old-model" in msg
        assert "new-model" in msg


# ---------------------------------------------------------------------------
# edit_setting — error cases
# ---------------------------------------------------------------------------

class TestEditSettingErrors:
    """Tests for edit_setting validation and error cases."""

    def test_unknown_key_returns_error(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "nonexistent_xyz", "value")
        assert ok is False
        assert "Unknown setting" in msg

    def test_unknown_key_suggests_close_match(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "model", "value")  # 'model' exists, so use a near miss
        # model exists so it should succeed; test with a partial match
        ok2, msg2 = edit_setting(s, "models", "value")
        assert ok2 is False
        assert "model" in msg2.lower()

    def test_invalid_verbosity_value(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "verbosity", "super_loud")
        assert ok is False
        assert "Must be one of" in msg

    def test_invalid_int_value(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "max_iterations", "not_a_number")
        assert ok is False
        assert "expected int" in msg

    def test_invalid_float_value(self) -> None:
        s = Settings()
        ok, msg = edit_setting(s, "max_cost_per_session", "abc")
        assert ok is False
        assert "expected float" in msg


# ---------------------------------------------------------------------------
# show_settings
# ---------------------------------------------------------------------------

class TestShowSettings:
    """Tests for show_settings display output."""

    def test_show_settings_does_not_crash(self) -> None:
        s = Settings()
        console = Console(file=MagicMock(), width=120)
        show_settings(s, console)  # should not raise

    def test_show_settings_with_custom_values(self) -> None:
        s = Settings(model="gpt-4o", verbosity="verbose", confirm_commands=False)
        console = Console(file=MagicMock(), width=120)
        show_settings(s, console)


# ---------------------------------------------------------------------------
# save_and_notify
# ---------------------------------------------------------------------------

class TestSaveAndNotify:
    """Tests for persisting settings with confirmation."""

    def test_returns_confirmation_message(self, tmp_path: Path) -> None:
        s = Settings()
        with patch("unjess.settings_ui.save_config", return_value=tmp_path / "config.yaml"):
            msg = save_and_notify(s)
        assert "Settings saved" in msg
        assert "config.yaml" in msg


# ---------------------------------------------------------------------------
# Categories structure
# ---------------------------------------------------------------------------

class TestCategories:
    """Tests ensuring the category registry is well-formed."""

    def test_all_category_fields_exist_on_settings(self) -> None:
        s = Settings()
        for category, fields in _CATEGORIES.items():
            for field_name, field_type, description in fields:
                assert hasattr(s, field_name), (
                    f"Settings missing field '{field_name}' from category '{category}'"
                )

    def test_valid_values_keys_are_known_fields(self) -> None:
        all_names = _all_field_names()
        for key in _VALID_VALUES:
            assert key in all_names, f"_VALID_VALUES key '{key}' not in _CATEGORIES"
