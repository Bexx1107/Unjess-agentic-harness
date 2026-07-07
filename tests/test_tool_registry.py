"""Tests for unjess.tools — ToolDefinition, ToolRegistry, and argument normalization."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from unjess.tools import ToolDefinition, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_handler(**kwargs: Any) -> str:
    """Handler that echoes its kwargs as a sorted repr."""
    return repr(sorted(kwargs.items()))


def _make_search_schema() -> dict[str, Any]:
    """Schema resembling a grep/search tool with common params."""
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search term"},
            "path": {"type": "string", "description": "Directory to search"},
            "case_insensitive": {"type": "boolean"},
        },
        "required": ["query", "path"],
    }


def _make_file_schema() -> dict[str, Any]:
    """Schema resembling a file-write tool."""
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }


def _make_run_schema() -> dict[str, Any]:
    """Schema resembling a command-run tool."""
    return {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    }


def _populated_registry() -> ToolRegistry:
    """Return a registry with two tools pre-registered."""
    reg = ToolRegistry()
    reg.register("search", "Search files", _make_search_schema(), _dummy_handler)
    reg.register("write_file", "Write a file", _make_file_schema(), _dummy_handler)
    return reg


# ===================================================================
# ToolDefinition
# ===================================================================


class TestToolDefinition:
    """Tests for the ToolDefinition dataclass."""

    def test_to_schema_contains_all_fields(self) -> None:
        td = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=_dummy_handler,
        )
        schema = td.to_schema()
        assert schema["name"] == "read_file"
        assert schema["description"] == "Read a file"
        assert "properties" in schema["parameters"]

    def test_to_schema_excludes_handler(self) -> None:
        td = ToolDefinition(
            name="t", description="d", parameters={}, handler=_dummy_handler,
        )
        schema = td.to_schema()
        assert "handler" not in schema

    def test_to_schema_preserves_parameter_schema_exactly(self) -> None:
        params = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a"],
        }
        td = ToolDefinition(name="t", description="d", parameters=params, handler=_dummy_handler)
        assert td.to_schema()["parameters"] is params  # same object ref


# ===================================================================
# ToolRegistry — registration & retrieval
# ===================================================================


class TestToolRegistryBasics:
    """Tests for register, get_tool, get_tools, tool_names, __len__."""

    def test_register_and_get_tool(self) -> None:
        reg = ToolRegistry()
        reg.register("ping", "Ping", {}, _dummy_handler)
        tool = reg.get_tool("ping")
        assert tool is not None
        assert tool.name == "ping"
        assert tool.handler is _dummy_handler

    def test_get_tool_unknown_returns_none(self) -> None:
        reg = ToolRegistry()
        assert reg.get_tool("no_such_tool") is None

    def test_get_tools_returns_schemas(self) -> None:
        reg = _populated_registry()
        schemas = reg.get_tools()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"search", "write_file"}
        # Each schema must have name, description, parameters
        for s in schemas:
            assert set(s.keys()) == {"name", "description", "parameters"}

    def test_get_tools_empty_registry(self) -> None:
        reg = ToolRegistry()
        assert reg.get_tools() == []

    def test_tool_names_property(self) -> None:
        reg = _populated_registry()
        assert set(reg.tool_names) == {"search", "write_file"}

    def test_tool_names_order_matches_registration(self) -> None:
        reg = ToolRegistry()
        reg.register("b", "B", {}, _dummy_handler)
        reg.register("a", "A", {}, _dummy_handler)
        assert reg.tool_names == ["b", "a"]

    def test_len(self) -> None:
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register("x", "X", {}, _dummy_handler)
        assert len(reg) == 1
        reg.register("y", "Y", {}, _dummy_handler)
        assert len(reg) == 2

    def test_register_overwrites_existing(self) -> None:
        reg = ToolRegistry()
        handler_a = MagicMock(return_value="a")
        handler_b = MagicMock(return_value="b")
        reg.register("t", "first", {}, handler_a)
        reg.register("t", "second", {}, handler_b)
        assert len(reg) == 1
        assert reg.get_tool("t").description == "second"


# ===================================================================
# ToolRegistry — execute()
# ===================================================================


class TestToolRegistryExecute:
    """Tests for execute(), including error handling."""

    def test_execute_calls_handler_with_correct_args(self) -> None:
        handler = MagicMock(return_value="ok")
        reg = ToolRegistry()
        reg.register("greet", "Say hi", {"type": "object", "properties": {"name": {"type": "string"}}}, handler)
        result = reg.execute("greet", {"name": "Alice"})
        assert result == "ok"
        handler.assert_called_once_with(name="Alice")

    def test_execute_unknown_tool_returns_error(self) -> None:
        reg = _populated_registry()
        result = reg.execute("nonexistent", {})
        assert "Error: Unknown tool 'nonexistent'" in result
        assert "search" in result  # lists available tools
        assert "write_file" in result

    def test_execute_type_error_returns_descriptive_error(self) -> None:
        def strict_handler(*, required_param: str) -> str:
            return required_param

        reg = ToolRegistry()
        reg.register("strict", "needs param", {}, strict_handler)
        result = reg.execute("strict", {"wrong_param": "val"})
        assert result.startswith("Error: Invalid arguments for tool 'strict':")

    def test_execute_generic_exception_returns_error(self) -> None:
        def exploding_handler(**kwargs: Any) -> str:
            raise ValueError("boom")

        reg = ToolRegistry()
        reg.register("bomb", "explodes", {}, exploding_handler)
        result = reg.execute("bomb", {})
        assert "Error executing 'bomb'" in result
        assert "ValueError" in result
        assert "boom" in result

    def test_execute_with_empty_arguments(self) -> None:
        handler = MagicMock(return_value="done")
        reg = ToolRegistry()
        reg.register("noop", "No args", {"type": "object", "properties": {}}, handler)
        result = reg.execute("noop", {})
        assert result == "done"
        handler.assert_called_once_with()


# ===================================================================
# _normalize_arguments — fast path
# ===================================================================


class TestNormalizeArgumentsFastPath:
    """Tests for the fast path where all arguments are already valid."""

    def test_all_correct_args_returned_unchanged(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        args = {"query": "hello", "path": "/tmp"}
        result = reg._normalize_arguments(tool, args)
        assert result == {"query": "hello", "path": "/tmp"}

    def test_fast_path_returns_same_object(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        args = {"query": "hello", "path": "/tmp"}
        result = reg._normalize_arguments(tool, args)
        assert result is args  # same dict object — no copy made

    def test_no_schema_properties_returns_args_as_is(self) -> None:
        reg = ToolRegistry()
        reg.register("bare", "No props", {"type": "object"}, _dummy_handler)
        tool = reg.get_tool("bare")
        args = {"anything": "goes"}
        result = reg._normalize_arguments(tool, args)
        assert result is args

    def test_empty_properties_returns_args_as_is(self) -> None:
        reg = ToolRegistry()
        reg.register("bare", "Empty props", {"type": "object", "properties": {}}, _dummy_handler)
        tool = reg.get_tool("bare")
        args = {"foo": "bar"}
        result = reg._normalize_arguments(tool, args)
        # All args are invalid (not in empty properties), but no aliases match either.
        # The corrected dict is returned (a copy), but values are preserved.
        assert result["foo"] == "bar"

    def test_empty_arguments_fast_path(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        args: dict[str, Any] = {}
        result = reg._normalize_arguments(tool, args)
        assert result is args


# ===================================================================
# _normalize_arguments — alias resolution
# ===================================================================


class TestNormalizeAliasResolution:
    """Tests for alias-based argument remapping via _PARAM_ALIASES."""

    def test_pattern_maps_to_query(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(tool, {"pattern": "hello", "path": "/tmp"})
        assert result == {"query": "hello", "path": "/tmp"}

    def test_file_maps_to_path(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(tool, {"query": "x", "file": "/tmp"})
        assert result == {"query": "x", "path": "/tmp"}

    def test_cmd_maps_to_command(self) -> None:
        reg = ToolRegistry()
        reg.register("run", "Run cmd", _make_run_schema(), _dummy_handler)
        tool = reg.get_tool("run")
        result = reg._normalize_arguments(tool, {"cmd": "ls"})
        assert result == {"command": "ls"}

    def test_ignore_case_maps_to_case_insensitive(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(
            tool, {"query": "x", "path": "/", "ignore_case": True}
        )
        assert result == {"query": "x", "path": "/", "case_insensitive": True}

    def test_filepath_maps_to_path(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("write_file")
        result = reg._normalize_arguments(tool, {"filepath": "/f.txt", "content": "hi"})
        assert result == {"path": "/f.txt", "content": "hi"}

    def test_code_maps_to_content(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("write_file")
        result = reg._normalize_arguments(tool, {"path": "/f.txt", "code": "x = 1"})
        assert result == {"path": "/f.txt", "content": "x = 1"}

    def test_shell_command_maps_to_command(self) -> None:
        reg = ToolRegistry()
        reg.register("run", "Run", _make_run_schema(), _dummy_handler)
        tool = reg.get_tool("run")
        result = reg._normalize_arguments(tool, {"shell_command": "echo hi"})
        assert result == {"command": "echo hi"}

    def test_search_query_maps_to_query(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(tool, {"search_query": "hi", "path": "/"})
        assert result == {"query": "hi", "path": "/"}

    def test_directory_maps_to_path(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(tool, {"query": "x", "directory": "/src"})
        assert result == {"query": "x", "path": "/src"}

    def test_folder_maps_to_path(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(tool, {"query": "x", "folder": "/src"})
        assert result == {"query": "x", "path": "/src"}

    def test_text_maps_to_query(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(tool, {"text": "abc", "path": "/"})
        assert result == {"query": "abc", "path": "/"}

    def test_new_code_maps_to_content(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("write_file")
        result = reg._normalize_arguments(tool, {"path": "/f", "new_code": "data"})
        assert result == {"path": "/f", "content": "data"}

    def test_run_alias_maps_to_command(self) -> None:
        reg = ToolRegistry()
        reg.register("exec", "Exec", _make_run_schema(), _dummy_handler)
        tool = reg.get_tool("exec")
        result = reg._normalize_arguments(tool, {"run": "whoami"})
        assert result == {"command": "whoami"}


# ===================================================================
# _normalize_arguments — alias guards
# ===================================================================


class TestNormalizeAliasGuards:
    """Tests that alias remapping only applies when conditions are met."""

    def test_alias_skipped_when_target_already_provided(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        # Both "query" (canonical) and "pattern" (alias) supplied — alias should NOT overwrite
        result = reg._normalize_arguments(
            tool, {"query": "original", "pattern": "should_not_win", "path": "/"}
        )
        assert result["query"] == "original"
        # "pattern" stays as-is (no valid target left)
        assert "pattern" in result

    def test_alias_skipped_when_target_not_in_schema(self) -> None:
        reg = ToolRegistry()
        schema = {
            "type": "object",
            "properties": {
                "something_else": {"type": "string"},
            },
        }
        reg.register("custom", "Custom tool", schema, _dummy_handler)
        tool = reg.get_tool("custom")
        # "pattern" alias targets ["query", "search", "term"] — none in schema
        result = reg._normalize_arguments(tool, {"pattern": "val"})
        assert "pattern" in result
        assert "query" not in result

    def test_alias_picks_first_matching_candidate(self) -> None:
        """When multiple alias targets exist in schema, the first wins."""
        reg = ToolRegistry()
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "search": {"type": "string"},
            },
        }
        reg.register("multi", "Multi", schema, _dummy_handler)
        tool = reg.get_tool("multi")
        # "pattern" -> candidates are ["query", "search", "term"]
        result = reg._normalize_arguments(tool, {"pattern": "val"})
        assert result["query"] == "val"
        assert "search" not in result


# ===================================================================
# _normalize_arguments — case insensitivity
# ===================================================================


class TestNormalizeCaseInsensitivity:
    """Tests that alias lookup is case-insensitive on the bad name."""

    def test_uppercase_alias_resolved(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(tool, {"PATTERN": "hi", "path": "/"})
        assert result == {"query": "hi", "path": "/"}

    def test_mixed_case_alias_resolved(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(tool, {"Ignore_Case": True, "query": "x", "path": "/"})
        assert result == {"case_insensitive": True, "query": "x", "path": "/"}


# ===================================================================
# _normalize_arguments — fallback heuristic
# ===================================================================


class TestNormalizeFallbackHeuristic:
    """Tests for the 'exactly 1 missing required param' fallback."""

    def test_fallback_remaps_to_single_missing_required(self) -> None:
        reg = ToolRegistry()
        reg.register("run", "Run", _make_run_schema(), _dummy_handler)
        tool = reg.get_tool("run")
        # "execute_this" is not a known alias, but "command" is the only missing required
        result = reg._normalize_arguments(tool, {"execute_this": "ls -la"})
        assert result == {"command": "ls -la"}

    def test_fallback_does_not_trigger_when_zero_missing_required(self) -> None:
        reg = ToolRegistry()
        reg.register("run", "Run", _make_run_schema(), _dummy_handler)
        tool = reg.get_tool("run")
        # "command" already provided, "extra" is unknown, 0 missing required → no remap
        result = reg._normalize_arguments(tool, {"command": "ls", "extra": "junk"})
        assert result["command"] == "ls"
        assert result["extra"] == "junk"

    def test_fallback_does_not_trigger_when_two_missing_required(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        # "query" and "path" are both required and both missing; unknown arg present
        result = reg._normalize_arguments(tool, {"totally_unknown": "val"})
        # 2 missing required → fallback doesn't fire
        assert "totally_unknown" in result
        assert "query" not in result
        assert "path" not in result

    def test_fallback_with_partially_correct_args(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        # "query" is correct, "path" is missing, unknown arg present → 1 missing required → fallback
        result = reg._normalize_arguments(tool, {"query": "x", "some_path_arg": "/tmp"})
        assert result == {"query": "x", "path": "/tmp"}

    def test_fallback_does_not_trigger_when_no_required_field(self) -> None:
        reg = ToolRegistry()
        schema = {
            "type": "object",
            "properties": {
                "optional_a": {"type": "string"},
                "optional_b": {"type": "string"},
            },
            # No "required" key at all
        }
        reg.register("opt", "Optional", schema, _dummy_handler)
        tool = reg.get_tool("opt")
        result = reg._normalize_arguments(tool, {"junk": "val"})
        # 0 missing required → no remap
        assert "junk" in result


# ===================================================================
# _normalize_arguments — mixed / integration scenarios
# ===================================================================


class TestNormalizeMixed:
    """Tests combining alias resolution and fallback in one call."""

    def test_multiple_aliases_resolved_at_once(self) -> None:
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(
            tool, {"pattern": "hello", "file": "/tmp", "ignore_case": True}
        )
        assert result == {"query": "hello", "path": "/tmp", "case_insensitive": True}

    def test_alias_and_fallback_combined(self) -> None:
        """One arg resolved via alias, another has no alias and may
        or may not be resolved by fallback depending on iteration order.

        Since ``invalid`` is a set, iteration order is non-deterministic.
        If the non-aliased unknown is processed before the aliases, there
        are >1 missing required params so the fallback heuristic won't fire.
        We verify at least the alias remapping succeeds.
        """
        reg = ToolRegistry()
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query", "path", "limit"],
        }
        reg.register("fancy_search", "Search", schema, _dummy_handler)
        tool = reg.get_tool("fancy_search")

        result = reg._normalize_arguments(
            tool, {"pattern": "hi", "file": "/src", "max_results": 10}
        )
        # Alias remapping always works
        assert result["query"] == "hi"
        assert result["path"] == "/src"
        # max_results may or may not be remapped to limit depending on set order
        assert result.get("limit") == 10 or result.get("max_results") == 10

    def test_correct_args_with_extra_unknown_no_remap(self) -> None:
        """All required already correct, one unknown arg can't be mapped."""
        reg = _populated_registry()
        tool = reg.get_tool("search")
        result = reg._normalize_arguments(
            tool, {"query": "x", "path": "/", "totally_new_arg": "v"}
        )
        # 0 missing required, no alias match → stays
        assert result["query"] == "x"
        assert result["path"] == "/"
        assert result["totally_new_arg"] == "v"


# ===================================================================
# Integration: execute() with normalization
# ===================================================================


class TestExecuteWithNormalization:
    """End-to-end tests that normalization feeds into execute()."""

    def test_execute_normalizes_and_runs(self) -> None:
        handler = MagicMock(return_value="found 3 results")
        reg = ToolRegistry()
        reg.register("search", "Search", _make_search_schema(), handler)
        result = reg.execute("search", {"pattern": "TODO", "file": "/src"})
        assert result == "found 3 results"
        handler.assert_called_once_with(query="TODO", path="/src")

    def test_execute_with_fallback_normalization(self) -> None:
        handler = MagicMock(return_value="ran")
        reg = ToolRegistry()
        reg.register("run", "Run cmd", _make_run_schema(), handler)
        result = reg.execute("run", {"execute_this": "ls"})
        assert result == "ran"
        handler.assert_called_once_with(command="ls")
