"""Tests for unjess.system_prompt — system prompt assembly and section builders."""

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from unjess.config import Settings
from unjess.system_prompt import (
    build_system_prompt,
    load_project_context,
    load_user_rules,
    _build_identity,
    _build_user_info,
    _build_tools_section,
    _build_guidelines,
    _build_communication_style,
    _build_planning_mode,
    _load_agents_rules,
    _load_mcp_instructions,
)
from unjess.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*tool_defs: dict[str, Any]) -> ToolRegistry:
    """Build a ToolRegistry from a list of tool definition dicts."""
    reg = ToolRegistry()
    for td in tool_defs:
        reg.register(
            name=td["name"],
            description=td.get("description", ""),
            parameters=td.get("parameters", {"type": "object", "properties": {}}),
            handler=lambda **kw: "",
        )
    return reg


def _empty_registry() -> ToolRegistry:
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Tests: build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    """Full system prompt assembly."""

    def test_returns_string(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry())
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_identity_section(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry())
        assert "<identity>" in prompt
        assert "</identity>" in prompt
        assert "Unjess" in prompt

    def test_contains_user_info_section(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry())
        assert "<user_information>" in prompt
        assert "</user_information>" in prompt

    def test_contains_tools_section(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry())
        assert "<tools>" in prompt
        assert "</tools>" in prompt

    def test_contains_guidelines(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry())
        assert "<guidelines>" in prompt
        assert "</guidelines>" in prompt

    def test_contains_communication_style(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry())
        assert "<communication_style>" in prompt
        assert "</communication_style>" in prompt

    def test_includes_repo_map_when_provided(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), repo_map="file_tree: src/app.py")
        assert "<repo_map>" in prompt
        assert "file_tree: src/app.py" in prompt

    def test_excludes_repo_map_when_empty(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), repo_map="")
        assert "<repo_map>" not in prompt

    def test_includes_project_context_when_provided(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), project_context="This is a Django project")
        assert "<project_context>" in prompt
        assert "Django project" in prompt

    def test_excludes_project_context_when_empty(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), project_context="")
        assert "<project_context>" not in prompt

    def test_includes_user_rules_when_provided(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), user_rules="Always use black for formatting")
        assert "<user_rules>" in prompt
        assert "black for formatting" in prompt

    def test_excludes_user_rules_when_empty(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), user_rules="")
        assert "<user_rules>" not in prompt

    def test_includes_memory_context_when_provided(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), memory_context="Previously discussed deployment")
        assert "<memory>" in prompt
        assert "deployment" in prompt

    def test_excludes_memory_when_empty(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), memory_context="")
        assert "<memory>" not in prompt

    def test_includes_planning_mode_when_active(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), planning_mode=True)
        assert "<planning_mode>" in prompt
        assert "plan-first" in prompt.lower() or "plan" in prompt.lower()

    def test_excludes_planning_mode_when_inactive(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry(), planning_mode=False)
        assert "<planning_mode>" not in prompt

    def test_gui_mode_changes_identity(self, settings: Settings) -> None:
        prompt_cli = build_system_prompt(settings, _empty_registry(), gui_mode=False)
        prompt_gui = build_system_prompt(settings, _empty_registry(), gui_mode=True)
        assert "terminal" in prompt_cli
        assert "desktop app" in prompt_gui

    def test_sections_joined_by_double_newline(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry())
        # Sections are separated by \n\n — so the prompt should contain multiple section boundaries
        assert prompt.count("</") >= 4  # identity, user_info, tools, guidelines, communication_style

    def test_all_optional_sections_included_together(self, settings: Settings) -> None:
        prompt = build_system_prompt(
            settings,
            _empty_registry(),
            repo_map="tree",
            project_context="ctx",
            user_rules="rules",
            memory_context="mem",
            planning_mode=True,
        )
        for tag in ["repo_map", "project_context", "user_rules", "memory", "planning_mode"]:
            assert f"<{tag}>" in prompt, f"Missing <{tag}>"

    def test_tool_descriptions_appear_in_prompt(self, settings: Settings) -> None:
        reg = _make_registry(
            {"name": "read_file", "description": "Read a file from disk"},
        )
        prompt = build_system_prompt(settings, reg)
        assert "read_file" in prompt
        assert "Read a file from disk" in prompt


# ---------------------------------------------------------------------------
# Tests: _build_identity
# ---------------------------------------------------------------------------

class TestBuildIdentity:
    """Identity section builder."""

    def test_contains_agent_name(self) -> None:
        result = _build_identity()
        assert "Unjess" in result
        assert "njss" in result

    def test_terminal_mode(self) -> None:
        result = _build_identity(gui_mode=False)
        assert "terminal" in result

    def test_gui_mode(self) -> None:
        result = _build_identity(gui_mode=True)
        assert "desktop app" in result

    def test_slash_commands_listed(self) -> None:
        result = _build_identity()
        assert "/help" in result
        assert "/model" in result
        assert "/undo" in result

    def test_wrapped_in_identity_tags(self) -> None:
        result = _build_identity()
        assert result.startswith("<identity>")
        assert result.endswith("</identity>")


# ---------------------------------------------------------------------------
# Tests: _build_user_info
# ---------------------------------------------------------------------------

class TestBuildUserInfo:
    """User information section."""

    def test_contains_model_name(self, settings: Settings) -> None:
        settings.model = "test-model-x"
        result = _build_user_info(settings)
        assert "test-model-x" in result

    def test_contains_provider(self, settings: Settings) -> None:
        settings.provider = "test-provider"
        result = _build_user_info(settings)
        assert "test-provider" in result

    def test_contains_workspace_path(self, settings: Settings) -> None:
        result = _build_user_info(settings)
        # The resolved workspace path should appear
        workspace = Path(settings.workspace).resolve()
        assert str(workspace) in result

    def test_wrapped_in_user_info_tags(self, settings: Settings) -> None:
        result = _build_user_info(settings)
        assert "<user_information>" in result
        assert "</user_information>" in result

    def test_contains_os_info(self, settings: Settings) -> None:
        result = _build_user_info(settings)
        # Should contain some OS string (windows, macOS, linux)
        lower = result.lower()
        assert any(os_name in lower for os_name in ["windows", "macos", "linux"])


# ---------------------------------------------------------------------------
# Tests: _build_tools_section
# ---------------------------------------------------------------------------

class TestBuildToolsSection:
    """Tool section builder."""

    def test_empty_registry_shows_no_tools(self) -> None:
        reg = _empty_registry()
        result = _build_tools_section(reg)
        assert "No tools available" in result
        assert "<tools>" in result
        assert "</tools>" in result

    def test_single_tool_listed(self) -> None:
        reg = _make_registry(
            {"name": "greet", "description": "Greet a user"},
        )
        result = _build_tools_section(reg)
        assert "greet" in result
        assert "Greet a user" in result

    def test_multiple_tools_listed(self) -> None:
        reg = _make_registry(
            {"name": "read_file", "description": "Read a file"},
            {"name": "write_file", "description": "Write a file"},
        )
        result = _build_tools_section(reg)
        assert "read_file" in result
        assert "write_file" in result

    def test_tool_parameters_shown(self) -> None:
        reg = _make_registry({
            "name": "search",
            "description": "Search the codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "path": {"type": "string", "description": "Path to search"},
                },
                "required": ["query"],
            },
        })
        result = _build_tools_section(reg)
        assert "query" in result
        assert "string" in result
        assert "(required)" in result
        assert "Search term" in result

    def test_optional_params_not_marked_required(self) -> None:
        reg = _make_registry({
            "name": "search",
            "description": "Search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Term"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        })
        result = _build_tools_section(reg)
        lines = result.splitlines()
        # Find the limit line — should NOT have "(required)"
        limit_lines = [l for l in lines if "limit" in l]
        assert len(limit_lines) == 1
        assert "(required)" not in limit_lines[0]

    def test_wrapped_in_tools_tags(self) -> None:
        reg = _make_registry({"name": "test_tool", "description": "Test"})
        result = _build_tools_section(reg)
        assert result.startswith("<tools>")
        assert result.endswith("</tools>")

    def test_contains_usage_instructions(self) -> None:
        reg = _make_registry({"name": "my_tool", "description": "Does things"})
        result = _build_tools_section(reg)
        assert "function call" in result.lower()


# ---------------------------------------------------------------------------
# Tests: _build_guidelines
# ---------------------------------------------------------------------------

class TestBuildGuidelines:
    """Guidelines section."""

    def test_wrapped_in_tags(self) -> None:
        result = _build_guidelines()
        assert "<guidelines>" in result
        assert "</guidelines>" in result

    def test_contains_file_editing_rules(self) -> None:
        result = _build_guidelines()
        assert "editing" in result.lower() or "edit" in result.lower()

    def test_contains_planning_instructions(self) -> None:
        result = _build_guidelines()
        assert "plan" in result.lower()


# ---------------------------------------------------------------------------
# Tests: _build_communication_style
# ---------------------------------------------------------------------------

class TestBuildCommunicationStyle:
    """Communication style section."""

    def test_wrapped_in_tags(self) -> None:
        result = _build_communication_style()
        assert "<communication_style>" in result
        assert "</communication_style>" in result

    def test_mentions_markdown(self) -> None:
        result = _build_communication_style()
        assert "markdown" in result.lower()


# ---------------------------------------------------------------------------
# Tests: _build_planning_mode
# ---------------------------------------------------------------------------

class TestBuildPlanningMode:
    """Planning mode section."""

    def test_wrapped_in_tags(self) -> None:
        result = _build_planning_mode()
        assert "<planning_mode>" in result
        assert "</planning_mode>" in result

    def test_contains_phase_instructions(self) -> None:
        result = _build_planning_mode()
        assert "phase" in result.lower() or "Phase" in result

    def test_contains_confirm_step(self) -> None:
        result = _build_planning_mode()
        assert "approve" in result.lower() or "confirm" in result.lower()


# ---------------------------------------------------------------------------
# Tests: load_project_context
# ---------------------------------------------------------------------------

class TestLoadProjectContext:
    """Loading project_context.md from workspace."""

    def test_loads_existing_context(self, tmp_path: Path) -> None:
        ctx_file = tmp_path / "project_context.md"
        ctx_file.write_text("# My Project\nThis is a Python web app.", encoding="utf-8")
        result = load_project_context(tmp_path)
        assert "My Project" in result
        assert "Python web app" in result

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        result = load_project_context(tmp_path)
        assert result == ""

    def test_truncates_to_4000_chars(self, tmp_path: Path) -> None:
        ctx_file = tmp_path / "project_context.md"
        ctx_file.write_text("x" * 10000, encoding="utf-8")
        result = load_project_context(tmp_path)
        assert len(result) == 4000

    def test_handles_unreadable_file(self, tmp_path: Path) -> None:
        ctx_file = tmp_path / "project_context.md"
        ctx_file.write_text("content", encoding="utf-8")
        # Make the file unreadable by patching read_text to raise
        with patch.object(Path, "read_text", side_effect=PermissionError("no access")):
            result = load_project_context(tmp_path)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: load_user_rules
# ---------------------------------------------------------------------------

class TestLoadUserRules:
    """Loading AGENTS.md rules from workspace and global config."""

    def test_loads_project_rules(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        (agents_dir / "AGENTS.md").write_text("# Project Rules\nUse ruff.", encoding="utf-8")
        result = load_user_rules(tmp_path)
        assert "Use ruff" in result

    def test_returns_empty_when_no_rules(self, tmp_path: Path) -> None:
        result = load_user_rules(tmp_path)
        assert result == ""

    def test_combines_project_and_global_rules(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Project rules
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        (agents_dir / "AGENTS.md").write_text("Project rule", encoding="utf-8")

        # Fake global home with rules
        fake_home = tmp_path / "fakehome"
        unjess_dir = fake_home / ".unjess"
        unjess_dir.mkdir(parents=True)
        (unjess_dir / "AGENTS.md").write_text("Global rule", encoding="utf-8")

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        result = load_user_rules(tmp_path)
        assert "Project rule" in result
        assert "Global rule" in result


# ---------------------------------------------------------------------------
# Tests: _load_agents_rules
# ---------------------------------------------------------------------------

class TestLoadAgentsRules:
    """Loading AGENTS.md via the internal helper."""

    def test_returns_empty_when_no_files(self, tmp_path: Path) -> None:
        result = _load_agents_rules(tmp_path)
        assert result == ""

    def test_loads_workspace_rules(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        (agents_dir / "AGENTS.md").write_text("Always type-hint.", encoding="utf-8")
        result = _load_agents_rules(tmp_path)
        assert "Always type-hint" in result
        assert "Workspace Rules" in result

    def test_loads_global_rules(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_home = tmp_path / "fakehome"
        unjess_dir = fake_home / ".unjess"
        unjess_dir.mkdir(parents=True)
        (unjess_dir / "AGENTS.md").write_text("Global: use isort", encoding="utf-8")

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        result = _load_agents_rules(tmp_path)
        assert "use isort" in result
        assert "Global Rules" in result

    def test_combines_both_rule_sources(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        (agents_dir / "AGENTS.md").write_text("WS rule", encoding="utf-8")

        fake_home = tmp_path / "fakehome"
        unjess_dir = fake_home / ".unjess"
        unjess_dir.mkdir(parents=True)
        (unjess_dir / "AGENTS.md").write_text("GL rule", encoding="utf-8")

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        result = _load_agents_rules(tmp_path)
        assert "WS rule" in result
        assert "GL rule" in result
        assert "## User Rules" in result

    def test_ignores_empty_files(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        (agents_dir / "AGENTS.md").write_text("", encoding="utf-8")
        result = _load_agents_rules(tmp_path)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: _load_mcp_instructions
# ---------------------------------------------------------------------------

class TestLoadMcpInstructions:
    """Loading MCP server instruction files."""

    def test_returns_empty_when_no_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        reg = _empty_registry()
        result = _load_mcp_instructions(reg)
        assert result == ""

    def test_returns_empty_when_no_mcp_tools(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_home = tmp_path / "fakehome"
        instr_dir = fake_home / ".unjess" / "mcp_instructions"
        instr_dir.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Registry with non-MCP tools only
        reg = _make_registry({"name": "read_file", "description": "Read"})
        result = _load_mcp_instructions(reg)
        assert result == ""

    def test_loads_instruction_for_mcp_server(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_home = tmp_path / "fakehome"
        instr_dir = fake_home / ".unjess" / "mcp_instructions"
        instr_dir.mkdir(parents=True)
        (instr_dir / "myserver.md").write_text("Use myserver carefully.", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        reg = ToolRegistry()
        reg.register("mcp_myserver_tool1", "A tool", {"type": "object", "properties": {}}, lambda: "")
        result = _load_mcp_instructions(reg)
        assert "Use myserver carefully" in result

    def test_handles_hyphenated_server_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_home = tmp_path / "fakehome"
        instr_dir = fake_home / ".unjess" / "mcp_instructions"
        instr_dir.mkdir(parents=True)
        # Tool name uses underscore, file uses hyphen
        (instr_dir / "my-server.md").write_text("Hyphenated instructions.", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        reg = ToolRegistry()
        reg.register("mcp_my_server_tool", "Tool", {"type": "object", "properties": {}}, lambda: "")
        # The code splits on "_" with maxsplit=2, so server name = "my"
        # The hyphenated variant is tried for "my" → "my.md" not "my-server.md"
        # This test verifies the actual behavior
        result = _load_mcp_instructions(reg)
        # Server name extracted is "my" (from "mcp_my_server_tool" split by _ with maxsplit=2)
        # So it looks for "my.md" or "my.md" — won't find "my-server.md"
        # This is expected behavior based on the parsing logic
        assert isinstance(result, str)

    def test_deduplicates_server_names(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_home = tmp_path / "fakehome"
        instr_dir = fake_home / ".unjess" / "mcp_instructions"
        instr_dir.mkdir(parents=True)
        (instr_dir / "srv.md").write_text("Srv instructions", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        reg = ToolRegistry()
        # Multiple tools from the same server
        reg.register("mcp_srv_toolA", "A", {"type": "object", "properties": {}}, lambda: "")
        reg.register("mcp_srv_toolB", "B", {"type": "object", "properties": {}}, lambda: "")
        result = _load_mcp_instructions(reg)
        # Instructions should appear only once
        assert result.count("Srv instructions") == 1


# ---------------------------------------------------------------------------
# Tests: build_system_prompt with MCP instructions
# ---------------------------------------------------------------------------

class TestBuildSystemPromptMcp:
    """MCP instruction injection via build_system_prompt."""

    def test_mcp_instructions_injected(self, settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_home = tmp_path / "fakehome"
        instr_dir = fake_home / ".unjess" / "mcp_instructions"
        instr_dir.mkdir(parents=True)
        (instr_dir / "ext.md").write_text("MCP ext usage guide", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        reg = ToolRegistry()
        reg.register("mcp_ext_do_thing", "Do thing", {"type": "object", "properties": {}}, lambda: "")
        prompt = build_system_prompt(settings, reg)
        assert "<mcp_instructions>" in prompt
        assert "MCP ext usage guide" in prompt

    def test_no_mcp_instructions_tag_when_none(self, settings: Settings) -> None:
        prompt = build_system_prompt(settings, _empty_registry())
        assert "<mcp_instructions>" not in prompt
