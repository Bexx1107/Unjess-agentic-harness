"""Tests for unjess.plugins — plugin discovery and management."""

import json
from pathlib import Path

import pytest

from unjess.plugins import Plugin, PluginManager


# ---------------------------------------------------------------------------
# Plugin dataclass
# ---------------------------------------------------------------------------


class TestPlugin:
    """Tests for the Plugin dataclass and its properties."""

    def test_default_values(self) -> None:
        p = Plugin(name="test")
        assert p.name == "test"
        assert p.description == ""
        assert p.version == ""
        assert p.enabled is True
        assert p.skill_paths == []
        assert p.agent_paths == []
        assert p.config == {}

    def test_has_skills_true(self, tmp_path: Path) -> None:
        p = Plugin(name="t", skill_paths=[tmp_path / "skill1"])
        assert p.has_skills is True

    def test_has_skills_false(self) -> None:
        p = Plugin(name="t")
        assert p.has_skills is False

    def test_has_agents_true(self, tmp_path: Path) -> None:
        p = Plugin(name="t", agent_paths=[tmp_path / "agent.yaml"])
        assert p.has_agents is True

    def test_has_agents_false(self) -> None:
        p = Plugin(name="t")
        assert p.has_agents is False


# ---------------------------------------------------------------------------
# Helper to create plugin directories
# ---------------------------------------------------------------------------


def _make_plugin(
    parent: Path,
    name: str,
    *,
    description: str = "",
    version: str = "1.0",
    enabled: bool = True,
    with_skills: bool = False,
    with_agents: bool = False,
    in_plugins_subdir: bool = False,
) -> Path:
    """Create a plugin directory structure with plugin.json."""
    if in_plugins_subdir:
        base = parent / "plugins"
        base.mkdir(exist_ok=True)
    else:
        base = parent

    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "name": name,
        "description": description,
        "version": version,
        "enabled": enabled,
    }
    (plugin_dir / "plugin.json").write_text(json.dumps(data), encoding="utf-8")

    if with_skills:
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        skill_sub = skills_dir / "my-skill"
        skill_sub.mkdir()
        (skill_sub / "SKILL.md").write_text(
            "---\nname: my-skill\n---\nInstructions\n", encoding="utf-8"
        )

    if with_agents:
        agents_dir = plugin_dir / "agents"
        agents_dir.mkdir()
        (agents_dir / "helper.yaml").write_text("name: helper\n", encoding="utf-8")

    return plugin_dir


# ---------------------------------------------------------------------------
# PluginManager — discovery
# ---------------------------------------------------------------------------


class TestPluginManagerDiscovery:
    """Tests for PluginManager.discover() scanning plugin directories."""

    def test_discover_single_plugin(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "my-plugin", description="A test plugin")
        mgr = PluginManager(plugin_roots=[tmp_path])
        count = mgr.discover()
        assert count == 1
        assert "my-plugin" in mgr.plugin_names

    def test_discover_multiple_plugins(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "alpha", description="Plugin A")
        _make_plugin(tmp_path, "beta", description="Plugin B")
        mgr = PluginManager(plugin_roots=[tmp_path])
        count = mgr.discover()
        assert count == 2

    def test_discover_in_plugins_subdirectory(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "sub-plugin", in_plugins_subdir=True)
        mgr = PluginManager(plugin_roots=[tmp_path])
        count = mgr.discover()
        assert count == 1
        assert "sub-plugin" in mgr.plugin_names

    def test_discover_no_plugins(self, tmp_path: Path) -> None:
        mgr = PluginManager(plugin_roots=[tmp_path])
        count = mgr.discover()
        assert count == 0

    def test_discover_nonexistent_root(self, tmp_path: Path) -> None:
        mgr = PluginManager(plugin_roots=[tmp_path / "no_such_dir"])
        count = mgr.discover()
        assert count == 0

    def test_discover_invalid_json(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "bad-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text("not valid json{{{", encoding="utf-8")
        mgr = PluginManager(plugin_roots=[tmp_path])
        count = mgr.discover()
        assert count == 0

    def test_discover_directory_without_plugin_json(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-plugin").mkdir()
        mgr = PluginManager(plugin_roots=[tmp_path])
        count = mgr.discover()
        assert count == 0

    def test_discover_file_in_root_ignored(self, tmp_path: Path) -> None:
        # Files (not directories) in the root should be skipped
        (tmp_path / "readme.txt").write_text("hi", encoding="utf-8")
        mgr = PluginManager(plugin_roots=[tmp_path])
        count = mgr.discover()
        assert count == 0

    def test_discover_multiple_roots(self, tmp_path: Path) -> None:
        r1 = tmp_path / "root1"
        r1.mkdir()
        r2 = tmp_path / "root2"
        r2.mkdir()
        _make_plugin(r1, "p1")
        _make_plugin(r2, "p2")
        mgr = PluginManager(plugin_roots=[r1, r2])
        count = mgr.discover()
        assert count == 2


# ---------------------------------------------------------------------------
# PluginManager — skill/agent discovery within plugins
# ---------------------------------------------------------------------------


class TestPluginManagerSkillsAgents:
    """Tests for plugin skill and agent path discovery."""

    def test_plugin_with_skills(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "skill-plugin", with_skills=True)
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        p = mgr.get_plugin("skill-plugin")
        assert p is not None
        assert p.has_skills
        assert len(p.skill_paths) == 1

    def test_plugin_with_agents(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "agent-plugin", with_agents=True)
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        p = mgr.get_plugin("agent-plugin")
        assert p is not None
        assert p.has_agents
        assert len(p.agent_paths) == 1
        assert p.agent_paths[0].suffix == ".yaml"

    def test_plugin_without_extras(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "plain")
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        p = mgr.get_plugin("plain")
        assert p is not None
        assert not p.has_skills
        assert not p.has_agents

    def test_single_skill_at_skills_level(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "single-skill"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps({"name": "single-skill"}), encoding="utf-8"
        )
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "SKILL.md").write_text("---\nname: s\n---\nBody\n", encoding="utf-8")
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        p = mgr.get_plugin("single-skill")
        assert p is not None
        assert p.has_skills
        assert len(p.skill_paths) == 1
        assert p.skill_paths[0] == skills_dir

    def test_multiple_agent_formats(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "multi-agent"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps({"name": "multi-agent"}), encoding="utf-8"
        )
        agents_dir = plugin_dir / "agents"
        agents_dir.mkdir()
        (agents_dir / "a.yaml").write_text("name: a\n", encoding="utf-8")
        (agents_dir / "b.yml").write_text("name: b\n", encoding="utf-8")
        (agents_dir / "c.json").write_text('{"name": "c"}', encoding="utf-8")
        (agents_dir / "d.txt").write_text("ignored\n", encoding="utf-8")  # not a valid extension
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        p = mgr.get_plugin("multi-agent")
        assert p is not None
        assert len(p.agent_paths) == 3  # .yaml, .yml, .json — not .txt


# ---------------------------------------------------------------------------
# PluginManager — get_plugin, add_root, properties
# ---------------------------------------------------------------------------


class TestPluginManagerAccess:
    """Tests for get_plugin, add_root, plugins/plugin_names properties."""

    def test_get_plugin_found(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "findme", description="hello")
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        p = mgr.get_plugin("findme")
        assert p is not None
        assert p.description == "hello"

    def test_get_plugin_not_found(self, tmp_path: Path) -> None:
        mgr = PluginManager()
        assert mgr.get_plugin("nope") is None

    def test_plugins_property(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "p1")
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        assert len(mgr.plugins) == 1
        assert mgr.plugins[0].name == "p1"

    def test_plugin_names_property(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "x")
        _make_plugin(tmp_path, "y")
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        assert set(mgr.plugin_names) == {"x", "y"}

    def test_add_root(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _make_plugin(root, "added")
        mgr = PluginManager()
        mgr.add_root(root)
        mgr.discover()
        assert "added" in mgr.plugin_names

    def test_add_root_no_duplicates(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        mgr = PluginManager(plugin_roots=[root])
        mgr.add_root(root)
        assert len(mgr._roots) == 1


# ---------------------------------------------------------------------------
# PluginManager — get_skill_roots
# ---------------------------------------------------------------------------


class TestPluginManagerGetSkillRoots:
    """Tests for get_skill_roots() integration with SkillEngine."""

    def test_returns_skill_parent_dirs(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "sp", with_skills=True)
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        roots = mgr.get_skill_roots()
        assert len(roots) >= 1
        # Each root should be the parent of a skill path
        p = mgr.get_plugin("sp")
        assert p is not None
        for skill_path in p.skill_paths:
            assert skill_path.parent in roots

    def test_disabled_plugin_excluded(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "disabled", enabled=False, with_skills=True)
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        roots = mgr.get_skill_roots()
        assert roots == []

    def test_no_duplicates_in_roots(self, tmp_path: Path) -> None:
        # Two plugins that share the same skill parent
        p1 = _make_plugin(tmp_path, "p1", with_skills=True)
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        roots = mgr.get_skill_roots()
        assert len(roots) == len(set(roots))


# ---------------------------------------------------------------------------
# PluginManager — catalog_for_prompt
# ---------------------------------------------------------------------------


class TestPluginManagerCatalog:
    """Tests for catalog_for_prompt()."""

    def test_catalog_empty(self) -> None:
        mgr = PluginManager()
        assert mgr.catalog_for_prompt() == ""

    def test_catalog_with_plugins(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "alpha", description="Alpha plugin")
        _make_plugin(tmp_path, "beta", description="Beta plugin")
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        catalog = mgr.catalog_for_prompt()
        assert "Installed plugins:" in catalog
        assert "alpha" in catalog
        assert "beta" in catalog

    def test_catalog_shows_skill_count(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "rich", description="Rich plugin", with_skills=True)
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        catalog = mgr.catalog_for_prompt()
        assert "1 skills" in catalog

    def test_catalog_shows_agent_count(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "agent-p", description="Agent plugin", with_agents=True)
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        catalog = mgr.catalog_for_prompt()
        assert "1 agents" in catalog

    def test_catalog_excludes_disabled(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "on", description="Enabled")
        _make_plugin(tmp_path, "off", description="Disabled", enabled=False)
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        catalog = mgr.catalog_for_prompt()
        assert "on" in catalog
        assert "off" not in catalog

    def test_plugin_version_and_config_stored(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "versioned"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps({
                "name": "versioned",
                "version": "2.1.0",
                "config": {"key": "val"},
            }),
            encoding="utf-8",
        )
        mgr = PluginManager(plugin_roots=[tmp_path])
        mgr.discover()
        p = mgr.get_plugin("versioned")
        assert p is not None
        assert p.version == "2.1.0"
        assert p.config == {"key": "val"}
