"""Plugin system — bundles of skills, subagents, and config."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Plugin:
    """A discovered plugin."""

    name: str
    description: str = ""
    path: Path = Path(".")
    version: str = ""
    enabled: bool = True

    # Discovered contents
    skill_paths: list[Path] = field(default_factory=list)
    agent_paths: list[Path] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def has_skills(self) -> bool:
        """Whether this plugin provides skills."""
        return len(self.skill_paths) > 0

    @property
    def has_agents(self) -> bool:
        """Whether this plugin provides subagent definitions."""
        return len(self.agent_paths) > 0


# ---------------------------------------------------------------------------
# Plugin Manager
# ---------------------------------------------------------------------------

class PluginManager:
    """Discovers and manages plugins.

    Plugins are directories containing a ``plugin.json`` file and optionally
    ``skills/`` and ``agents/`` subdirectories.

    Args:
        plugin_roots: Directories to scan for plugins.
    """

    def __init__(self, plugin_roots: list[Path] | None = None) -> None:
        self._roots: list[Path] = plugin_roots or []
        self._plugins: dict[str, Plugin] = {}

    @property
    def plugins(self) -> list[Plugin]:
        """All discovered plugins."""
        return list(self._plugins.values())

    @property
    def plugin_names(self) -> list[str]:
        """Names of all discovered plugins."""
        return list(self._plugins.keys())

    # ----- Discovery -----

    def add_root(self, path: Path) -> None:
        """Add a plugin discovery root directory."""
        if path not in self._roots:
            self._roots.append(path)

    def discover(self) -> int:
        """Scan all roots for plugins.

        Looks for directories containing ``plugin.json``.

        Returns:
            Number of plugins discovered.
        """
        import json

        count = 0
        for root in self._roots:
            if not root.is_dir():
                continue

            # Check root/plugins/ subdirectory first
            plugins_dir = root / "plugins"
            scan_dirs = [plugins_dir] if plugins_dir.is_dir() else [root]

            for scan_dir in scan_dirs:
                if not scan_dir.is_dir():
                    continue

                for child in scan_dir.iterdir():
                    if not child.is_dir():
                        continue

                    plugin_json = child / "plugin.json"
                    if not plugin_json.exists():
                        continue

                    try:
                        data = json.loads(plugin_json.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as exc:
                        logger.warning("Failed to parse %s: %s", plugin_json, exc)
                        continue

                    plugin = self._parse_plugin(child, data)
                    if plugin:
                        self._plugins[plugin.name] = plugin
                        count += 1

        logger.info("Discovered %d plugins from %d roots", count, len(self._roots))
        return count

    def _parse_plugin(self, plugin_dir: Path, data: dict[str, Any]) -> Optional[Plugin]:
        """Parse a plugin from its directory and manifest.

        Args:
            plugin_dir: Plugin root directory.
            data: Parsed plugin.json content.

        Returns:
            Plugin object, or None if invalid.
        """
        name = data.get("name", plugin_dir.name)

        plugin = Plugin(
            name=name,
            description=data.get("description", ""),
            path=plugin_dir,
            version=data.get("version", ""),
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
        )

        # Discover skills
        skills_dir = plugin_dir / "skills"
        if skills_dir.is_dir():
            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    plugin.skill_paths.append(skill_dir)
                elif skill_dir.name == "SKILL.md":
                    # Single skill at skills/ level
                    plugin.skill_paths.append(skills_dir)
                    break

        # Discover agents
        agents_dir = plugin_dir / "agents"
        if agents_dir.is_dir():
            for agent_file in agents_dir.iterdir():
                if agent_file.suffix in (".yaml", ".yml", ".json"):
                    plugin.agent_paths.append(agent_file)

        return plugin

    # ----- Integration -----

    def get_skill_roots(self) -> list[Path]:
        """Get all skill directories from all plugins.

        Returns paths suitable for passing to ``SkillEngine.add_root()``.
        """
        roots: list[Path] = []
        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue
            for skill_path in plugin.skill_paths:
                # Add the parent of the skill dir so the engine can scan it
                if skill_path.parent not in roots:
                    roots.append(skill_path.parent)
        return roots

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    # ----- Context info -----

    def catalog_for_prompt(self) -> str:
        """Build a plugins catalog string for the system prompt."""
        if not self._plugins:
            return ""

        lines = ["Installed plugins:"]
        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue
            extras = []
            if plugin.has_skills:
                extras.append(f"{len(plugin.skill_paths)} skills")
            if plugin.has_agents:
                extras.append(f"{len(plugin.agent_paths)} agents")
            extras_str = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"  - {plugin.name}{extras_str}: {plugin.description}")

        return "\n".join(lines)
