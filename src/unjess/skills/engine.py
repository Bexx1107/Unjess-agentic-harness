"""Skills engine — discovery, trigger matching, and context injection.

Skills use JSON or Markdown for configuration:
  - Simple: a single ``.json`` file with inline instructions
  - Complex: a directory with ``skill.json`` + ``instructions.md``
  - Markdown: a directory with ``SKILL.md`` (YAML frontmatter + body)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    """A discovered skill."""

    name: str
    description: str
    path: Path  # path to the skill directory or .json file
    instructions_path: Path  # path to instructions source
    trigger_patterns: list[str] = field(default_factory=list)

    # Full metadata from skill.json
    metadata: dict[str, Any] = field(default_factory=dict)

    # Inline instructions (from JSON "instructions" field)
    _inline_instructions: str = ""

    @property
    def has_scripts(self) -> bool:
        """Whether this skill has helper scripts."""
        return self.path.is_dir() and (self.path / "scripts").is_dir()

    @property
    def has_examples(self) -> bool:
        """Whether this skill has examples."""
        return self.path.is_dir() and (self.path / "examples").is_dir()

    @property
    def has_references(self) -> bool:
        """Whether this skill has extended documentation."""
        return self.path.is_dir() and (self.path / "references").is_dir()


# ---------------------------------------------------------------------------
# JSON/Markdown parsing helpers
# ---------------------------------------------------------------------------

def _load_skill_json(path: Path) -> dict[str, Any]:
    """Load and parse a skill JSON file.

    Args:
        path: Path to the .json file.

    Returns:
        Parsed dict, or empty dict on failure.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read skill JSON %s: %s", path, exc)
        return {}


def _load_instructions_md(path: Path) -> str:
    """Load markdown instructions from a file.

    Args:
        path: Path to the .md file.

    Returns:
        File content, or empty string on failure.
    """
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _parse_yaml_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown file.

    Expects ``---`` delimiters at the top of the file.

    Args:
        text: Full file content.

    Returns:
        (frontmatter_dict, body_text) — frontmatter may be empty.
    """
    match = re.match(r'^---\s*\n(.*?)\n---\s*(?:\n(.*))?', text, re.DOTALL)
    if not match:
        return {}, text

    yaml_str = match.group(1)
    body = match.group(2) or ""

    # Minimal YAML key: value parsing (avoids requiring PyYAML import here)
    metadata: dict[str, Any] = {}
    for line in yaml_str.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        colon_idx = line.find(':')
        if colon_idx < 0:
            continue
        key = line[:colon_idx].strip()
        val = line[colon_idx + 1:].strip()
        # Strip surrounding quotes
        if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
            val = val[1:-1]
        metadata[key] = val

    return metadata, body


# ---------------------------------------------------------------------------
# Skill Engine
# ---------------------------------------------------------------------------

class SkillEngine:
    """Discovers, matches, and loads skills.

    Supports three skill formats:
        1. **Simple**: A ``.json`` file with ``name``, ``description``,
           ``triggers``, and ``instructions`` fields.
        2. **Complex**: A directory containing ``skill.json`` (metadata)
           and ``instructions.md`` (content), plus optional ``scripts/``,
           ``examples/``, and ``references/`` subdirectories.
        3. **Markdown**: A directory containing ``SKILL.md`` with YAML
           frontmatter (``name``, ``description``) and markdown body.

    Args:
        skill_roots: List of directories to scan for skills.
    """

    def __init__(self, skill_roots: list[Path] | None = None) -> None:
        self._roots: list[Path] = skill_roots or []
        self._skills: dict[str, Skill] = {}
        self._loaded_instructions: dict[str, str] = {}  # cached instruction text

    @property
    def skills(self) -> list[Skill]:
        """All discovered skills."""
        return list(self._skills.values())

    @property
    def skill_names(self) -> list[str]:
        """Names of all discovered skills."""
        return list(self._skills.keys())

    # ----- Discovery -----

    def add_root(self, path: Path) -> None:
        """Add a skill discovery root directory."""
        if path not in self._roots:
            self._roots.append(path)

    def discover(self) -> int:
        """Scan all roots for skills.

        Looks for:
            1. ``<name>.json`` files (simple skill)
            2. Directories containing ``skill.json`` (complex skill)
            3. Directories containing ``SKILL.md`` (markdown skill)

        When both ``skill.json`` and ``SKILL.md`` exist in the same
        directory, ``skill.json`` is preferred for backward compatibility.

        Returns:
            Number of skills discovered.
        """
        count = 0
        for root in self._roots:
            if not root.is_dir():
                continue

            for child in root.iterdir():
                skill: Optional[Skill] = None

                if child.is_dir():
                    # Complex skill: directory with skill.json (preferred)
                    skill_json = child / "skill.json"
                    skill_md = child / "SKILL.md"
                    if skill_json.exists():
                        skill = self._load_dir_skill(child, skill_json)
                    elif skill_md.exists():
                        skill = self._load_skillmd_skill(child, skill_md)

                elif child.suffix == ".json":
                    # Simple skill: standalone .json file
                    skill = self._load_json_skill(child)

                if skill:
                    self._skills[skill.name] = skill
                    count += 1

        logger.info("Discovered %d skills from %d roots", count, len(self._roots))
        return count

    def _load_json_skill(self, json_path: Path) -> Optional[Skill]:
        """Load a simple skill from a standalone .json file.

        Expected format::

            {
                "name": "django",
                "description": "Django project helpers",
                "triggers": ["migration", "makemigrations"],
                "instructions": "When working on Django projects..."
            }

        Args:
            json_path: Path to the .json file.

        Returns:
            Skill object, or None if parsing failed.
        """
        data = _load_skill_json(json_path)
        if not data:
            return None

        name = data.get("name", json_path.stem)
        if not name:
            return None

        return Skill(
            name=name,
            description=data.get("description", ""),
            path=json_path,
            instructions_path=json_path,
            trigger_patterns=data.get("triggers", []),
            metadata=data,
            _inline_instructions=data.get("instructions", ""),
        )

    def _load_dir_skill(self, skill_dir: Path, skill_json: Path) -> Optional[Skill]:
        """Load a complex skill from a directory with skill.json.

        Expected structure::

            my-skill/
                skill.json          # {"name": "...", "description": "...", "triggers": [...]}
                instructions.md     # Markdown instructions
                scripts/            # Optional helper scripts
                examples/           # Optional examples
                references/         # Optional docs

        Args:
            skill_dir: Path to the skill directory.
            skill_json: Path to skill.json.

        Returns:
            Skill object, or None if parsing failed.
        """
        data = _load_skill_json(skill_json)
        if not data:
            return None

        name = data.get("name", skill_dir.name)
        if not name:
            return None

        # Instructions can be in instructions.md or README.md
        instructions_path = skill_dir / "instructions.md"
        if not instructions_path.exists():
            instructions_path = skill_dir / "README.md"
        if not instructions_path.exists():
            instructions_path = skill_json  # fallback to json itself

        return Skill(
            name=name,
            description=data.get("description", ""),
            path=skill_dir,
            instructions_path=instructions_path,
            trigger_patterns=data.get("triggers", []),
            metadata=data,
            _inline_instructions=data.get("instructions", ""),
        )

    def _load_skillmd_skill(self, skill_dir: Path, skill_md: Path) -> Optional[Skill]:
        """Load a skill from a directory with SKILL.md (YAML frontmatter).

        Expected structure::

            my-skill/
                SKILL.md            # YAML frontmatter (name, description) + body
                scripts/            # Optional helper scripts
                examples/           # Optional examples
                references/         # Optional docs

        Args:
            skill_dir: Path to the skill directory.
            skill_md: Path to SKILL.md.

        Returns:
            Skill object, or None if parsing failed.
        """
        raw = _load_instructions_md(skill_md)
        if not raw:
            return None

        frontmatter, body = _parse_yaml_frontmatter(raw)

        name = frontmatter.get("name", skill_dir.name)
        if not name:
            return None

        description = frontmatter.get("description", "")

        # Triggers can be in frontmatter as comma-separated string
        triggers_raw = frontmatter.get("triggers", "")
        if isinstance(triggers_raw, str) and triggers_raw:
            triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()]
        elif isinstance(triggers_raw, list):
            triggers = triggers_raw
        else:
            triggers = []

        return Skill(
            name=name,
            description=description,
            path=skill_dir,
            instructions_path=skill_md,
            trigger_patterns=triggers,
            metadata=frontmatter,
            _inline_instructions=body,
        )

    # ----- Matching -----

    def match(self, user_input: str) -> list[Skill]:
        """Find skills that match the user's input.

        Matching strategy (ordered by priority):
        1. Exact name match (user types the skill name)
        2. Trigger pattern match (keywords from skill description)
        3. Fuzzy match (words from name/description appear in input)

        Args:
            user_input: The user's message.

        Returns:
            List of matching skills, ordered by relevance.
        """
        if not user_input:
            return []

        input_lower = user_input.lower()
        matches: list[tuple[int, Skill]] = []

        for skill in self._skills.values():
            score = 0

            # Exact name match
            if skill.name.lower() in input_lower:
                score += 100

            # Trigger pattern match
            for pattern in skill.trigger_patterns:
                if pattern.lower() in input_lower:
                    score += 50

            # Description keyword match
            if skill.description:
                desc_words = skill.description.lower().split()
                matching_words = sum(1 for w in desc_words if len(w) > 3 and w in input_lower)
                score += matching_words * 10

            if score > 0:
                matches.append((score, skill))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in matches]

    # ----- Loading -----

    def load(self, skill_name: str) -> str:
        """Load a skill's instructions for context injection.

        For JSON skills, returns the ``instructions`` field.
        For directory skills, reads ``instructions.md``.

        Args:
            skill_name: Name of the skill to load.

        Returns:
            The skill's instruction text, or empty string if not found.
        """
        # Check cache
        if skill_name in self._loaded_instructions:
            return self._loaded_instructions[skill_name]

        skill = self._skills.get(skill_name)
        if not skill:
            return ""

        # Prefer inline instructions from JSON
        if skill._inline_instructions:
            self._loaded_instructions[skill_name] = skill._inline_instructions
            return skill._inline_instructions

        # Load from instructions.md
        text = _load_instructions_md(skill.instructions_path)
        self._loaded_instructions[skill_name] = text
        return text

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    # ----- Context info -----

    def catalog_for_prompt(self) -> str:
        """Build a skills catalog string for the system prompt."""
        if not self._skills:
            return ""

        lines = ["Available skills:"]
        for skill in self._skills.values():
            lines.append(f"  - {skill.name}: {skill.description}")

        return "\n".join(lines)
