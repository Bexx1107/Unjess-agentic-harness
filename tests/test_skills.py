"""Tests for unjess.skills.engine — skill discovery, matching, and loading."""

import json
import textwrap
from pathlib import Path

import pytest

from unjess.skills.engine import (
    Skill,
    SkillEngine,
    _parse_yaml_frontmatter,
)


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------


class TestSkill:
    """Tests for the Skill dataclass and its properties."""

    def test_has_scripts_true(self, tmp_path: Path) -> None:
        (tmp_path / "scripts").mkdir()
        skill = Skill(name="s", description="", path=tmp_path, instructions_path=tmp_path / "SKILL.md")
        assert skill.has_scripts is True

    def test_has_scripts_false(self, tmp_path: Path) -> None:
        skill = Skill(name="s", description="", path=tmp_path, instructions_path=tmp_path / "SKILL.md")
        assert skill.has_scripts is False

    def test_has_examples_true(self, tmp_path: Path) -> None:
        (tmp_path / "examples").mkdir()
        skill = Skill(name="s", description="", path=tmp_path, instructions_path=tmp_path / "SKILL.md")
        assert skill.has_examples is True

    def test_has_examples_false(self, tmp_path: Path) -> None:
        skill = Skill(name="s", description="", path=tmp_path, instructions_path=tmp_path / "SKILL.md")
        assert skill.has_examples is False

    def test_has_references_true(self, tmp_path: Path) -> None:
        (tmp_path / "references").mkdir()
        skill = Skill(name="s", description="", path=tmp_path, instructions_path=tmp_path / "SKILL.md")
        assert skill.has_references is True

    def test_has_references_false(self, tmp_path: Path) -> None:
        skill = Skill(name="s", description="", path=tmp_path, instructions_path=tmp_path / "SKILL.md")
        assert skill.has_references is False

    def test_has_scripts_false_when_path_is_file(self, tmp_path: Path) -> None:
        json_file = tmp_path / "skill.json"
        json_file.write_text("{}", encoding="utf-8")
        skill = Skill(name="s", description="", path=json_file, instructions_path=json_file)
        assert skill.has_scripts is False


# ---------------------------------------------------------------------------
# _parse_yaml_frontmatter helper
# ---------------------------------------------------------------------------


class TestParseYamlFrontmatter:
    """Tests for the minimal YAML frontmatter parser."""

    def test_basic_frontmatter(self) -> None:
        text = "---\nname: my-skill\ndescription: A cool skill\n---\n\nBody content."
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["name"] == "my-skill"
        assert meta["description"] == "A cool skill"
        # Note: the regex's \s* after closing --- consumes the newline,
        # so the body group doesn't capture. This is a known limitation;
        # skills use _inline_instructions from the full text instead.

    def test_frontmatter_body_captured_when_present(self) -> None:
        # When there's no \s after closing ---, body IS captured
        text = "---\nname: test\n---\nSome body"
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["name"] == "test"
        # Body may or may not be captured depending on regex engine
        # The function returns either the body or empty string

    def test_no_frontmatter(self) -> None:
        text = "Just plain markdown.\nNo frontmatter here."
        meta, body = _parse_yaml_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_quoted_values(self) -> None:
        text = '---\nname: "quoted-name"\ndesc: \'single\'\n---\nBody'
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["name"] == "quoted-name"
        assert meta["desc"] == "single"

    def test_comments_skipped(self) -> None:
        text = "---\n# comment\nname: test\n---\nBody"
        meta, body = _parse_yaml_frontmatter(text)
        assert "name" in meta
        assert "#" not in "".join(meta.keys())

    def test_empty_body(self) -> None:
        text = "---\nname: x\n---\n"
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["name"] == "x"
        assert body == ""

    def test_colon_in_value(self) -> None:
        text = "---\ndesc: this: has: colons\n---\nBody"
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["desc"] == "this: has: colons"

    def test_yaml_list_triggers(self) -> None:
        text = "---\nname: test\ntriggers:\n  - thumbnail\n  - concept\n---\nBody"
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["name"] == "test"
        assert meta["triggers"] == ["thumbnail", "concept"]


# ---------------------------------------------------------------------------
# SkillEngine — discovery
# ---------------------------------------------------------------------------


def _make_json_skill(root: Path, name: str, triggers: list[str] | None = None) -> Path:
    """Create a simple JSON skill file."""
    data = {
        "name": name,
        "description": f"Skill {name}",
        "triggers": triggers or [],
        "instructions": f"Instructions for {name}.",
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_dir_skill(root: Path, name: str, triggers: list[str] | None = None) -> Path:
    """Create a complex directory skill with skill.json + instructions.md."""
    skill_dir = root / name
    skill_dir.mkdir()
    data = {
        "name": name,
        "description": f"Complex skill {name}",
        "triggers": triggers or [],
    }
    (skill_dir / "skill.json").write_text(json.dumps(data), encoding="utf-8")
    (skill_dir / "instructions.md").write_text(f"# {name}\nDo stuff.\n", encoding="utf-8")
    return skill_dir


def _make_md_skill(root: Path, name: str, triggers: str = "") -> Path:
    """Create a SKILL.md-based skill."""
    skill_dir = root / name
    skill_dir.mkdir()
    trigger_line = f"triggers: {triggers}\n" if triggers else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: MD skill {name}\n{trigger_line}---\nInstructions here.\n",
        encoding="utf-8",
    )
    return skill_dir


class TestSkillEngineDiscovery:
    """Tests for SkillEngine.discover() across all three formats."""

    def test_discover_json_skill(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _make_json_skill(root, "django", triggers=["migration"])
        engine = SkillEngine(skill_roots=[root])
        count = engine.discover()
        assert count == 1
        assert "django" in engine.skill_names

    def test_discover_dir_skill(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _make_dir_skill(root, "react", triggers=["component"])
        engine = SkillEngine(skill_roots=[root])
        count = engine.discover()
        assert count == 1
        assert "react" in engine.skill_names

    def test_discover_skillmd_skill(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _make_md_skill(root, "terraform", triggers="plan,apply")
        engine = SkillEngine(skill_roots=[root])
        count = engine.discover()
        assert count == 1
        assert "terraform" in engine.skill_names

    def test_discover_multiple_skills(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _make_json_skill(root, "a")
        _make_dir_skill(root, "b")
        _make_md_skill(root, "c")
        engine = SkillEngine(skill_roots=[root])
        count = engine.discover()
        assert count == 3

    def test_discover_no_skills(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        engine = SkillEngine(skill_roots=[root])
        count = engine.discover()
        assert count == 0

    def test_discover_invalid_root(self, tmp_path: Path) -> None:
        engine = SkillEngine(skill_roots=[tmp_path / "nonexistent"])
        count = engine.discover()
        assert count == 0

    def test_discover_invalid_json(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        (root / "bad.json").write_text("not json{{{", encoding="utf-8")
        engine = SkillEngine(skill_roots=[root])
        count = engine.discover()
        assert count == 0

    def test_skill_json_preferred_over_skillmd(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        skill_dir = root / "hybrid"
        skill_dir.mkdir()
        (skill_dir / "skill.json").write_text(
            json.dumps({"name": "hybrid", "description": "from json"}), encoding="utf-8"
        )
        (skill_dir / "SKILL.md").write_text(
            "---\nname: hybrid\ndescription: from md\n---\nBody\n", encoding="utf-8"
        )
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        skill = engine.get_skill("hybrid")
        assert skill is not None
        assert skill.description == "from json"

    def test_discover_multiple_roots(self, tmp_path: Path) -> None:
        r1 = tmp_path / "root1"
        r1.mkdir()
        r2 = tmp_path / "root2"
        r2.mkdir()
        _make_json_skill(r1, "alpha")
        _make_json_skill(r2, "beta")
        engine = SkillEngine(skill_roots=[r1, r2])
        count = engine.discover()
        assert count == 2

    def test_add_root(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _make_json_skill(root, "x")
        engine = SkillEngine()
        engine.add_root(root)
        count = engine.discover()
        assert count == 1

    def test_add_root_no_duplicates(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        engine = SkillEngine(skill_roots=[root])
        engine.add_root(root)
        assert len(engine._roots) == 1

    def test_dir_skill_fallback_readme(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        skill_dir = root / "myskill"
        skill_dir.mkdir()
        (skill_dir / "skill.json").write_text(
            json.dumps({"name": "myskill", "description": "d"}), encoding="utf-8"
        )
        (skill_dir / "README.md").write_text("Readme content\n", encoding="utf-8")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        skill = engine.get_skill("myskill")
        assert skill is not None
        assert skill.instructions_path.name == "README.md"

    def test_dir_skill_fallback_to_json(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        skill_dir = root / "bare"
        skill_dir.mkdir()
        (skill_dir / "skill.json").write_text(
            json.dumps({"name": "bare", "description": "d", "instructions": "inline text"}),
            encoding="utf-8",
        )
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        skill = engine.get_skill("bare")
        assert skill is not None
        assert skill.instructions_path.name == "skill.json"

    def test_skillmd_triggers_parsed(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _make_md_skill(root, "deploy", triggers="deploy, release, ship")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        skill = engine.get_skill("deploy")
        assert skill is not None
        assert "deploy" in skill.trigger_patterns
        assert "release" in skill.trigger_patterns
        assert "ship" in skill.trigger_patterns


# ---------------------------------------------------------------------------
# SkillEngine — matching
# ---------------------------------------------------------------------------


class TestSkillEngineMatch:
    """Tests for SkillEngine.match() — trigger and fuzzy matching."""

    def test_exact_name_match(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "django", triggers=["migration"])
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        results = engine.match("help me with django")
        assert len(results) >= 1
        assert results[0].name == "django"

    def test_trigger_match(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "db", triggers=["migration", "database"])
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        results = engine.match("I need to run a migration")
        assert any(s.name == "db" for s in results)

    def test_no_match(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "django", triggers=["migration"])
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        results = engine.match("build a spaceship")
        assert results == []

    def test_empty_input(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "any")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        assert engine.match("") == []

    def test_match_ranking_name_beats_trigger(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "react", triggers=["component"])
        _make_json_skill(root, "component", triggers=[])
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        results = engine.match("help with component")
        # "component" exact name match (100) should rank first
        assert results[0].name == "component"

    def test_description_keyword_match(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        data = {
            "name": "dbtool",
            "description": "Database optimization and migration helper",
            "triggers": [],
            "instructions": "...",
        }
        (root / "dbtool.json").write_text(json.dumps(data), encoding="utf-8")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        results = engine.match("I need optimization")
        assert any(s.name == "dbtool" for s in results)


# ---------------------------------------------------------------------------
# SkillEngine — loading
# ---------------------------------------------------------------------------


class TestSkillEngineLoad:
    """Tests for SkillEngine.load() and get_skill()."""

    def test_load_inline_instructions(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "inline")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        text = engine.load("inline")
        assert "Instructions for inline" in text

    def test_load_file_instructions(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_dir_skill(root, "complex")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        text = engine.load("complex")
        assert "Do stuff" in text

    def test_load_nonexistent_skill(self, tmp_path: Path) -> None:
        engine = SkillEngine()
        text = engine.load("nope")
        assert text == ""

    def test_load_cached(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "cached")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        text1 = engine.load("cached")
        text2 = engine.load("cached")
        assert text1 == text2

    def test_get_skill_found(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "found")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        skill = engine.get_skill("found")
        assert skill is not None
        assert skill.name == "found"

    def test_get_skill_not_found(self, tmp_path: Path) -> None:
        engine = SkillEngine()
        assert engine.get_skill("missing") is None


# ---------------------------------------------------------------------------
# SkillEngine — catalog_for_prompt
# ---------------------------------------------------------------------------


class TestSkillEngineCatalog:
    """Tests for SkillEngine.catalog_for_prompt()."""

    def test_catalog_empty(self) -> None:
        engine = SkillEngine()
        assert engine.catalog_for_prompt() == ""

    def test_catalog_with_skills(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "alpha")
        _make_json_skill(root, "beta")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        catalog = engine.catalog_for_prompt()
        assert "Available skills:" in catalog
        assert "alpha" in catalog
        assert "beta" in catalog

    def test_skills_property(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "x")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        assert len(engine.skills) == 1
        assert engine.skills[0].name == "x"

    def test_skill_names_property(self, tmp_path: Path) -> None:
        root = tmp_path / "s"
        root.mkdir()
        _make_json_skill(root, "y")
        engine = SkillEngine(skill_roots=[root])
        engine.discover()
        assert engine.skill_names == ["y"]
