"""Tests for unjess.workspace — project detection, ignore patterns, project type info."""

import textwrap
from pathlib import Path

import pytest

from unjess.workspace import (
    ProjectInfo,
    detect_project_root,
    detect_project_type,
    load_ignore_patterns,
    should_ignore,
    _DEFAULT_IGNORE,
    _ROOT_MARKERS,
)


# ---------------------------------------------------------------------------
# ProjectInfo dataclass
# ---------------------------------------------------------------------------


class TestProjectInfo:
    """Tests for the ProjectInfo dataclass and its summary() method."""

    def test_default_values(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path)
        assert info.root == tmp_path
        assert info.language == "unknown"
        assert info.framework == ""
        assert info.build_tool == ""
        assert info.test_command == ""
        assert info.has_git is False
        assert info.is_typescript is False
        assert info.detected_frameworks == []

    def test_summary_unknown_project(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path)
        assert info.summary() == "Unknown project"

    def test_summary_language_only(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, language="python")
        assert info.summary() == "Python"

    def test_summary_with_framework(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, language="python", framework="django")
        assert "Python" in info.summary()
        assert "django" in info.summary()

    def test_summary_with_build_tool(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, language="python", build_tool="pip")
        assert "(pip)" in info.summary()

    def test_summary_typescript_overrides_language(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, language="javascript", is_typescript=True)
        assert info.summary().startswith("TypeScript")

    def test_summary_detected_frameworks_fallback(self, tmp_path: Path) -> None:
        info = ProjectInfo(
            root=tmp_path,
            language="javascript",
            detected_frameworks=["next.js", "tailwind"],
        )
        result = info.summary()
        assert "next.js" in result
        assert "tailwind" in result

    def test_summary_detected_frameworks_max_three(self, tmp_path: Path) -> None:
        info = ProjectInfo(
            root=tmp_path,
            language="python",
            detected_frameworks=["django", "pytest", "docker", "tox"],
        )
        result = info.summary()
        # Should include at most 3 frameworks
        assert "tox" not in result

    def test_summary_framework_takes_priority_over_detected(self, tmp_path: Path) -> None:
        info = ProjectInfo(
            root=tmp_path,
            language="python",
            framework="django",
            detected_frameworks=["flask", "pytest"],
        )
        result = info.summary()
        assert "django" in result
        # detected_frameworks should not appear when framework is set
        assert "flask" not in result


# ---------------------------------------------------------------------------
# detect_project_root
# ---------------------------------------------------------------------------


class TestDetectProjectRoot:
    """Tests for detect_project_root — finding project root from markers."""

    def test_finds_git_marker(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert detect_project_root(nested) == tmp_path

    def test_finds_pyproject_toml(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        sub = tmp_path / "src"
        sub.mkdir()
        assert detect_project_root(sub) == tmp_path

    def test_finds_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        sub = tmp_path / "lib"
        sub.mkdir()
        assert detect_project_root(sub) == tmp_path

    def test_prefers_closest_marker(self, tmp_path: Path) -> None:
        # outer has .git, inner has pyproject.toml — inner should win
        (tmp_path / ".git").mkdir()
        inner = tmp_path / "inner"
        inner.mkdir()
        (inner / "pyproject.toml").write_text("", encoding="utf-8")
        sub = inner / "sub"
        sub.mkdir()
        assert detect_project_root(sub) == inner

    def test_returns_start_path_when_no_marker(self, tmp_path: Path) -> None:
        # No marker files at all
        sub = tmp_path / "orphan"
        sub.mkdir()
        result = detect_project_root(sub)
        assert result == sub.resolve()

    def test_resolves_start_path(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        result = detect_project_root(tmp_path)
        assert result == tmp_path.resolve()

    @pytest.mark.parametrize("marker", _ROOT_MARKERS[:6])
    def test_all_root_markers_are_recognized(self, tmp_path: Path, marker: str) -> None:
        if "." in marker and not marker.startswith("."):
            (tmp_path / marker).write_text("", encoding="utf-8")
        else:
            (tmp_path / marker).mkdir(exist_ok=True)
        sub = tmp_path / "child"
        sub.mkdir()
        assert detect_project_root(sub) == tmp_path


# ---------------------------------------------------------------------------
# detect_project_type
# ---------------------------------------------------------------------------


class TestDetectProjectType:
    """Tests for detect_project_type — detecting language, framework, build tool."""

    def test_detects_python_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.language == "python"
        assert info.build_tool == "pip"

    def test_detects_git(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        info = detect_project_type(tmp_path)
        assert info.has_git is True

    def test_no_git(self, tmp_path: Path) -> None:
        info = detect_project_type(tmp_path)
        assert info.has_git is False

    def test_detects_javascript_npm(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.language == "javascript"
        assert info.build_tool == "npm"

    def test_detects_rust_cargo(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.language == "rust"
        assert info.build_tool == "cargo"

    def test_detects_go(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("module example.com\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.language == "go"
        assert info.build_tool == "go"

    def test_detects_java_maven(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.language == "java"
        assert info.framework == "maven"
        assert info.build_tool == "mvn"

    def test_detects_java_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.language == "java"
        assert info.framework == "gradle"

    def test_detects_c_cmake(self, tmp_path: Path) -> None:
        (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.0)\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.language == "c/c++"
        assert info.build_tool == "cmake"

    def test_detects_typescript(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.is_typescript is True
        assert info.language == "typescript"

    def test_typescript_without_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.is_typescript is True
        # Language stays unknown since no primary signature matched
        assert info.language == "unknown"

    def test_detects_nextjs_framework(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "next.config.js").write_text("module.exports = {}", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.framework == "next.js"
        assert "next.js" in info.detected_frameworks

    def test_detects_django_from_manage_py(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "manage.py").write_text("import django\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert "django" in info.detected_frameworks

    def test_test_command_python(self, tmp_path: Path) -> None:
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.test_command == "pytest"

    def test_test_command_npm(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.test_command == "npm test"

    def test_test_command_cargo(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.test_command == "cargo test"

    def test_detects_fastapi_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["fastapi"]\n', encoding="utf-8"
        )
        info = detect_project_type(tmp_path)
        assert info.framework == "fastapi"

    def test_detects_flask_from_requirements(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("flask>=2.0\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.framework == "flask"

    def test_detects_django_from_requirements(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("django>=4.0\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.framework == "django"

    def test_unknown_project(self, tmp_path: Path) -> None:
        info = detect_project_type(tmp_path)
        assert info.language == "unknown"
        assert info.framework == ""
        assert info.build_tool == ""

    def test_root_is_set(self, tmp_path: Path) -> None:
        info = detect_project_type(tmp_path)
        assert info.root == tmp_path

    def test_multiple_frameworks_detected(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "next.config.js").write_text("", encoding="utf-8")
        (tmp_path / "tailwind.config.js").write_text("", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert "next.js" in info.detected_frameworks
        assert "tailwind" in info.detected_frameworks

    def test_framework_file_does_not_override_primary_framework(self, tmp_path: Path) -> None:
        # pom.xml sets framework to "maven", Dockerfile shouldn't override it
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        (tmp_path / "Dockerfile").write_text("FROM python:3.10\n", encoding="utf-8")
        info = detect_project_type(tmp_path)
        assert info.framework == "maven"
        assert "docker" in info.detected_frameworks


# ---------------------------------------------------------------------------
# load_ignore_patterns / should_ignore
# ---------------------------------------------------------------------------


class TestIgnorePatterns:
    """Tests for load_ignore_patterns and should_ignore."""

    def test_default_ignore_patterns_included(self, tmp_path: Path) -> None:
        patterns = load_ignore_patterns(tmp_path)
        for default in [".git", "__pycache__", "node_modules", "*.pyc"]:
            assert default in patterns

    def test_gitignore_patterns_loaded(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.log\nsecret.txt\n", encoding="utf-8")
        patterns = load_ignore_patterns(tmp_path)
        assert "*.log" in patterns
        assert "secret.txt" in patterns

    def test_agentignore_patterns_loaded(self, tmp_path: Path) -> None:
        (tmp_path / ".agentignore").write_text("big_model.bin\n", encoding="utf-8")
        patterns = load_ignore_patterns(tmp_path)
        assert "big_model.bin" in patterns

    def test_both_gitignore_and_agentignore_merged(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        (tmp_path / ".agentignore").write_text("*.bin\n", encoding="utf-8")
        patterns = load_ignore_patterns(tmp_path)
        assert "*.log" in patterns
        assert "*.bin" in patterns

    def test_comments_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("# comment\n*.log\n", encoding="utf-8")
        patterns = load_ignore_patterns(tmp_path)
        assert "# comment" not in patterns
        assert "*.log" in patterns

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("\n\n*.log\n\n", encoding="utf-8")
        patterns = load_ignore_patterns(tmp_path)
        assert "" not in patterns

    def test_negation_patterns_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("!important.py\n*.log\n", encoding="utf-8")
        patterns = load_ignore_patterns(tmp_path)
        assert "!important.py" not in patterns

    def test_trailing_slash_stripped(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
        patterns = load_ignore_patterns(tmp_path)
        # "build" should be present (without trailing slash); default also has "build"
        assert "build" in patterns

    def test_deduplication(self, tmp_path: Path) -> None:
        # .git is in _DEFAULT_IGNORE and also in .gitignore
        (tmp_path / ".gitignore").write_text(".git\n", encoding="utf-8")
        patterns = load_ignore_patterns(tmp_path)
        assert patterns.count(".git") == 1

    def test_no_ignore_files(self, tmp_path: Path) -> None:
        patterns = load_ignore_patterns(tmp_path)
        # Should still have defaults
        assert len(patterns) == len(set(_DEFAULT_IGNORE))


class TestShouldIgnore:
    """Tests for should_ignore — pattern matching on paths."""

    def test_matches_filename(self) -> None:
        assert should_ignore(Path("foo.pyc"), ["*.pyc"]) is True

    def test_no_match(self) -> None:
        assert should_ignore(Path("foo.py"), ["*.pyc"]) is False

    def test_matches_directory_name(self) -> None:
        assert should_ignore(Path("node_modules/express/index.js"), ["node_modules"]) is True

    def test_matches_path_component(self) -> None:
        assert should_ignore(Path("src/__pycache__/mod.pyc"), ["__pycache__"]) is True

    def test_matches_full_path(self) -> None:
        assert should_ignore(Path("src/secrets.txt"), ["src/secrets.txt"]) is True

    def test_hidden_file_not_matched_without_pattern(self) -> None:
        assert should_ignore(Path(".hidden"), []) is False

    def test_git_directory(self) -> None:
        assert should_ignore(Path(".git"), [".git"]) is True

    def test_nested_pattern_match(self) -> None:
        assert should_ignore(Path("deep/nested/__pycache__/cache.pyc"), ["__pycache__"]) is True

    def test_glob_pattern_match(self) -> None:
        assert should_ignore(Path("debug.log"), ["*.log"]) is True

    def test_empty_patterns(self) -> None:
        assert should_ignore(Path("anything.py"), []) is False
