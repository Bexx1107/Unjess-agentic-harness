"""Tests for unjess.project_init — project scanning and context generation."""

import textwrap
from pathlib import Path

import pytest

from unjess.project_init import (
    _format_size,
    _header,
    _key_files,
    _conventions,
    _dev_commands,
    _directory_tree,
    init_project,
    load_project_context,
)
from unjess.workspace import ProjectInfo


# ---------------------------------------------------------------------------
# _format_size
# ---------------------------------------------------------------------------

class TestFormatSize:
    """Unit tests for the _format_size helper."""

    def test_bytes_below_1kb(self) -> None:
        assert _format_size(0) == "0B"
        assert _format_size(512) == "512B"
        assert _format_size(1023) == "1023B"

    def test_kilobytes(self) -> None:
        assert _format_size(1024) == "1.0KB"
        assert _format_size(2048) == "2.0KB"
        assert _format_size(1536) == "1.5KB"

    def test_megabytes(self) -> None:
        assert _format_size(1024 * 1024) == "1.0MB"
        assert _format_size(5 * 1024 * 1024) == "5.0MB"


# ---------------------------------------------------------------------------
# _header
# ---------------------------------------------------------------------------

class TestHeader:
    """Tests for the header section builder."""

    def test_basic_header_contains_project_name(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path)
        header = _header(tmp_path, info)
        assert f"# Project: {tmp_path.name}" in header

    def test_header_includes_git_when_present(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, has_git=True)
        header = _header(tmp_path, info)
        assert "**VCS**: Git" in header

    def test_header_excludes_git_when_absent(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, has_git=False)
        header = _header(tmp_path, info)
        assert "**VCS**: Git" not in header

    def test_header_includes_test_command(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, test_command="pytest")
        header = _header(tmp_path, info)
        assert "`pytest`" in header

    def test_header_includes_build_tool(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, build_tool="npm")
        header = _header(tmp_path, info)
        assert "`npm`" in header

    def test_header_includes_frameworks(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, detected_frameworks=["flask", "pytest"])
        header = _header(tmp_path, info)
        assert "flask" in header
        assert "pytest" in header

    def test_header_includes_generated_timestamp(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path)
        header = _header(tmp_path, info)
        assert "*Generated:" in header


# ---------------------------------------------------------------------------
# _key_files
# ---------------------------------------------------------------------------

class TestKeyFiles:
    """Tests for key file detection section."""

    def test_detects_readme(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
        info = ProjectInfo(root=tmp_path)
        result = _key_files(tmp_path, info)
        assert "README.md" in result
        assert "Project documentation" in result

    def test_detects_pyproject_toml(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
        info = ProjectInfo(root=tmp_path)
        result = _key_files(tmp_path, info)
        assert "pyproject.toml" in result

    def test_detects_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        info = ProjectInfo(root=tmp_path)
        result = _key_files(tmp_path, info)
        assert "package.json" in result

    def test_detects_dockerfile(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text("FROM python:3", encoding="utf-8")
        info = ProjectInfo(root=tmp_path)
        result = _key_files(tmp_path, info)
        assert "Dockerfile" in result

    def test_no_key_files_shows_message(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path)
        result = _key_files(tmp_path, info)
        assert "No standard key files detected" in result

    def test_detects_config_glob_files(self, tmp_path: Path) -> None:
        (tmp_path / "jest.config.js").write_text("", encoding="utf-8")
        info = ProjectInfo(root=tmp_path)
        result = _key_files(tmp_path, info)
        assert "jest.config.js" in result
        assert "Configuration file" in result


# ---------------------------------------------------------------------------
# _conventions
# ---------------------------------------------------------------------------

class TestConventions:
    """Tests for conventions detection section."""

    def test_detects_src_layout(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "`src/` directory" in result

    def test_detects_lib_layout(self, tmp_path: Path) -> None:
        (tmp_path / "lib").mkdir()
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "`lib/` directory" in result

    def test_detects_app_layout(self, tmp_path: Path) -> None:
        (tmp_path / "app").mkdir()
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "`app/` directory" in result

    def test_detects_tests_directory(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "`tests/` directory" in result

    def test_detects_test_directory(self, tmp_path: Path) -> None:
        (tmp_path / "test").mkdir()
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "`test/` directory" in result

    def test_detects_dunder_tests_directory(self, tmp_path: Path) -> None:
        (tmp_path / "__tests__").mkdir()
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "`__tests__/` directory" in result

    def test_detects_spec_directory(self, tmp_path: Path) -> None:
        (tmp_path / "spec").mkdir()
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "`spec/` directory" in result

    def test_detects_typescript(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path, is_typescript=True)
        result = _conventions(tmp_path, info)
        assert "TypeScript" in result

    def test_detects_linter_files(self, tmp_path: Path) -> None:
        (tmp_path / "ruff.toml").write_text("", encoding="utf-8")
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "Ruff" in result

    def test_no_conventions_detected(self, tmp_path: Path) -> None:
        info = ProjectInfo(root=tmp_path)
        result = _conventions(tmp_path, info)
        assert "No specific conventions detected" in result


# ---------------------------------------------------------------------------
# _dev_commands
# ---------------------------------------------------------------------------

class TestDevCommands:
    """Tests for development commands section builder."""

    def test_pip_commands(self) -> None:
        info = ProjectInfo(root=Path("."), build_tool="pip")
        result = _dev_commands(info)
        assert "pip install -e ." in result

    def test_pip_with_test_command(self) -> None:
        info = ProjectInfo(root=Path("."), build_tool="pip", test_command="pytest")
        result = _dev_commands(info)
        assert "pytest" in result

    def test_npm_commands(self) -> None:
        info = ProjectInfo(root=Path("."), build_tool="npm")
        result = _dev_commands(info)
        assert "npm install" in result
        assert "npm run dev" in result
        assert "npm test" in result

    def test_cargo_commands(self) -> None:
        info = ProjectInfo(root=Path("."), build_tool="cargo")
        result = _dev_commands(info)
        assert "cargo build" in result
        assert "cargo test" in result
        assert "cargo run" in result

    def test_go_commands(self) -> None:
        info = ProjectInfo(root=Path("."), build_tool="go")
        result = _dev_commands(info)
        assert "go build" in result
        assert "go test" in result
        assert "go run" in result

    def test_unknown_build_tool(self) -> None:
        info = ProjectInfo(root=Path("."), build_tool="")
        result = _dev_commands(info)
        assert "could not be auto-detected" in result


# ---------------------------------------------------------------------------
# _directory_tree
# ---------------------------------------------------------------------------

class TestDirectoryTree:
    """Tests for the directory tree section builder."""

    def test_tree_shows_files(self, tmp_path: Path) -> None:
        (tmp_path / "hello.py").write_text("print('hi')", encoding="utf-8")
        result = _directory_tree(tmp_path, [])
        assert "hello.py" in result

    def test_tree_shows_dirs_with_slash(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        result = _directory_tree(tmp_path, [])
        assert "subdir/" in result

    def test_tree_respects_ignore_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "keep.py").write_text("", encoding="utf-8")
        result = _directory_tree(tmp_path, ["node_modules"])
        assert "node_modules" not in result
        assert "keep.py" in result

    def test_tree_hides_dotfiles(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("", encoding="utf-8")
        (tmp_path / "visible.py").write_text("", encoding="utf-8")
        result = _directory_tree(tmp_path, [])
        assert ".hidden" not in result
        assert "visible.py" in result

    def test_tree_wrapped_in_code_block(self, tmp_path: Path) -> None:
        result = _directory_tree(tmp_path, [])
        assert "```" in result
        assert "## Directory Structure" in result


# ---------------------------------------------------------------------------
# init_project
# ---------------------------------------------------------------------------

class TestInitProject:
    """Tests for the top-level init_project function."""

    def test_creates_context_file(self, tmp_workspace: Path) -> None:
        result_path = init_project(tmp_workspace)
        assert result_path.exists()
        assert result_path.name == "project_context.md"
        assert result_path.parent.name == ".unjess"

    def test_content_contains_sections(self, tmp_workspace: Path) -> None:
        path = init_project(tmp_workspace)
        content = path.read_text(encoding="utf-8")
        assert "# Project:" in content
        assert "## Directory Structure" in content
        assert "## Key Files" in content
        assert "## Conventions" in content
        assert "## Development Commands" in content

    def test_skip_existing_without_force(self, tmp_workspace: Path) -> None:
        path1 = init_project(tmp_workspace)
        original = path1.read_text(encoding="utf-8")
        path2 = init_project(tmp_workspace, force=False)
        assert path1 == path2
        assert path2.read_text(encoding="utf-8") == original

    def test_overwrite_existing_with_force(self, tmp_workspace: Path) -> None:
        path1 = init_project(tmp_workspace)
        # Tamper with the file
        path1.write_text("tampered", encoding="utf-8")
        path2 = init_project(tmp_workspace, force=True)
        assert path2.read_text(encoding="utf-8") != "tampered"

    def test_creates_unjess_dir_if_missing(self, tmp_path: Path) -> None:
        result_path = init_project(tmp_path)
        assert (tmp_path / ".unjess").is_dir()
        assert result_path.exists()


# ---------------------------------------------------------------------------
# load_project_context
# ---------------------------------------------------------------------------

class TestLoadProjectContext:
    """Tests for loading the context document."""

    def test_returns_content_when_exists(self, tmp_workspace: Path) -> None:
        init_project(tmp_workspace)
        content = load_project_context(tmp_workspace)
        assert content
        assert "# Project:" in content

    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        assert load_project_context(tmp_path) == ""

    def test_returns_empty_on_read_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx_dir = tmp_path / ".unjess"
        ctx_dir.mkdir()
        ctx_file = ctx_dir / "project_context.md"
        ctx_file.write_text("content", encoding="utf-8")

        # Monkey-patch read_text to raise
        original_read_text = Path.read_text

        def broken_read(self_path: Path, *args, **kwargs):
            if self_path == ctx_file:
                raise OSError("Disk error")
            return original_read_text(self_path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", broken_read)
        assert load_project_context(tmp_path) == ""
