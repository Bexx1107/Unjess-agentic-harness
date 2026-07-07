"""Project init — scan workspace and generate .unjess/project_context.md.

Creates a rich context document the agent can load on future conversations
to understand the project structure, stack, conventions, and layout.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from unjess.workspace import ProjectInfo, detect_project_type, load_ignore_patterns, should_ignore

logger = logging.getLogger(__name__)

_UNJESS_DIR = ".unjess"
_CONTEXT_FILE = "project_context.md"


def init_project(workspace: Path, force: bool = False) -> Path:
    """Scan the project and generate a context document.

    Creates ``.unjess/project_context.md`` with:
    - Project overview (language, framework, build)
    - Directory structure tree
    - Key file descriptions
    - Detected conventions

    Args:
        workspace: Project root directory.
        force: If True, overwrite existing context file.

    Returns:
        Path to the generated context file.
    """
    unjess_dir = workspace / _UNJESS_DIR
    context_path = unjess_dir / _CONTEXT_FILE

    if context_path.exists() and not force:
        logger.info("Project context already exists at %s", context_path)
        return context_path

    # Detect project info
    info = detect_project_type(workspace)
    ignore_patterns = load_ignore_patterns(workspace)

    # Build the document
    sections: list[str] = []
    sections.append(_header(workspace, info))
    sections.append(_directory_tree(workspace, ignore_patterns))
    sections.append(_key_files(workspace, info))
    sections.append(_conventions(workspace, info))
    sections.append(_dev_commands(info))

    content = "\n\n---\n\n".join(sections)

    # Write
    unjess_dir.mkdir(parents=True, exist_ok=True)
    context_path.write_text(content, encoding="utf-8")
    logger.info("Generated project context at %s", context_path)

    return context_path


def load_project_context(workspace: Path) -> str:
    """Load the project context document if it exists.

    Args:
        workspace: Project root.

    Returns:
        Content of the context file, or empty string.
    """
    context_path = workspace / _UNJESS_DIR / _CONTEXT_FILE
    if context_path.exists():
        try:
            return context_path.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _header(workspace: Path, info: ProjectInfo) -> str:
    """Build the header section."""
    lines = [
        f"# Project: {workspace.name}",
        "",
        f"- **Type**: {info.summary()}",
        f"- **Root**: `{workspace}`",
    ]
    if info.has_git:
        lines.append("- **VCS**: Git")
    if info.test_command:
        lines.append(f"- **Test**: `{info.test_command}`")
    if info.build_tool:
        lines.append(f"- **Build**: `{info.build_tool}`")
    if info.detected_frameworks:
        lines.append(f"- **Frameworks**: {', '.join(info.detected_frameworks)}")

    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    return "\n".join(lines)


def _directory_tree(workspace: Path, ignore_patterns: list[str], max_depth: int = 3) -> str:
    """Build a directory tree section."""
    lines = ["## Directory Structure", "```"]

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return

        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        # Filter ignored entries
        visible = [
            c for c in children
            if not should_ignore(c.relative_to(workspace), ignore_patterns)
            and not c.name.startswith(".")
        ]

        for i, child in enumerate(visible[:30]):  # cap entries per level
            is_last = (i == len(visible) - 1) or (i == 29)
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if child.is_dir():
                lines.append(f"{prefix}{connector}{child.name}/")
                _walk(child, prefix + extension, depth + 1)
            else:
                size = child.stat().st_size
                size_str = _format_size(size)
                lines.append(f"{prefix}{connector}{child.name} ({size_str})")

        if len(visible) > 30:
            lines.append(f"{prefix}... ({len(visible) - 30} more entries)")

    _walk(workspace, "", 0)
    lines.append("```")
    return "\n".join(lines)


def _key_files(workspace: Path, info: ProjectInfo) -> str:
    """Identify and describe key files."""
    lines = ["## Key Files"]

    key_files: dict[str, str] = {}

    # Check for common key files
    candidates: dict[str, str] = {
        "README.md": "Project documentation",
        "README.rst": "Project documentation",
        "pyproject.toml": "Python project configuration",
        "package.json": "Node.js project configuration",
        "Cargo.toml": "Rust project configuration",
        "go.mod": "Go module definition",
        "Dockerfile": "Container configuration",
        "docker-compose.yml": "Docker Compose services",
        "Makefile": "Build automation",
        "requirements.txt": "Python dependencies",
        "setup.py": "Python package setup",
        ".env.example": "Environment variable template",
        "tsconfig.json": "TypeScript configuration",
    }

    for filename, description in candidates.items():
        if (workspace / filename).exists():
            key_files[filename] = description

    # Config files
    for config in workspace.glob("*.config.*"):
        if config.is_file():
            key_files[config.name] = "Configuration file"

    if key_files:
        for filename, description in key_files.items():
            lines.append(f"- **`{filename}`**: {description}")
    else:
        lines.append("*No standard key files detected.*")

    return "\n".join(lines)


def _conventions(workspace: Path, info: ProjectInfo) -> str:
    """Detect and describe project conventions."""
    lines = ["## Conventions"]

    # Source layout
    if (workspace / "src").is_dir():
        lines.append("- **Source layout**: `src/` directory")
    elif (workspace / "lib").is_dir():
        lines.append("- **Source layout**: `lib/` directory")
    elif (workspace / "app").is_dir():
        lines.append("- **Source layout**: `app/` directory")

    # Test layout
    if (workspace / "tests").is_dir():
        lines.append("- **Tests**: `tests/` directory")
    elif (workspace / "test").is_dir():
        lines.append("- **Tests**: `test/` directory")
    elif (workspace / "__tests__").is_dir():
        lines.append("- **Tests**: `__tests__/` directory")
    elif (workspace / "spec").is_dir():
        lines.append("- **Tests**: `spec/` directory")

    # TypeScript
    if info.is_typescript:
        lines.append("- **Language**: TypeScript (with `tsconfig.json`)")

    # Linting / formatting
    linter_files = {
        ".eslintrc.js": "ESLint",
        ".eslintrc.json": "ESLint",
        ".prettierrc": "Prettier",
        "ruff.toml": "Ruff",
        ".flake8": "Flake8",
        "mypy.ini": "Mypy",
        ".editorconfig": "EditorConfig",
    }
    for lfile, lname in linter_files.items():
        if (workspace / lfile).exists():
            lines.append(f"- **Linting**: {lname}")

    if len(lines) == 1:
        lines.append("*No specific conventions detected.*")

    return "\n".join(lines)


def _dev_commands(info: ProjectInfo) -> str:
    """Suggest development commands based on project type."""
    lines = ["## Development Commands"]

    if info.build_tool == "pip":
        lines.append("```bash")
        lines.append("pip install -e .         # Install in dev mode")
        if info.test_command:
            lines.append(f"{info.test_command}                 # Run tests")
        lines.append("```")

    elif info.build_tool == "npm":
        lines.append("```bash")
        lines.append("npm install              # Install dependencies")
        lines.append("npm run dev              # Start dev server")
        lines.append("npm test                 # Run tests")
        lines.append("```")

    elif info.build_tool == "cargo":
        lines.append("```bash")
        lines.append("cargo build              # Build")
        lines.append("cargo test               # Test")
        lines.append("cargo run                # Run")
        lines.append("```")

    elif info.build_tool == "go":
        lines.append("```bash")
        lines.append("go build ./...           # Build")
        lines.append("go test ./...            # Test")
        lines.append("go run .                 # Run")
        lines.append("```")

    else:
        lines.append("*Build commands could not be auto-detected.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_size(size: int) -> str:
    """Format a file size in human-readable form."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"
