"""Workspace management — project detection, ignore patterns, project type info."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import fnmatch


# ---------------------------------------------------------------------------
# Project root markers (ordered by priority)
# ---------------------------------------------------------------------------

_ROOT_MARKERS: list[str] = [
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "CMakeLists.txt",
    "Makefile",
    "docker-compose.yml",
    ".project",
]

# Map marker files → (language, framework_hint, build_tool)
_PROJECT_SIGNATURES: dict[str, tuple[str, str, str]] = {
    "pyproject.toml":   ("python",     "",           "pip"),
    "setup.py":         ("python",     "",           "pip"),
    "setup.cfg":        ("python",     "",           "pip"),
    "package.json":     ("javascript", "",           "npm"),
    "Cargo.toml":       ("rust",       "",           "cargo"),
    "go.mod":           ("go",         "",           "go"),
    "pom.xml":          ("java",       "maven",      "mvn"),
    "build.gradle":     ("java",       "gradle",     "gradle"),
    "CMakeLists.txt":   ("c/c++",      "cmake",      "cmake"),
}

# Framework detection — filename → framework name
_FRAMEWORK_FILES: dict[str, str] = {
    "next.config.js":     "next.js",
    "next.config.mjs":    "next.js",
    "next.config.ts":     "next.js",
    "nuxt.config.ts":     "nuxt",
    "nuxt.config.js":     "nuxt",
    "vite.config.ts":     "vite",
    "vite.config.js":     "vite",
    "angular.json":       "angular",
    "svelte.config.js":   "svelte",
    "astro.config.mjs":   "astro",
    "remix.config.js":    "remix",
    "tailwind.config.js": "tailwind",
    "tailwind.config.ts": "tailwind",
    "tsconfig.json":      "typescript",
    "manage.py":          "django",
    "app.py":             "flask",
    "fastapi":            "fastapi",
    "Dockerfile":         "docker",
    "flake.nix":          "nix",
    ".flake8":            "flake8",
    "pytest.ini":         "pytest",
    "tox.ini":            "tox",
}

# Test command detection
_TEST_COMMANDS: dict[str, str] = {
    "pytest.ini":    "pytest",
    "pyproject.toml": "pytest",  # most Python projects use pytest
    "setup.cfg":     "pytest",
    "package.json":  "npm test",
    "Cargo.toml":    "cargo test",
    "go.mod":        "go test ./...",
    "pom.xml":       "mvn test",
    "build.gradle":  "gradle test",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProjectInfo:
    """Detected information about the current project."""

    root: Path
    language: str = "unknown"
    framework: str = ""
    build_tool: str = ""
    test_command: str = ""
    has_git: bool = False
    is_typescript: bool = False
    detected_frameworks: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = []
        if self.is_typescript:
            parts.append("TypeScript")
        elif self.language != "unknown":
            parts.append(self.language.capitalize())

        if self.framework:
            parts.append(self.framework)
        elif self.detected_frameworks:
            parts.append(" + ".join(self.detected_frameworks[:3]))

        if self.build_tool:
            parts.append(f"({self.build_tool})")

        return " ".join(parts) if parts else "Unknown project"


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_project_root(start_path: Path) -> Path:
    """Walk up from ``start_path`` looking for project root markers.

    Returns the directory containing the first marker found, or
    ``start_path`` itself if no marker is found.
    """
    current = start_path.resolve()

    while True:
        for marker in _ROOT_MARKERS:
            if (current / marker).exists():
                return current

        parent = current.parent
        if parent == current:
            # Reached filesystem root without finding a marker
            return start_path.resolve()
        current = parent


def detect_project_type(workspace: Path) -> ProjectInfo:
    """Detect the project language, framework, and build system.

    Scans the workspace root for known marker files and config files.

    Args:
        workspace: The project root directory.

    Returns:
        Populated ``ProjectInfo`` dataclass.
    """
    info = ProjectInfo(root=workspace)

    # Check for git
    info.has_git = (workspace / ".git").exists()

    # Primary language/build detection from marker files
    for marker, (lang, fw, build) in _PROJECT_SIGNATURES.items():
        if (workspace / marker).exists():
            if info.language == "unknown":
                info.language = lang
            if fw and not info.framework:
                info.framework = fw
            if build and not info.build_tool:
                info.build_tool = build

    # TypeScript detection
    if (workspace / "tsconfig.json").exists():
        info.is_typescript = True
        if info.language == "javascript":
            info.language = "typescript"

    # Framework detection
    for filename, framework in _FRAMEWORK_FILES.items():
        if (workspace / filename).exists():
            if framework not in info.detected_frameworks:
                info.detected_frameworks.append(framework)
            if not info.framework and framework not in ("typescript", "docker", "tailwind", "pytest", "tox", "flake8", "nix"):
                info.framework = framework

    # Test command detection
    for marker, cmd in _TEST_COMMANDS.items():
        if (workspace / marker).exists():
            info.test_command = cmd
            break

    # Special case: detect FastAPI from requirements/pyproject
    if info.language == "python" and not info.framework:
        info.framework = _detect_python_framework(workspace)

    return info


def _detect_python_framework(workspace: Path) -> str:
    """Try to detect a Python web framework from dependency files."""
    files_to_check = [
        workspace / "pyproject.toml",
        workspace / "requirements.txt",
        workspace / "setup.py",
    ]

    for fpath in files_to_check:
        if not fpath.exists():
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue

        if "fastapi" in content:
            return "fastapi"
        if "flask" in content:
            return "flask"
        if "django" in content:
            return "django"

    return ""


# ---------------------------------------------------------------------------
# Ignore patterns
# ---------------------------------------------------------------------------

_DEFAULT_IGNORE: list[str] = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db", ".env",
    "coverage", ".coverage", "htmlcov",
]


def load_ignore_patterns(workspace: Path) -> list[str]:
    """Load ignore patterns from .gitignore and .agentignore.

    Returns a merged list of glob patterns.
    """
    patterns = list(_DEFAULT_IGNORE)

    # Load .gitignore
    gitignore = workspace / ".gitignore"
    if gitignore.exists():
        patterns.extend(_parse_ignore_file(gitignore))

    # Load .agentignore (takes priority)
    agentignore = workspace / ".agentignore"
    if agentignore.exists():
        patterns.extend(_parse_ignore_file(agentignore))

    return list(set(patterns))  # deduplicate


def _parse_ignore_file(path: Path) -> list[str]:
    """Parse a .gitignore-style file into glob patterns."""
    patterns: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return patterns

    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        # Skip negation patterns (complex to implement)
        if line.startswith("!"):
            continue
        # Strip trailing slash (we treat dirs and files the same)
        patterns.append(line.rstrip("/"))

    return patterns


def should_ignore(path: Path, patterns: list[str]) -> bool:
    """Check whether a path matches any ignore pattern.

    Args:
        path: The path to check (relative to workspace root).
        patterns: List of glob patterns.

    Returns:
        True if the path should be ignored.
    """
    name = path.name
    path_str = str(path).replace("\\", "/")

    for pattern in patterns:
        # Match against filename
        if fnmatch.fnmatch(name, pattern):
            return True
        # Match against full relative path
        if fnmatch.fnmatch(path_str, pattern):
            return True
        # Match if any path component matches
        for part in path.parts:
            if fnmatch.fnmatch(part, pattern):
                return True

    return False
