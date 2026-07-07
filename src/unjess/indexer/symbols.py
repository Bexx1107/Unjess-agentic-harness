"""Symbol extractor — regex-based extraction of functions, classes, and imports.

Uses regular expressions for Phase 4 (fast, no dependencies).
Can be upgraded to tree-sitter AST parsing later.
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """An extracted code symbol."""

    name: str
    kind: str  # "function", "class", "method", "import", "variable", "decorator"
    line: int = 0
    signature: str = ""  # full signature if available
    parent: str = ""  # parent class for methods
    docstring: str = ""  # first line of docstring

    def display(self, indent: int = 0) -> str:
        """Format for repo map display."""
        prefix = "  " * indent
        if self.kind == "class":
            return f"{prefix}class {self.name}:"
        elif self.kind == "method":
            sig = self.signature or f"def {self.name}(...)"
            return f"{prefix}  {sig}"
        elif self.kind == "function":
            sig = self.signature or f"def {self.name}(...)"
            return f"{prefix}{sig}"
        elif self.kind == "import":
            return f"{prefix}import: {self.name}"
        return f"{prefix}{self.name}"


@dataclass
class FileSymbols:
    """All symbols extracted from a single file."""

    path: Path
    relative_path: str = ""
    language: str = ""
    symbols: list[Symbol] = field(default_factory=list)
    line_count: int = 0
    size_bytes: int = 0

    @property
    def classes(self) -> list[Symbol]:
        """All class symbols."""
        return [s for s in self.symbols if s.kind == "class"]

    @property
    def functions(self) -> list[Symbol]:
        """All top-level function symbols."""
        return [s for s in self.symbols if s.kind == "function"]

    @property
    def methods(self) -> list[Symbol]:
        """All method symbols."""
        return [s for s in self.symbols if s.kind == "method"]


# ---------------------------------------------------------------------------
# Language-specific patterns
# ---------------------------------------------------------------------------

# Python
_PY_CLASS = re.compile(r'^class\s+(\w+)\s*[:\(]', re.MULTILINE)
_PY_DEF = re.compile(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?\s*:', re.MULTILINE)
_PY_IMPORT = re.compile(r'^(?:from\s+\S+\s+)?import\s+(.+)$', re.MULTILINE)
_PY_DECORATOR = re.compile(r'^@(\w[\w.]*)', re.MULTILINE)

# JavaScript / TypeScript
_JS_FUNCTION = re.compile(
    r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE
)
_JS_CLASS = re.compile(r'(?:export\s+)?class\s+(\w+)', re.MULTILINE)
_JS_ARROW = re.compile(
    r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(', re.MULTILINE
)
_JS_IMPORT = re.compile(r'import\s+.+\s+from\s+["\']([^"\']+)["\']', re.MULTILINE)

# Rust
_RS_FN = re.compile(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]', re.MULTILINE)
_RS_STRUCT = re.compile(r'(?:pub\s+)?struct\s+(\w+)', re.MULTILINE)
_RS_IMPL = re.compile(r'impl\s+(?:\w+\s+for\s+)?(\w+)', re.MULTILINE)
_RS_TRAIT = re.compile(r'(?:pub\s+)?trait\s+(\w+)', re.MULTILINE)

# Go
_GO_FUNC = re.compile(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(', re.MULTILINE)
_GO_TYPE = re.compile(r'type\s+(\w+)\s+(?:struct|interface)', re.MULTILINE)

# Java / Kotlin
_JAVA_CLASS = re.compile(r'(?:public|private|protected)?\s*(?:abstract\s+)?class\s+(\w+)', re.MULTILINE)
_JAVA_METHOD = re.compile(
    r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(', re.MULTILINE
)

# C / C++
_C_FUNCTION = re.compile(r'^\w[\w\s*]+\s+(\w+)\s*\([^;]*$', re.MULTILINE)
_C_STRUCT = re.compile(r'(?:typedef\s+)?struct\s+(\w+)', re.MULTILINE)

# Map extensions to language
_LANG_MAP: dict[str, str] = {
    ".py": "python", ".pyx": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java", ".kt": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".rb": "ruby", ".php": "php",
    ".cs": "csharp", ".swift": "swift", ".dart": "dart",
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class SymbolExtractor:
    """Extracts symbols from source code files using regex patterns.

    Args:
        workspace: Project root (for computing relative paths).
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    def extract(self, file_path: Path) -> FileSymbols:
        """Extract all symbols from a single file.

        Args:
            file_path: Absolute path to the source file.

        Returns:
            FileSymbols with all discovered symbols.
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return FileSymbols(path=file_path)

        ext = file_path.suffix.lower()
        language = _LANG_MAP.get(ext, "")

        try:
            rel_path = str(file_path.relative_to(self._workspace)).replace("\\", "/")
        except ValueError:
            rel_path = file_path.name

        result = FileSymbols(
            path=file_path,
            relative_path=rel_path,
            language=language,
            line_count=content.count("\n") + 1,
            size_bytes=len(content.encode("utf-8")),
        )

        if language == "python":
            result.symbols = self._extract_python(content)
        elif language in ("javascript", "typescript"):
            result.symbols = self._extract_js_ts(content)
        elif language == "rust":
            result.symbols = self._extract_rust(content)
        elif language == "go":
            result.symbols = self._extract_go(content)
        elif language in ("java", "kotlin"):
            result.symbols = self._extract_java(content)
        elif language in ("c", "cpp"):
            result.symbols = self._extract_c(content)

        return result

    # ----- Python -----

    def _extract_python(self, content: str) -> list[Symbol]:
        symbols: list[Symbol] = []

        for match in _PY_CLASS.finditer(content):
            name = match.group(1)
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=name, kind="class", line=line))

        for match in _PY_DEF.finditer(content):
            indent = match.group(1)
            name = match.group(2)
            params = match.group(3).strip()
            return_type = (match.group(4) or "").strip()
            line = content[:match.start()].count("\n") + 1

            ret = f" -> {return_type}" if return_type else ""
            sig = f"def {name}({params}){ret}"

            if len(indent) >= 4:
                # It's a method — find parent class
                parent = ""
                for s in reversed(symbols):
                    if s.kind == "class" and s.line < line:
                        parent = s.name
                        break
                symbols.append(Symbol(name=name, kind="method", line=line, signature=sig, parent=parent))
            else:
                symbols.append(Symbol(name=name, kind="function", line=line, signature=sig))

        # AST fallback: catch multi-line signatures the regex missed
        symbols = self._ast_fallback_python(content, symbols)
        return symbols

    def _ast_fallback_python(self, content: str, regex_symbols: list[Symbol]) -> list[Symbol]:
        """Find Python functions missed by regex using AST."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return regex_symbols  # Can't parse, keep regex results

        regex_lines = {s.line for s in regex_symbols if s.kind in ('function', 'method')}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno not in regex_lines:
                    # Regex missed this function (likely multi-line sig)
                    parent = ""
                    kind = "function"
                    # Check if it's inside a class
                    for class_sym in regex_symbols:
                        if class_sym.kind == "class" and class_sym.line < node.lineno:
                            parent = class_sym.name
                            kind = "method"

                    # Build signature from AST
                    args = [a.arg for a in node.args.args]
                    returns = ast.unparse(node.returns) if node.returns else ""
                    ret_str = f" -> {returns}" if returns else ""
                    sig = f"def {node.name}({', '.join(args)}){ret_str}"

                    regex_symbols.append(Symbol(
                        name=node.name, kind=kind, line=node.lineno,
                        signature=sig, parent=parent,
                    ))

        # Sort by line number
        regex_symbols.sort(key=lambda s: s.line)
        return regex_symbols

    # ----- JavaScript / TypeScript -----

    def _extract_js_ts(self, content: str) -> list[Symbol]:
        symbols: list[Symbol] = []

        for match in _JS_CLASS.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="class", line=line))

        for match in _JS_FUNCTION.finditer(content):
            name = match.group(1)
            params = match.group(2).strip()
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=name, kind="function", line=line, signature=f"function {name}({params})"))

        for match in _JS_ARROW.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="function", line=line))

        return symbols

    # ----- Rust -----

    def _extract_rust(self, content: str) -> list[Symbol]:
        symbols: list[Symbol] = []

        for match in _RS_STRUCT.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="class", line=line))

        for match in _RS_TRAIT.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="class", line=line))

        for match in _RS_FN.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="function", line=line))

        return symbols

    # ----- Go -----

    def _extract_go(self, content: str) -> list[Symbol]:
        symbols: list[Symbol] = []

        for match in _GO_TYPE.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="class", line=line))

        for match in _GO_FUNC.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="function", line=line))

        return symbols

    # ----- Java -----

    def _extract_java(self, content: str) -> list[Symbol]:
        symbols: list[Symbol] = []

        for match in _JAVA_CLASS.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="class", line=line))

        for match in _JAVA_METHOD.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="method", line=line))

        return symbols

    # ----- C/C++ -----

    def _extract_c(self, content: str) -> list[Symbol]:
        symbols: list[Symbol] = []

        for match in _C_STRUCT.finditer(content):
            line = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(name=match.group(1), kind="class", line=line))

        for match in _C_FUNCTION.finditer(content):
            name = match.group(1)
            if name not in ("if", "for", "while", "switch", "return", "sizeof"):
                line = content[:match.start()].count("\n") + 1
                symbols.append(Symbol(name=name, kind="function", line=line))

        return symbols
