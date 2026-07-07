"""Tests for unjess.indexer — file scanner and symbol extractor."""

import textwrap
from pathlib import Path

import pytest

from unjess.indexer.scanner import FileScanner, _CODE_EXTENSIONS, _MAX_FILE_SIZE
from unjess.indexer.symbols import (
    FileSymbols,
    Symbol,
    SymbolExtractor,
    _LANG_MAP,
)


# ---------------------------------------------------------------------------
# Symbol dataclass
# ---------------------------------------------------------------------------


class TestSymbol:
    """Tests for the Symbol dataclass and its display() method."""

    def test_display_class(self) -> None:
        s = Symbol(name="MyClass", kind="class", line=1)
        assert s.display() == "class MyClass:"

    def test_display_function_with_signature(self) -> None:
        s = Symbol(name="foo", kind="function", line=5, signature="def foo(x: int) -> str")
        assert s.display() == "def foo(x: int) -> str"

    def test_display_function_without_signature(self) -> None:
        s = Symbol(name="bar", kind="function", line=3)
        assert s.display() == "def bar(...)"

    def test_display_method_with_signature(self) -> None:
        s = Symbol(name="do_stuff", kind="method", line=10, signature="def do_stuff(self)")
        assert s.display() == "  def do_stuff(self)"

    def test_display_method_without_signature(self) -> None:
        s = Symbol(name="m", kind="method", line=8)
        assert s.display() == "  def m(...)"

    def test_display_import(self) -> None:
        s = Symbol(name="os", kind="import", line=1)
        assert s.display() == "import: os"

    def test_display_other_kind(self) -> None:
        s = Symbol(name="MY_VAR", kind="variable", line=2)
        assert s.display() == "MY_VAR"

    def test_display_indent(self) -> None:
        s = Symbol(name="Cls", kind="class", line=1)
        assert s.display(indent=2) == "    class Cls:"


# ---------------------------------------------------------------------------
# FileSymbols dataclass
# ---------------------------------------------------------------------------


class TestFileSymbols:
    """Tests for the FileSymbols dataclass and its property filters."""

    def test_classes_filter(self, tmp_path: Path) -> None:
        fs = FileSymbols(
            path=tmp_path / "test.py",
            symbols=[
                Symbol(name="A", kind="class"),
                Symbol(name="b", kind="function"),
                Symbol(name="C", kind="class"),
            ],
        )
        assert [s.name for s in fs.classes] == ["A", "C"]

    def test_functions_filter(self, tmp_path: Path) -> None:
        fs = FileSymbols(
            path=tmp_path / "test.py",
            symbols=[
                Symbol(name="f1", kind="function"),
                Symbol(name="A", kind="class"),
                Symbol(name="f2", kind="function"),
            ],
        )
        assert [s.name for s in fs.functions] == ["f1", "f2"]

    def test_methods_filter(self, tmp_path: Path) -> None:
        fs = FileSymbols(
            path=tmp_path / "test.py",
            symbols=[
                Symbol(name="m1", kind="method"),
                Symbol(name="f1", kind="function"),
                Symbol(name="m2", kind="method"),
            ],
        )
        assert [s.name for s in fs.methods] == ["m1", "m2"]

    def test_empty_symbols(self, tmp_path: Path) -> None:
        fs = FileSymbols(path=tmp_path / "empty.py")
        assert fs.classes == []
        assert fs.functions == []
        assert fs.methods == []


# ---------------------------------------------------------------------------
# FileScanner
# ---------------------------------------------------------------------------


class TestFileScanner:
    """Tests for FileScanner — walking workspace files with filtering."""

    def test_scan_finds_python_files(self, tmp_workspace: Path) -> None:
        scanner = FileScanner(tmp_workspace)
        files = list(scanner.scan())
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) >= 3  # app.py, utils.py, test_app.py

    def test_scan_finds_markdown(self, tmp_workspace: Path) -> None:
        scanner = FileScanner(tmp_workspace)
        files = list(scanner.scan())
        md_files = [f for f in files if f.suffix == ".md"]
        assert len(md_files) >= 1  # README.md

    def test_scan_finds_toml(self, tmp_workspace: Path) -> None:
        scanner = FileScanner(tmp_workspace)
        files = list(scanner.scan())
        toml_files = [f for f in files if f.suffix == ".toml"]
        assert len(toml_files) >= 1  # pyproject.toml

    def test_scan_skips_hidden_files(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "visible.py").write_text("y = 2\n", encoding="utf-8")
        scanner = FileScanner(tmp_path)
        files = list(scanner.scan())
        names = [f.name for f in files]
        assert "visible.py" in names
        assert ".hidden.py" not in names

    def test_scan_skips_hidden_directories(self, tmp_path: Path) -> None:
        hidden_dir = tmp_path / ".mydir"
        hidden_dir.mkdir()
        (hidden_dir / "code.py").write_text("x = 1\n", encoding="utf-8")
        scanner = FileScanner(tmp_path)
        files = list(scanner.scan())
        assert not any(f.name == "code.py" for f in files)

    def test_scan_skips_ignored_directories(self, tmp_path: Path) -> None:
        node = tmp_path / "node_modules"
        node.mkdir()
        (node / "pkg.js").write_text("module.exports = {}\n", encoding="utf-8")
        (tmp_path / "app.js").write_text("console.log('hi')\n", encoding="utf-8")
        scanner = FileScanner(tmp_path)
        files = list(scanner.scan())
        names = [f.name for f in files]
        assert "app.js" in names
        assert "pkg.js" not in names

    def test_scan_skips_non_code_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "data.dat").write_text("binary-ish\n", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
        scanner = FileScanner(tmp_path)
        files = list(scanner.scan())
        names = [f.name for f in files]
        assert "code.py" in names
        assert "data.dat" not in names
        assert "image.png" not in names

    def test_scan_skips_large_files(self, tmp_path: Path) -> None:
        big_file = tmp_path / "big.py"
        big_file.write_text("x = 1\n" * (_MAX_FILE_SIZE // 6 + 1), encoding="utf-8")
        small_file = tmp_path / "small.py"
        small_file.write_text("y = 2\n", encoding="utf-8")
        scanner = FileScanner(tmp_path)
        files = list(scanner.scan())
        names = [f.name for f in files]
        assert "small.py" in names
        assert "big.py" not in names

    def test_scan_respects_gitignore(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
        (tmp_path / "ignored.py").write_text("x\n", encoding="utf-8")
        (tmp_path / "kept.py").write_text("y\n", encoding="utf-8")
        scanner = FileScanner(tmp_path)
        files = list(scanner.scan())
        names = [f.name for f in files]
        assert "kept.py" in names
        assert "ignored.py" not in names

    def test_count_files(self, tmp_workspace: Path) -> None:
        scanner = FileScanner(tmp_workspace)
        count = scanner.count_files()
        assert count >= 3  # at least app.py, utils.py, test_app.py

    def test_get_file_stats(self, tmp_workspace: Path) -> None:
        scanner = FileScanner(tmp_workspace)
        stats = scanner.get_file_stats()
        assert ".py" in stats
        assert stats[".py"] >= 3

    def test_get_file_stats_sorted_descending(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"mod{i}.py").write_text(f"x = {i}\n", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# hi\n", encoding="utf-8")
        scanner = FileScanner(tmp_path)
        stats = scanner.get_file_stats()
        values = list(stats.values())
        assert values == sorted(values, reverse=True)

    def test_scan_recursive(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("x = 1\n", encoding="utf-8")
        scanner = FileScanner(tmp_path)
        files = list(scanner.scan())
        assert any(f.name == "deep.py" for f in files)

    def test_scan_empty_workspace(self, tmp_path: Path) -> None:
        scanner = FileScanner(tmp_path)
        files = list(scanner.scan())
        assert files == []


# ---------------------------------------------------------------------------
# SymbolExtractor — Python
# ---------------------------------------------------------------------------


class TestSymbolExtractorPython:
    """Tests for Python symbol extraction."""

    def test_extract_function(self, tmp_path: Path) -> None:
        py = tmp_path / "mod.py"
        py.write_text("def greet(name: str) -> str:\n    return f'hi {name}'\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(py)
        assert result.language == "python"
        funcs = result.functions
        assert len(funcs) == 1
        assert funcs[0].name == "greet"
        assert "name: str" in funcs[0].signature

    def test_extract_class(self, tmp_path: Path) -> None:
        py = tmp_path / "mod.py"
        py.write_text("class Dog:\n    pass\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(py)
        assert len(result.classes) == 1
        assert result.classes[0].name == "Dog"

    def test_extract_method_with_parent(self, tmp_path: Path) -> None:
        py = tmp_path / "mod.py"
        py.write_text(textwrap.dedent("""\
            class Animal:
                def speak(self) -> str:
                    return "..."
        """), encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(py)
        methods = result.methods
        assert len(methods) == 1
        assert methods[0].name == "speak"
        assert methods[0].parent == "Animal"

    def test_extract_multiple_classes_and_functions(self, tmp_path: Path) -> None:
        py = tmp_path / "mod.py"
        py.write_text(textwrap.dedent("""\
            class A:
                def method_a(self):
                    pass

            class B:
                def method_b(self):
                    pass

            def standalone():
                pass
        """), encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(py)
        assert len(result.classes) == 2
        assert len(result.methods) >= 2
        # Note: AST fallback may add standalone as method too (sees class B above it)
        assert any(s.name == "standalone" for s in result.symbols)

    def test_extract_return_type(self, tmp_path: Path) -> None:
        py = tmp_path / "mod.py"
        py.write_text("def calc(x: int) -> float:\n    return x * 1.0\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(py)
        assert "-> float" in result.functions[0].signature

    def test_file_metadata(self, tmp_path: Path) -> None:
        py = tmp_path / "mod.py"
        content = "x = 1\ny = 2\n"
        py.write_text(content, encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(py)
        assert result.line_count == 3  # two lines + trailing empty
        assert result.size_bytes == len(content.encode("utf-8"))
        assert result.relative_path == "mod.py"

    def test_relative_path_with_subdirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg" / "sub"
        sub.mkdir(parents=True)
        py = sub / "mod.py"
        py.write_text("x = 1\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(py)
        assert result.relative_path == "pkg/sub/mod.py"

    def test_unreadable_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.py"
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(missing)
        assert result.symbols == []

    def test_syntax_error_file_still_returns_regex_results(self, tmp_path: Path) -> None:
        py = tmp_path / "bad.py"
        py.write_text("def foo():\n    return\n\ndef bar(\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(py)
        # foo should be found by regex even though AST parsing fails
        assert any(s.name == "foo" for s in result.symbols)


# ---------------------------------------------------------------------------
# SymbolExtractor — JavaScript / TypeScript
# ---------------------------------------------------------------------------


class TestSymbolExtractorJavaScript:
    """Tests for JavaScript/TypeScript symbol extraction."""

    def test_extract_js_function(self, tmp_path: Path) -> None:
        js = tmp_path / "app.js"
        js.write_text("function greet(name) {\n  return 'hi ' + name;\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(js)
        assert result.language == "javascript"
        assert len(result.functions) == 1
        assert result.functions[0].name == "greet"

    def test_extract_js_class(self, tmp_path: Path) -> None:
        js = tmp_path / "widget.js"
        js.write_text("class Widget {\n  constructor() {}\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(js)
        assert len(result.classes) == 1
        assert result.classes[0].name == "Widget"

    def test_extract_ts_function(self, tmp_path: Path) -> None:
        ts = tmp_path / "util.ts"
        ts.write_text("export function helper(x: number): string {\n  return String(x);\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(ts)
        assert result.language == "typescript"
        assert len(result.functions) == 1
        assert result.functions[0].name == "helper"

    def test_extract_arrow_function(self, tmp_path: Path) -> None:
        js = tmp_path / "arrows.js"
        js.write_text("const handler = async (req, res) => {\n  res.send('ok');\n};\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(js)
        assert any(s.name == "handler" for s in result.functions)

    def test_extract_async_function(self, tmp_path: Path) -> None:
        js = tmp_path / "async.js"
        js.write_text("async function fetchData() {\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(js)
        assert any(s.name == "fetchData" for s in result.functions)

    def test_extract_export_class(self, tmp_path: Path) -> None:
        ts = tmp_path / "comp.tsx"
        ts.write_text("export class AppComponent {\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(ts)
        assert len(result.classes) == 1
        assert result.classes[0].name == "AppComponent"


# ---------------------------------------------------------------------------
# SymbolExtractor — Rust
# ---------------------------------------------------------------------------


class TestSymbolExtractorRust:
    """Tests for Rust symbol extraction."""

    def test_extract_rust_function(self, tmp_path: Path) -> None:
        rs = tmp_path / "lib.rs"
        rs.write_text("pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(rs)
        assert result.language == "rust"
        assert any(s.name == "add" and s.kind == "function" for s in result.symbols)

    def test_extract_rust_struct(self, tmp_path: Path) -> None:
        rs = tmp_path / "types.rs"
        rs.write_text("pub struct Point {\n    x: f64,\n    y: f64,\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(rs)
        assert any(s.name == "Point" and s.kind == "class" for s in result.symbols)

    def test_extract_rust_trait(self, tmp_path: Path) -> None:
        rs = tmp_path / "traits.rs"
        rs.write_text("pub trait Drawable {\n    fn draw(&self);\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(rs)
        assert any(s.name == "Drawable" and s.kind == "class" for s in result.symbols)


# ---------------------------------------------------------------------------
# SymbolExtractor — Go
# ---------------------------------------------------------------------------


class TestSymbolExtractorGo:
    """Tests for Go symbol extraction."""

    def test_extract_go_function(self, tmp_path: Path) -> None:
        go = tmp_path / "main.go"
        go.write_text("func main() {\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(go)
        assert result.language == "go"
        assert any(s.name == "main" and s.kind == "function" for s in result.symbols)

    def test_extract_go_struct(self, tmp_path: Path) -> None:
        go = tmp_path / "types.go"
        go.write_text("type Server struct {\n    Port int\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(go)
        assert any(s.name == "Server" and s.kind == "class" for s in result.symbols)

    def test_extract_go_interface(self, tmp_path: Path) -> None:
        go = tmp_path / "iface.go"
        go.write_text("type Reader interface {\n    Read(p []byte) (int, error)\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(go)
        assert any(s.name == "Reader" and s.kind == "class" for s in result.symbols)


# ---------------------------------------------------------------------------
# SymbolExtractor — Java
# ---------------------------------------------------------------------------


class TestSymbolExtractorJava:
    """Tests for Java symbol extraction."""

    def test_extract_java_class(self, tmp_path: Path) -> None:
        java = tmp_path / "App.java"
        java.write_text("public class App {\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(java)
        assert result.language == "java"
        assert any(s.name == "App" and s.kind == "class" for s in result.symbols)

    def test_extract_java_method(self, tmp_path: Path) -> None:
        java = tmp_path / "App.java"
        java.write_text("public class App {\n    public void run() {}\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(java)
        assert any(s.name == "run" and s.kind == "method" for s in result.symbols)


# ---------------------------------------------------------------------------
# SymbolExtractor — C/C++
# ---------------------------------------------------------------------------


class TestSymbolExtractorC:
    """Tests for C/C++ symbol extraction."""

    def test_extract_c_function(self, tmp_path: Path) -> None:
        c = tmp_path / "main.c"
        c.write_text("int main(int argc, char *argv[])\n{\n    return 0;\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(c)
        assert result.language == "c"
        assert any(s.name == "main" and s.kind == "function" for s in result.symbols)

    def test_extract_c_struct(self, tmp_path: Path) -> None:
        c = tmp_path / "types.h"
        c.write_text("struct Point {\n    int x;\n    int y;\n};\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(c)
        assert any(s.name == "Point" and s.kind == "class" for s in result.symbols)

    def test_c_skips_keywords(self, tmp_path: Path) -> None:
        c = tmp_path / "control.c"
        c.write_text("if (x > 0)\n{\n    return 1;\n}\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(c)
        # "if" and "return" should not be extracted as functions
        names = [s.name for s in result.symbols]
        assert "if" not in names
        assert "return" not in names


# ---------------------------------------------------------------------------
# SymbolExtractor — unknown language
# ---------------------------------------------------------------------------


class TestSymbolExtractorUnknown:
    """Tests for files with unrecognized extensions."""

    def test_unknown_extension_returns_empty_symbols(self, tmp_path: Path) -> None:
        sql = tmp_path / "query.sql"
        sql.write_text("SELECT * FROM users;\n", encoding="utf-8")
        ex = SymbolExtractor(tmp_path)
        result = ex.extract(sql)
        assert result.language == ""
        assert result.symbols == []

    def test_file_outside_workspace_uses_filename(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "other" / "file.py"
        outside.parent.mkdir()
        outside.write_text("x = 1\n", encoding="utf-8")
        ex = SymbolExtractor(workspace)
        result = ex.extract(outside)
        assert result.relative_path == "file.py"
