"""Tests for unjess.tools.web_tools — search, read_url, HTML conversion."""

import json
from http.client import HTTPResponse
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from unjess.tools.web_tools import (
    _html_to_markdown,
    _HTMLToMarkdown,
    _read_url,
    _search_duckduckgo,
    _search_web,
    register_web_tools,
)


# ---------------------------------------------------------------------------
# _HTMLToMarkdown converter
# ---------------------------------------------------------------------------

class TestHTMLToMarkdown:
    """Tests for the lightweight HTML-to-Markdown converter."""

    def test_plain_text_passthrough(self) -> None:
        result = _html_to_markdown("Hello World")
        assert "Hello World" in result

    def test_heading_conversion(self) -> None:
        result = _html_to_markdown("<h1>Title</h1>")
        assert "# Title" in result

    def test_h2_conversion(self) -> None:
        result = _html_to_markdown("<h2>Subtitle</h2>")
        assert "## Subtitle" in result

    def test_h3_conversion(self) -> None:
        result = _html_to_markdown("<h3>Section</h3>")
        assert "### Section" in result

    def test_paragraph_adds_blank_lines(self) -> None:
        result = _html_to_markdown("<p>First</p><p>Second</p>")
        assert "First" in result
        assert "Second" in result

    def test_bold_conversion(self) -> None:
        result = _html_to_markdown("<b>bold</b>")
        assert "**bold**" in result

    def test_strong_conversion(self) -> None:
        result = _html_to_markdown("<strong>important</strong>")
        assert "**important**" in result

    def test_italic_conversion(self) -> None:
        result = _html_to_markdown("<i>italic</i>")
        assert "*italic*" in result

    def test_em_conversion(self) -> None:
        result = _html_to_markdown("<em>emphasis</em>")
        assert "*emphasis*" in result

    def test_link_conversion(self) -> None:
        result = _html_to_markdown('<a href="https://example.com">Click</a>')
        assert "[Click](https://example.com)" in result

    def test_code_inline_conversion(self) -> None:
        result = _html_to_markdown("<code>x = 1</code>")
        assert "`x = 1`" in result

    def test_pre_block_conversion(self) -> None:
        result = _html_to_markdown("<pre>line1\nline2</pre>")
        assert "```" in result
        assert "line1" in result

    def test_list_item_conversion(self) -> None:
        result = _html_to_markdown("<ul><li>Item 1</li><li>Item 2</li></ul>")
        assert "- Item 1" in result
        assert "- Item 2" in result

    def test_hr_conversion(self) -> None:
        result = _html_to_markdown("<hr>")
        assert "---" in result

    def test_br_conversion(self) -> None:
        result = _html_to_markdown("Line1<br>Line2")
        assert "Line1" in result
        assert "Line2" in result

    def test_blockquote_conversion(self) -> None:
        result = _html_to_markdown("<blockquote>quoted</blockquote>")
        assert "> quoted" in result

    def test_script_tags_stripped(self) -> None:
        result = _html_to_markdown("<p>Visible</p><script>alert('xss')</script>")
        assert "Visible" in result
        assert "alert" not in result

    def test_style_tags_stripped(self) -> None:
        result = _html_to_markdown("<p>Text</p><style>body { color: red; }</style>")
        assert "Text" in result
        assert "color" not in result

    def test_noscript_stripped(self) -> None:
        result = _html_to_markdown("<p>OK</p><noscript>Enable JS</noscript>")
        assert "OK" in result
        assert "Enable JS" not in result

    def test_whitespace_collapsed(self) -> None:
        result = _html_to_markdown("<p>  lots   of   spaces  </p>")
        assert "lots of spaces" in result

    def test_excessive_newlines_collapsed(self) -> None:
        result = _html_to_markdown("<p>A</p>\n\n\n\n<p>B</p>")
        assert result.count("\n\n\n") == 0

    def test_head_tag_stripped(self) -> None:
        html = "<head><title>Page</title><meta charset='utf-8'></head><body><p>Hello</p></body>"
        result = _html_to_markdown(html)
        assert "Hello" in result
        assert "charset" not in result


# ---------------------------------------------------------------------------
# _search_web
# ---------------------------------------------------------------------------

class TestSearchWeb:
    """Tests for the _search_web function."""

    def test_empty_query_returns_error(self) -> None:
        result = _search_web("")
        assert "Error" in result
        assert "Empty" in result

    def test_whitespace_query_returns_error(self) -> None:
        result = _search_web("   ")
        assert "Error" in result

    def test_max_results_clamped_low(self) -> None:
        # max_results < 1 should be clamped to 1
        with patch("unjess.tools.web_tools._search_duckduckgo", return_value=[]) as mock_ddg:
            _search_web("test", max_results=0)
            mock_ddg.assert_called_once_with("test", 1)

    def test_max_results_clamped_high(self) -> None:
        with patch("unjess.tools.web_tools._search_duckduckgo", return_value=[]) as mock_ddg:
            _search_web("test", max_results=100)
            mock_ddg.assert_called_once_with("test", 15)

    def test_no_results_found(self) -> None:
        with patch("unjess.tools.web_tools._search_duckduckgo", return_value=[]):
            result = _search_web("obscure query 123456")
        assert "No results found" in result

    def test_formats_results_correctly(self) -> None:
        fake_results = [
            {"title": "Result One", "url": "https://example.com/1", "snippet": "Snippet one"},
            {"title": "Result Two", "url": "https://example.com/2", "snippet": "Snippet two"},
        ]
        with patch("unjess.tools.web_tools._search_duckduckgo", return_value=fake_results):
            result = _search_web("test query")
        assert "Result One" in result
        assert "https://example.com/1" in result
        assert "Snippet one" in result
        assert "Result Two" in result
        assert "test query" in result

    def test_http_error_handled(self) -> None:
        with patch(
            "unjess.tools.web_tools._search_duckduckgo",
            side_effect=HTTPError("url", 503, "Service Unavailable", {}, None),
        ):
            result = _search_web("test")
        assert "Error" in result
        assert "failed" in result.lower()

    def test_url_error_handled(self) -> None:
        with patch(
            "unjess.tools.web_tools._search_duckduckgo",
            side_effect=URLError("Connection refused"),
        ):
            result = _search_web("test")
        assert "Error" in result

    def test_generic_exception_handled(self) -> None:
        with patch(
            "unjess.tools.web_tools._search_duckduckgo",
            side_effect=RuntimeError("unexpected"),
        ):
            result = _search_web("test")
        assert "Error" in result
        assert "RuntimeError" in result

    def test_results_have_numbered_headings(self) -> None:
        fake_results = [
            {"title": "First", "url": "https://a.com", "snippet": "s1"},
        ]
        with patch("unjess.tools.web_tools._search_duckduckgo", return_value=fake_results):
            result = _search_web("test")
        assert "### 1." in result

    def test_empty_snippet_omitted(self) -> None:
        fake_results = [
            {"title": "No Snippet", "url": "https://a.com", "snippet": ""},
        ]
        with patch("unjess.tools.web_tools._search_duckduckgo", return_value=fake_results):
            result = _search_web("test")
        assert "No Snippet" in result


# ---------------------------------------------------------------------------
# _read_url
# ---------------------------------------------------------------------------

def _mock_urlopen(content: bytes, content_type: str = "text/html", charset: str = "utf-8"):
    """Build a mock context manager mimicking urlopen."""
    resp = MagicMock()
    resp.read.return_value = content
    resp.headers = {"Content-Type": f"{content_type}; charset={charset}"}
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestReadUrl:
    """Tests for the _read_url function."""

    def test_empty_url_returns_error(self) -> None:
        result = _read_url("")
        assert "Error" in result
        assert "Empty" in result

    def test_whitespace_url_returns_error(self) -> None:
        result = _read_url("   ")
        assert "Error" in result

    def test_file_protocol_blocked(self) -> None:
        result = _read_url("file:///etc/passwd")
        assert "Error" in result
        assert "Only http" in result

    def test_ftp_protocol_blocked(self) -> None:
        result = _read_url("ftp://evil.com/file")
        assert "Error" in result

    def test_data_protocol_blocked(self) -> None:
        result = _read_url("data:text/html,<h1>hi</h1>")
        assert "Error" in result

    def test_javascript_protocol_blocked(self) -> None:
        result = _read_url("javascript:alert(1)")
        assert "Error" in result

    def test_auto_prepends_https(self) -> None:
        html = b"<html><body><p>Hello</p></body></html>"
        mock_resp = _mock_urlopen(html)
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("example.com")
        assert "Hello" in result

    def test_html_page_converted_to_markdown(self) -> None:
        html = b"<html><head><title>Test Page</title></head><body><h1>Welcome</h1><p>Content here.</p></body></html>"
        mock_resp = _mock_urlopen(html)
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("https://example.com")
        assert "Test Page" in result
        assert "Welcome" in result
        assert "Content here" in result

    def test_json_content_pretty_printed(self) -> None:
        data = {"key": "value", "num": 42}
        mock_resp = _mock_urlopen(json.dumps(data).encode(), content_type="application/json")
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("https://api.example.com/data")
        assert '"key": "value"' in result
        assert "```json" in result

    def test_json_parse_error_returns_raw(self) -> None:
        mock_resp = _mock_urlopen(b"not valid json", content_type="application/json")
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("https://api.example.com/bad")
        assert "not valid json" in result
        assert "Raw response" in result

    def test_plain_text_returned_as_is(self) -> None:
        mock_resp = _mock_urlopen(b"Plain text content", content_type="text/plain")
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("https://example.com/file.txt")
        assert "Plain text content" in result
        assert "Content from:" in result

    def test_truncation_when_exceeds_max_length(self) -> None:
        long_html = b"<html><body>" + b"x" * 5000 + b"</body></html>"
        mock_resp = _mock_urlopen(long_html)
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("https://example.com", max_length=1000)
        assert "truncated" in result

    def test_max_length_clamped_to_minimum(self) -> None:
        html = b"<html><body><p>Short</p></body></html>"
        mock_resp = _mock_urlopen(html)
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            # max_length=1 should be clamped to 1000
            result = _read_url("https://example.com", max_length=1)
        # Should not crash, min is 1000

    def test_http_error_returns_message(self) -> None:
        with patch(
            "unjess.tools.web_tools.urlopen",
            side_effect=HTTPError("https://example.com", 404, "Not Found", {}, None),
        ):
            result = _read_url("https://example.com/404")
        assert "Error" in result
        assert "404" in result

    def test_url_error_returns_message(self) -> None:
        with patch(
            "unjess.tools.web_tools.urlopen",
            side_effect=URLError("Name resolution failed"),
        ):
            result = _read_url("https://nonexistent.example.com")
        assert "Error" in result
        assert "Could not reach" in result

    def test_generic_exception_returns_message(self) -> None:
        with patch(
            "unjess.tools.web_tools.urlopen",
            side_effect=TimeoutError("Connection timed out"),
        ):
            result = _read_url("https://slow.example.com")
        assert "Error" in result
        assert "TimeoutError" in result

    def test_html_without_title(self) -> None:
        html = b"<html><body><p>No title here</p></body></html>"
        mock_resp = _mock_urlopen(html)
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("https://example.com")
        assert "No title here" in result
        assert "**URL:**" in result

    def test_content_detected_as_html_by_doctype(self) -> None:
        html = b"<!DOCTYPE html><html><body><p>DOCTYPE page</p></body></html>"
        mock_resp = _mock_urlopen(html, content_type="text/plain")
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("https://example.com")
        assert "DOCTYPE page" in result

    def test_content_detected_as_html_by_tag(self) -> None:
        html = b"<html><body><p>Tagged page</p></body></html>"
        mock_resp = _mock_urlopen(html, content_type="application/octet-stream")
        with patch("unjess.tools.web_tools.urlopen", return_value=mock_resp):
            result = _read_url("https://example.com")
        assert "Tagged page" in result


# ---------------------------------------------------------------------------
# register_web_tools
# ---------------------------------------------------------------------------

class TestRegisterWebTools:
    """Tests for tool registration."""

    def test_registers_search_web(self) -> None:
        from unjess.tools import ToolRegistry
        registry = ToolRegistry()
        register_web_tools(registry)
        tool = registry.get_tool("search_web")
        assert tool is not None
        assert "search" in tool.description.lower()

    def test_registers_read_url(self) -> None:
        from unjess.tools import ToolRegistry
        registry = ToolRegistry()
        register_web_tools(registry)
        tool = registry.get_tool("read_url")
        assert tool is not None
        assert "fetch" in tool.description.lower()

    def test_registered_handlers_are_callable(self) -> None:
        from unjess.tools import ToolRegistry
        registry = ToolRegistry()
        register_web_tools(registry)
        for name in ("search_web", "read_url"):
            tool = registry.get_tool(name)
            assert tool is not None
            assert callable(tool.handler)

    def test_search_web_schema_requires_query(self) -> None:
        from unjess.tools import ToolRegistry
        registry = ToolRegistry()
        register_web_tools(registry)
        tool = registry.get_tool("search_web")
        assert tool is not None
        assert "query" in tool.parameters.get("required", [])

    def test_read_url_schema_requires_url(self) -> None:
        from unjess.tools import ToolRegistry
        registry = ToolRegistry()
        register_web_tools(registry)
        tool = registry.get_tool("read_url")
        assert tool is not None
        assert "url" in tool.parameters.get("required", [])
