"""Web tools — search_web and read_url for internet access.

Provides the agent with the ability to search the web and read web pages.
Uses stdlib ``urllib`` for HTTP and a lightweight HTML→Markdown converter
so no external dependencies are required.
"""

import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from unjess.tools import ToolRegistry

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_MAX_RESPONSE_BYTES = 512 * 1024  # 512 KB cap for read_url
_SEARCH_TIMEOUT = 10
_READ_TIMEOUT = 15


# ---------------------------------------------------------------------------
# HTML → Markdown converter (stdlib only)
# ---------------------------------------------------------------------------

class _HTMLToMarkdown(HTMLParser):
    """Lightweight HTML-to-Markdown converter.

    Handles headings, paragraphs, links, lists, code blocks, bold, italic,
    and strips scripts/styles. Not a full converter — optimised for
    extracting readable text from web pages.
    """

    _BLOCK_TAGS = {"p", "div", "section", "article", "main", "header",
                   "footer", "nav", "blockquote", "li", "tr", "h1", "h2",
                   "h3", "h4", "h5", "h6", "pre", "br", "hr"}
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._output: list[str] = []
        self._skip_depth = 0
        self._tag_stack: list[str] = []
        self._in_pre = False
        self._link_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)

        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        attr_dict = dict(attrs)

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self._output.append("\n\n" + "#" * level + " ")
        elif tag == "p":
            self._output.append("\n\n")
        elif tag == "br":
            self._output.append("\n")
        elif tag == "hr":
            self._output.append("\n\n---\n\n")
        elif tag == "a":
            self._link_href = attr_dict.get("href", "")
            self._output.append("[")
        elif tag in ("b", "strong"):
            self._output.append("**")
        elif tag in ("i", "em"):
            self._output.append("*")
        elif tag == "code" and not self._in_pre:
            self._output.append("`")
        elif tag == "pre":
            self._in_pre = True
            self._output.append("\n\n```\n")
        elif tag == "li":
            self._output.append("\n- ")
        elif tag in ("ul", "ol"):
            self._output.append("\n")
        elif tag == "blockquote":
            self._output.append("\n\n> ")
        elif tag in self._BLOCK_TAGS:
            self._output.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if tag == "a" and self._link_href:
            self._output.append(f"]({self._link_href})")
            self._link_href = None
        elif tag in ("b", "strong"):
            self._output.append("**")
        elif tag in ("i", "em"):
            self._output.append("*")
        elif tag == "code" and not self._in_pre:
            self._output.append("`")
        elif tag == "pre":
            self._in_pre = False
            self._output.append("\n```\n\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._output.append("\n\n")
        elif tag in ("p", "div", "section", "article"):
            self._output.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_pre:
            self._output.append(data)
        else:
            # Collapse whitespace
            text = re.sub(r'\s+', ' ', data)
            self._output.append(text)

    def get_markdown(self) -> str:
        """Return the converted markdown text."""
        text = "".join(self._output)
        # Clean up excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def _html_to_markdown(html: str) -> str:
    """Convert HTML to readable Markdown using stdlib only."""
    parser = _HTMLToMarkdown()
    parser.feed(html)
    return parser.get_markdown()


# ---------------------------------------------------------------------------
# search_web — DuckDuckGo HTML (no API key needed)
# ---------------------------------------------------------------------------

def _search_duckduckgo(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Search DuckDuckGo Lite and parse the results.

    Returns a list of dicts with 'title', 'url', 'snippet' keys.
    """
    from urllib.parse import urlencode
    url = "https://lite.duckduckgo.com/lite/"
    data = urlencode({"q": query}).encode()

    req = Request(url, data=data, method="POST")
    req.add_header("User-Agent", _USER_AGENT)

    with urlopen(req, timeout=_SEARCH_TIMEOUT) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    results: list[dict[str, str]] = []

    # DDG Lite format: <a href="..." class='result-link'>title</a>
    # href comes BEFORE class, so we match both orders
    link_pattern = re.compile(
        r"<a[^>]+class=['\"]result-link['\"][^>]*>(.*?)</a>",
        re.DOTALL | re.IGNORECASE,
    )
    href_pattern = re.compile(
        r"""href=['"](https?://[^'"]+)['"]""",
        re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r"<td[^>]+class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
        re.DOTALL | re.IGNORECASE,
    )

    # Find all result link <a> tags
    link_tags = list(re.finditer(
        r"<a[^>]+class=['\"]result-link['\"][^>]*>.*?</a>",
        html, re.DOTALL | re.IGNORECASE,
    ))
    snippets = snippet_pattern.findall(html)

    for i, tag_match in enumerate(link_tags[:max_results]):
        tag_html = tag_match.group(0)
        # Extract href from the <a> tag
        href_m = href_pattern.search(tag_html)
        href = href_m.group(1) if href_m else ""
        # Extract title text
        title_m = re.search(r">(.*?)</a>", tag_html, re.DOTALL)
        title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else ""
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()

        if href and title:
            results.append({
                "title": title,
                "url": href,
                "snippet": snippet,
            })

    return results


def _search_web(
    query: str,
    max_results: int = 8,
) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    if not query or not query.strip():
        return "Error: Empty search query."

    max_results = min(max(1, max_results), 15)

    try:
        results = _search_duckduckgo(query, max_results)
    except (HTTPError, URLError) as exc:
        return f"Error: Search request failed: {exc}"
    except Exception as exc:
        logger.exception("Web search failed")
        return f"Error: Search failed: {type(exc).__name__}: {exc}"

    if not results:
        return f"No results found for: {query}"

    lines = [f"**Web search results for:** {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r['title']}")
        lines.append(f"**URL:** {r['url']}")
        if r.get("snippet"):
            lines.append(r["snippet"])
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# read_url — fetch and convert to markdown
# ---------------------------------------------------------------------------

def _read_url(url: str, max_length: int = 20000) -> str:
    """Fetch a URL and return its content as Markdown.

    Handles HTML pages (converts to markdown), JSON (pretty-prints),
    and plain text (returns as-is).

    Args:
        url: The URL to fetch.
        max_length: Maximum character length of the returned content.

    Returns:
        The page content as readable text/markdown.
    """
    if not url or not url.strip():
        return "Error: Empty URL."

    url = url.strip()
    # Security: only allow http/https — block file://, ftp://, data: etc.
    if url.startswith(("file://", "ftp://", "data:", "javascript:")):
        return "Error: Only http:// and https:// URLs are supported."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    max_length = min(max(1000, max_length), 50000)

    req = Request(url, method="GET")
    req.add_header("User-Agent", _USER_AGENT)
    req.add_header("Accept", "text/html,application/json,text/plain,*/*")
    req.add_header("Accept-Language", "en-US,en;q=0.9")

    try:
        with urlopen(req, timeout=_READ_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            raw = resp.read(_MAX_RESPONSE_BYTES)

            # Detect encoding
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()

            text = raw.decode(charset, errors="replace")
    except HTTPError as exc:
        return f"Error: HTTP {exc.code} — {exc.reason} for URL: {url}"
    except URLError as exc:
        return f"Error: Could not reach URL: {url} — {exc.reason}"
    except Exception as exc:
        return f"Error: Failed to fetch URL: {type(exc).__name__}: {exc}"

    # Handle different content types
    if "json" in content_type:
        try:
            parsed = json.loads(text)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            result = f"**JSON from:** {url}\n\n```json\n{formatted}\n```"
        except json.JSONDecodeError:
            result = f"**Raw response from:** {url}\n\n{text}"
    elif "html" in content_type or text.strip().startswith("<!") or "<html" in text[:500].lower():
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.DOTALL | re.IGNORECASE)
        title = re.sub(r'\s+', ' ', title_match.group(1)).strip() if title_match else ""

        markdown = _html_to_markdown(text)
        header = f"**Page:** {title}\n**URL:** {url}\n\n---\n\n" if title else f"**URL:** {url}\n\n---\n\n"
        result = header + markdown
    else:
        # Plain text or unknown
        result = f"**Content from:** {url}\n\n{text}"

    # Truncate if needed
    if len(result) > max_length:
        result = result[:max_length] + f"\n\n... [truncated — {len(result)} chars total, showing first {max_length}]"

    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_web_tools(registry: ToolRegistry) -> None:
    """Register web tools on the given registry.

    Args:
        registry: The tool registry to add tools to.
    """
    registry.register(
        name="search_web",
        description=(
            "Search the web for information. Returns titles, URLs, and snippets "
            "from search results. Use this when you need to look up documentation, "
            "find solutions, or research a topic."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (1-15). Default: 8.",
                },
            },
            "required": ["query"],
        },
        handler=_search_web,
    )

    registry.register(
        name="read_url",
        description=(
            "Fetch a web page and return its content as readable Markdown. "
            "Handles HTML pages, JSON APIs, and plain text. Use this to read "
            "documentation, API responses, or any web content."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Max character length of returned content (1000-50000). Default: 20000.",
                },
            },
            "required": ["url"],
        },
        handler=_read_url,
    )
