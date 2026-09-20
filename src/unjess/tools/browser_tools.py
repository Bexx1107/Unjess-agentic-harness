"""Browser tools — expose Playwright browser automation as agent tools.

Wraps the BrowserDriver for use by the agent via the tool registry.
The browser is lazily started on first tool call.
"""

import logging
from pathlib import Path
from typing import Optional

from unjess.browser.driver import BrowserDriver
from unjess.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Shared driver instance — lazily initialized
_driver: Optional[BrowserDriver] = None


def _get_driver() -> BrowserDriver:
    """Get or create the shared browser driver."""
    global _driver
    if _driver is None:
        _driver = BrowserDriver(headless=True)
    if not _driver.is_available:
        raise RuntimeError(
            "Playwright is not installed. Run: pip install playwright && python -m playwright install chromium"
        )
    if not _driver.is_started:
        if not _driver.start():
            raise RuntimeError(
                "Failed to start browser. Ensure Chromium is installed: python -m playwright install chromium"
            )
    return _driver


def _browser_navigate(url: str, wait_until: str = "domcontentloaded") -> str:
    """Navigate to a URL."""
    driver = _get_driver()
    result = driver.navigate(url, wait_until=wait_until)
    if "error" in result:
        return f"Navigation error: {result['error']}"
    return f"Navigated to: {result['url']} (title: {result['title']}, status: {result['status']})"


def _browser_screenshot(path: str = "", full_page: bool = False) -> str:
    """Take a screenshot of the current page."""
    driver = _get_driver()
    screenshots_dir = Path.home() / ".unjess" / "screenshots"
    if not path:
        safe_path = screenshots_dir / "latest.png"
    else:
        # Force screenshots under ~/.unjess/screenshots/ for safety
        safe_path = screenshots_dir / Path(path).name
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    saved = driver.screenshot(str(safe_path), full_page=full_page)
    if saved:
        return f"Screenshot saved to: {saved}"
    return "Screenshot failed."


def _browser_click(selector: str) -> str:
    """Click an element on the page."""
    driver = _get_driver()
    success = driver.click(selector)
    if success:
        return f"Clicked: {selector}"
    return f"Click failed for: {selector}"


def _browser_type(selector: str, text: str, clear: bool = True) -> str:
    """Type text into an input field."""
    driver = _get_driver()
    success = driver.type_text(selector, text, clear=clear)
    if success:
        return f"Typed into {selector}: {text[:50]}..."
    return f"Type failed for: {selector}"


def _browser_get_text(selector: str = "body") -> str:
    """Extract text content from the page or a specific element."""
    driver = _get_driver()
    text = driver.get_text(selector)
    if len(text) > 15000:
        text = text[:15000] + f"\n\n... [truncated — {len(text)} chars total]"
    return text


def _browser_eval_js(expression: str) -> str:
    """Evaluate JavaScript on the page and return the result."""
    driver = _get_driver()
    result = driver.evaluate_js(expression)
    return str(result)


def _browser_close() -> str:
    """Close the browser."""
    global _driver
    if _driver and _driver.is_started:
        _driver.stop()
    _driver = None
    return "Browser closed."


def register_browser_tools(registry: ToolRegistry) -> None:
    """Register browser automation tools."""
    registry.register(
        name="browser_navigate",
        description="Navigate the browser to a URL. Starts the browser if not already running.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to."},
                "wait_until": {"type": "string", "description": "Wait condition: 'domcontentloaded', 'load', 'networkidle'. Default: 'domcontentloaded'."},
            },
            "required": ["url"],
        },
        handler=_browser_navigate,
    )

    registry.register(
        name="browser_screenshot",
        description="Take a screenshot of the current browser page. Returns the file path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to save screenshot. Default: ~/.unjess/screenshots/latest.png"},
                "full_page": {"type": "boolean", "description": "Capture full scrollable page. Default: false."},
            },
        },
        handler=_browser_screenshot,
    )

    registry.register(
        name="browser_click",
        description="Click an element on the page by CSS selector.",
        parameters={
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector of the element to click."},
            },
            "required": ["selector"],
        },
        handler=_browser_click,
    )

    registry.register(
        name="browser_type",
        description="Type text into an input field identified by CSS selector.",
        parameters={
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector of the input field."},
                "text": {"type": "string", "description": "Text to type."},
                "clear": {"type": "boolean", "description": "Clear field before typing. Default: true."},
            },
            "required": ["selector", "text"],
        },
        handler=_browser_type,
    )

    registry.register(
        name="browser_get_text",
        description="Extract text content from the current page or a specific element.",
        parameters={
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector. Default: 'body' (entire page)."},
            },
        },
        handler=_browser_get_text,
    )

    registry.register(
        name="browser_eval_js",
        description="Evaluate a JavaScript expression on the current page and return the result.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "JavaScript expression to evaluate."},
            },
            "required": ["expression"],
        },
        handler=_browser_eval_js,
    )

    registry.register(
        name="browser_close",
        description="Close the browser and free resources.",
        parameters={"type": "object", "properties": {}},
        handler=_browser_close,
    )
