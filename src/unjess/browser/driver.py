"""Browser driver — Playwright-based web automation.

Provides navigate, click, type, screenshot, get_text, and evaluate_js.
Falls back gracefully if Playwright is not installed.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright, Page, Browser
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False
    Page = Any  # type: ignore[assignment,misc]
    Browser = Any  # type: ignore[assignment,misc]


class BrowserDriver:
    """Playwright-based browser automation.

    Manages a headless Chromium browser for navigating pages,
    interacting with elements, taking screenshots, and extracting text.

    Usage:
        driver = BrowserDriver()
        driver.start()
        driver.navigate("https://example.com")
        text = driver.get_text()
        driver.screenshot("/tmp/page.png")
        driver.stop()
    """

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._started = False

    @property
    def is_available(self) -> bool:
        """Whether Playwright is installed."""
        return _HAS_PLAYWRIGHT

    @property
    def is_started(self) -> bool:
        """Whether the browser is running."""
        return self._started

    @property
    def current_url(self) -> str:
        """Current page URL."""
        if self._page:
            return self._page.url
        return ""

    @property
    def title(self) -> str:
        """Current page title."""
        if self._page:
            return self._page.title()
        return ""

    # ----- Lifecycle -----

    def start(self) -> bool:
        """Start the browser.

        Returns:
            True if started successfully.
        """
        if not _HAS_PLAYWRIGHT:
            logger.error(
                "Playwright is not installed. Install with: "
                "pip install playwright && python -m playwright install chromium"
            )
            return False

        if self._started:
            return True

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self._headless)
            self._page = self._browser.new_page()
            self._started = True
            logger.info("Browser started (headless=%s)", self._headless)
            return True
        except Exception as exc:
            logger.error("Failed to start browser: %s", exc)
            return False

    def stop(self) -> None:
        """Stop the browser and clean up."""
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        finally:
            self._browser = None
            self._page = None
            self._playwright = None
            self._started = False
            logger.info("Browser stopped")

    # ----- Navigation -----

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> dict[str, Any]:
        """Navigate to a URL.

        Args:
            url: The URL to navigate to.
            wait_until: Wait strategy ("domcontentloaded", "load", "networkidle").

        Returns:
            Dict with status, url, title.
        """
        self._ensure_started()
        try:
            response = self._page.goto(url, wait_until=wait_until, timeout=30000)
            status = response.status if response else 0
            return {
                "status": status,
                "url": self._page.url,
                "title": self._page.title(),
            }
        except Exception as exc:
            return {"status": 0, "url": url, "error": str(exc)}

    # ----- Interaction -----

    def click(self, selector: str) -> bool:
        """Click an element.

        Args:
            selector: CSS selector or text selector.

        Returns:
            True if clicked successfully.
        """
        self._ensure_started()
        try:
            self._page.click(selector, timeout=5000)
            return True
        except Exception as exc:
            logger.warning("Click failed for '%s': %s", selector, exc)
            return False

    def type_text(self, selector: str, text: str, clear: bool = True) -> bool:
        """Type text into an input element.

        Args:
            selector: CSS selector for the input.
            text: Text to type.
            clear: Whether to clear existing text first.

        Returns:
            True if typed successfully.
        """
        self._ensure_started()
        try:
            if clear:
                self._page.fill(selector, text, timeout=5000)
            else:
                self._page.type(selector, text, timeout=5000)
            return True
        except Exception as exc:
            logger.warning("Type failed for '%s': %s", selector, exc)
            return False

    def press_key(self, key: str) -> bool:
        """Press a keyboard key.

        Args:
            key: Key name (e.g., "Enter", "Tab", "Escape").

        Returns:
            True if pressed successfully.
        """
        self._ensure_started()
        try:
            self._page.keyboard.press(key)
            return True
        except Exception:
            return False

    # ----- Content extraction -----

    def get_text(self, selector: str = "body") -> str:
        """Get text content of an element.

        Args:
            selector: CSS selector (default "body" for full page).

        Returns:
            Text content.
        """
        self._ensure_started()
        try:
            return self._page.text_content(selector, timeout=5000) or ""
        except Exception:
            return ""

    def get_html(self, selector: str = "html") -> str:
        """Get inner HTML of an element.

        Args:
            selector: CSS selector.

        Returns:
            HTML content.
        """
        self._ensure_started()
        try:
            return self._page.inner_html(selector, timeout=5000)
        except Exception:
            return ""

    def get_attribute(self, selector: str, attribute: str) -> str:
        """Get an attribute value from an element.

        Args:
            selector: CSS selector.
            attribute: Attribute name.

        Returns:
            Attribute value or empty string.
        """
        self._ensure_started()
        try:
            return self._page.get_attribute(selector, attribute, timeout=5000) or ""
        except Exception:
            return ""

    def query_selector_all(self, selector: str) -> list[dict[str, str]]:
        """Query all matching elements and return their text + attributes.

        Args:
            selector: CSS selector.

        Returns:
            List of dicts with text, href, id, class info.
        """
        self._ensure_started()
        try:
            elements = self._page.query_selector_all(selector)
            results = []
            for el in elements[:50]:  # cap at 50
                results.append({
                    "text": (el.text_content() or "").strip()[:200],
                    "tag": el.evaluate("el => el.tagName.toLowerCase()"),
                    "href": el.get_attribute("href") or "",
                    "id": el.get_attribute("id") or "",
                })
            return results
        except Exception:
            return []

    # ----- Screenshots -----

    def screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
    ) -> str:
        """Take a screenshot.

        Args:
            path: File path to save to (or None for base64).
            full_page: Whether to capture the full scrollable page.

        Returns:
            File path if saved, or base64 string.
        """
        self._ensure_started()
        try:
            if path:
                self._page.screenshot(path=path, full_page=full_page)
                return path
            else:
                raw = self._page.screenshot(full_page=full_page)
                return base64.b64encode(raw).decode("ascii")
        except Exception as exc:
            logger.error("Screenshot failed: %s", exc)
            return ""

    # ----- JavaScript -----

    def evaluate_js(self, expression: str) -> Any:
        """Evaluate a JavaScript expression in the page context.

        Args:
            expression: JavaScript to evaluate.

        Returns:
            Result of the expression.
        """
        self._ensure_started()
        try:
            return self._page.evaluate(expression)
        except Exception as exc:
            logger.error("JS evaluation failed: %s", exc)
            return None

    # ----- Waiting -----

    def wait_for_selector(self, selector: str, timeout: int = 10000) -> bool:
        """Wait for an element to appear.

        Args:
            selector: CSS selector.
            timeout: Timeout in milliseconds.

        Returns:
            True if element appeared.
        """
        self._ensure_started()
        try:
            self._page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    def wait_for_navigation(self, timeout: int = 10000) -> bool:
        """Wait for a navigation to complete.

        Args:
            timeout: Timeout in milliseconds.

        Returns:
            True if navigation completed.
        """
        self._ensure_started()
        try:
            self._page.wait_for_load_state("domcontentloaded", timeout=timeout)
            return True
        except Exception:
            return False

    # ----- Internal -----

    def _ensure_started(self) -> None:
        """Ensure the browser is started."""
        if not self._started:
            if not self.start():
                raise RuntimeError("Browser is not available. Install playwright.")
