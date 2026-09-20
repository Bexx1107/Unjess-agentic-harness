"""Browser driver — Playwright-based web automation.

Provides navigate, click, type, screenshot, get_text, and evaluate_js.
Falls back gracefully if Playwright is not installed.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_HAS_PLAYWRIGHT: Optional[bool] = None
sync_playwright: Any = None
Page: Any = Any
Browser: Any = Any


def _configure_playwright_browsers_path() -> None:
    """Ensure PLAYWRIGHT_BROWSERS_PATH points to the installed browsers directory."""
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ or not os.environ["PLAYWRIGHT_BROWSERS_PATH"]:
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            browsers_dir = os.path.join(local_app_data, "ms-playwright")
            if os.path.isdir(browsers_dir):
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = browsers_dir
            else:
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
        else:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"


def _find_system_site_packages() -> list[str]:
    """Find system Python site-packages directories if running in a frozen bundle."""
    found: list[str] = []
    # 1. Environment variables (Windows AppData)
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    app_data = os.environ.get("APPDATA", "")
    if local_app_data:
        found.extend(glob.glob(os.path.join(local_app_data, "Programs", "Python", "Python*", "Lib", "site-packages")))
    if app_data:
        found.extend(glob.glob(os.path.join(app_data, "Python", "Python*", "site-packages")))
    # 2. PATH inspection
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if "python" in p.lower():
            site_pkg = os.path.join(p, "Lib", "site-packages")
            if os.path.isdir(site_pkg) and site_pkg not in found:
                found.append(site_pkg)
            parent_site_pkg = os.path.join(os.path.dirname(p), "Lib", "site-packages")
            if os.path.isdir(parent_site_pkg) and parent_site_pkg not in found:
                found.append(parent_site_pkg)
    return found


def _ensure_playwright() -> bool:
    """Dynamically import playwright, locating system site-packages if frozen."""
    global _HAS_PLAYWRIGHT, sync_playwright, Page, Browser
    if _HAS_PLAYWRIGHT is True:
        return True

    _configure_playwright_browsers_path()

    # 1. Direct import attempt
    try:
        from playwright.sync_api import sync_playwright as _sp, Page as _P, Browser as _B
        sync_playwright = _sp
        Page = _P
        Browser = _B
        _HAS_PLAYWRIGHT = True
        return True
    except ImportError:
        pass

    # 2. Fallback: discover system site-packages (for frozen PyInstaller bundle)
    for path in _find_system_site_packages():
        if path not in sys.path and os.path.isdir(path):
            sys.path.insert(0, path)

    try:
        from playwright.sync_api import sync_playwright as _sp, Page as _P, Browser as _B
        sync_playwright = _sp
        Page = _P
        Browser = _B
        _HAS_PLAYWRIGHT = True
        return True
    except ImportError:
        _HAS_PLAYWRIGHT = False
        return False


# Try initial load (non-blocking)
_ensure_playwright()


class BrowserDriver:
    """Playwright-based browser automation.

    Manages a headless Chromium browser for navigating pages,
    interacting with elements, taking screenshots, and extracting text.
    All Playwright operations run in a dedicated background worker thread
    to prevent conflicts with asyncio event loops (e.g. in NiceGUI).

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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unjess-browser")
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._started = False
        self._last_error = ""

    @property
    def is_available(self) -> bool:
        """Whether Playwright is installed."""
        return _ensure_playwright()

    @property
    def is_started(self) -> bool:
        """Whether the browser is running."""
        return self._started

    @property
    def last_error(self) -> str:
        """Last error message encountered during browser lifecycle."""
        return self._last_error

    @property
    def current_url(self) -> str:
        """Current page URL."""
        if self._page:
            try:
                return self._executor.submit(lambda: self._page.url if self._page else "").result()
            except Exception:
                return ""
        return ""

    @property
    def title(self) -> str:
        """Current page title."""
        if self._page:
            try:
                return self._executor.submit(lambda: self._page.title() if self._page else "").result()
            except Exception:
                return ""
        return ""

    # ----- Lifecycle -----

    def start(self) -> bool:
        """Start the browser.

        Returns:
            True if started successfully.
        """
        if not _ensure_playwright():
            self._last_error = "Playwright is not installed. Run: pip install playwright && python -m playwright install chromium"
            logger.error(self._last_error)
            return False

        if self._started:
            return True

        def _do_start() -> bool:
            try:
                _configure_playwright_browsers_path()
                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch(
                    headless=self._headless,
                    args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
                )
                self._page = self._browser.new_page()
                self._started = True
                self._last_error = ""
                logger.info("Browser started (headless=%s)", self._headless)
                return True
            except Exception as exc:
                self._last_error = str(exc)
                logger.error("Failed to start browser: %s", exc)
                return False

        return self._executor.submit(_do_start).result()

    def stop(self) -> None:
        """Stop the browser and clean up."""
        def _do_stop() -> None:
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

        try:
            self._executor.submit(_do_stop).result()
        except Exception:
            self._browser = None
            self._page = None
            self._playwright = None
            self._started = False

    def _run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a callable inside the dedicated browser worker thread."""
        return self._executor.submit(fn, *args, **kwargs).result()

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
        def _do_nav() -> dict[str, Any]:
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
        return self._run(_do_nav)

    # ----- Interaction -----

    def click(self, selector: str) -> bool:
        """Click an element.

        Args:
            selector: CSS selector or text selector.

        Returns:
            True if clicked successfully.
        """
        self._ensure_started()
        def _do_click() -> bool:
            try:
                self._page.click(selector, timeout=5000)
                return True
            except Exception as exc:
                logger.warning("Click failed for '%s': %s", selector, exc)
                return False
        return self._run(_do_click)

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
        def _do_type() -> bool:
            try:
                if clear:
                    self._page.fill(selector, text, timeout=5000)
                else:
                    self._page.type(selector, text, timeout=5000)
                return True
            except Exception as exc:
                logger.warning("Type failed for '%s': %s", selector, exc)
                return False
        return self._run(_do_type)

    def press_key(self, key: str) -> bool:
        """Press a keyboard key.

        Args:
            key: Key name (e.g., "Enter", "Tab", "Escape").

        Returns:
            True if pressed successfully.
        """
        self._ensure_started()
        def _do_press() -> bool:
            try:
                self._page.keyboard.press(key)
                return True
            except Exception:
                return False
        return self._run(_do_press)

    # ----- Content extraction -----

    def get_text(self, selector: str = "body") -> str:
        """Get text content of an element.

        Args:
            selector: CSS selector (default "body" for full page).

        Returns:
            Text content.
        """
        self._ensure_started()
        def _do_get() -> str:
            try:
                return self._page.text_content(selector, timeout=5000) or ""
            except Exception:
                return ""
        return self._run(_do_get)

    def get_html(self, selector: str = "html") -> str:
        """Get inner HTML of an element.

        Args:
            selector: CSS selector.

        Returns:
            HTML content.
        """
        self._ensure_started()
        def _do_html() -> str:
            try:
                return self._page.inner_html(selector, timeout=5000)
            except Exception:
                return ""
        return self._run(_do_html)

    def get_attribute(self, selector: str, attribute: str) -> str:
        """Get an attribute value from an element.

        Args:
            selector: CSS selector.
            attribute: Attribute name.

        Returns:
            Attribute value or empty string.
        """
        self._ensure_started()
        def _do_attr() -> str:
            try:
                return self._page.get_attribute(selector, attribute, timeout=5000) or ""
            except Exception:
                return ""
        return self._run(_do_attr)

    def query_selector_all(self, selector: str) -> list[dict[str, str]]:
        """Query all matching elements and return their text + attributes.

        Args:
            selector: CSS selector.

        Returns:
            List of dicts with text, href, id, class info.
        """
        self._ensure_started()
        def _do_query() -> list[dict[str, str]]:
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
        return self._run(_do_query)

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
        def _do_screenshot() -> str:
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
        return self._run(_do_screenshot)

    # ----- JavaScript -----

    def evaluate_js(self, expression: str) -> Any:
        """Evaluate a JavaScript expression in the page context.

        Args:
            expression: JavaScript to evaluate.

        Returns:
            Result of the expression.
        """
        self._ensure_started()
        def _do_eval() -> Any:
            try:
                return self._page.evaluate(expression)
            except Exception as exc:
                logger.error("JS evaluation failed: %s", exc)
                return None
        return self._run(_do_eval)

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
        def _do_wait() -> bool:
            try:
                self._page.wait_for_selector(selector, timeout=timeout)
                return True
            except Exception:
                return False
        return self._run(_do_wait)

    def wait_for_navigation(self, timeout: int = 10000) -> bool:
        """Wait for a navigation to complete.

        Args:
            timeout: Timeout in milliseconds.

        Returns:
            True if navigation completed.
        """
        self._ensure_started()
        def _do_wait_nav() -> bool:
            try:
                self._page.wait_for_load_state("domcontentloaded", timeout=timeout)
                return True
            except Exception:
                return False
        return self._run(_do_wait_nav)

    # ----- Internal -----

    def _ensure_started(self) -> None:
        """Ensure the browser is started and healthy."""
        def _check_healthy() -> bool:
            if not self._started or not self._browser or not self._page:
                return False
            try:
                if self._page.is_closed() or not self._browser.is_connected():
                    return False
                return True
            except Exception:
                return False

        if not self._run(_check_healthy):
            self._started = False
            if not self.start():
                err = self._last_error or "Install playwright and chromium."
                raise RuntimeError(f"Browser is not available: {err}")
