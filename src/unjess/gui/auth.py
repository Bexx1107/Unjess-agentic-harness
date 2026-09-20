"""PIN Authentication module for Unjess Mobile Companion.

Provides PIN verification, touch PIN pad UI, and session authorization.
"""

from __future__ import annotations

import logging
from typing import Callable, TYPE_CHECKING
from nicegui import app, ui

if TYPE_CHECKING:
    from unjess.config import Settings

logger = logging.getLogger(__name__)


def get_all_local_ips() -> set[str]:
    """Gather all local IP addresses for host machine to auto-bypass PIN on local PC."""
    import socket
    ips = {"127.0.0.1", "localhost", "::1", "testclient", "0.0.0.0"}
    try:
        hostname = socket.gethostname()
        ips.add(socket.gethostbyname(hostname))
        for ip in socket.gethostbyname_ex(hostname)[2]:
            ips.add(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips


_LOCAL_IPS = get_all_local_ips()


def is_localhost_client() -> bool:
    """Check if the requesting client is running locally on PC."""
    try:
        from nicegui import context
        client = context.get_client()
        if client:
            ip = getattr(client, "ip", "")
            if ip and ip in _LOCAL_IPS:
                return True
            if hasattr(client, "request") and client.request and client.request.client:
                host = client.request.client.host
                if host in _LOCAL_IPS:
                    return True
    except Exception:
        pass
    return False


def is_authenticated(settings: Settings) -> bool:
    """Check if the current session is authenticated with PIN.

    Localhost (PC app window) is ALWAYS auto-authenticated.
    Remote PIN is only required if settings.enable_pin_auth is True AND mobile_pin is set.
    """
    # Local PC never requires PIN
    if is_localhost_client():
        return True

    # PIN auth must be explicitly enabled by user
    if not getattr(settings, "enable_pin_auth", False):
        return True

    # If no PIN is configured, access is unrestricted
    if not settings.mobile_pin or not settings.mobile_pin.strip():
        return True

    # Check session storage safely
    try:
        return bool(app.storage.user.get("authenticated", False))
    except Exception as exc:
        logger.debug("Storage user access check failed: %s", exc)
        return False


def render_pin_auth_screen(settings: Settings, on_success: Callable[[], None]) -> None:
    """Render a mobile-friendly 4-digit PIN authentication pad."""
    entered_digits: list[str] = []

    with ui.column().classes(
        "w-full min-h-screen bg-[#0f0f12] text-white flex flex-col items-center justify-center p-4"
    ):
        with ui.card().classes(
            "w-full max-w-sm bg-[#18181c] border border-gray-800 rounded-2xl p-6 shadow-2xl flex flex-col items-center gap-6"
        ):
            # Header icon & title
            ui.icon("lock", size="48px", color="primary").classes("mt-2")
            ui.label("Unjess Companion").classes("text-2xl font-bold tracking-tight text-gray-100")
            ui.label("Enter your 4-digit PIN to access").classes("text-sm text-gray-400 text-center -mt-4")

            # PIN indicators (4 circles)
            dot_container = ui.row().classes("gap-4 my-2")
            dots: list[ui.element] = []
            with dot_container:
                for _ in range(4):
                    dots.append(
                        ui.element("div").classes(
                            "w-4 h-4 rounded-full border-2 border-gray-600 transition-all duration-200"
                        )
                    )

            # Error message label
            err_label = ui.label("").classes("text-red-400 text-sm h-5 font-medium text-center")

            def update_dots() -> None:
                for i, dot in enumerate(dots):
                    if i < len(entered_digits):
                        dot.classes(replace="w-4 h-4 rounded-full bg-white border-white scale-110")
                    else:
                        dot.classes(replace="w-4 h-4 rounded-full border-2 border-gray-600")

            def verify_pin() -> None:
                pin_str = "".join(entered_digits)
                if pin_str == settings.mobile_pin.strip():
                    app.storage.user["authenticated"] = True
                    err_label.set_text("")
                    ui.notify("Access Granted!", type="positive", icon="lock_open")
                    on_success()
                else:
                    err_label.set_text("Incorrect PIN. Try again.")
                    entered_digits.clear()
                    update_dots()

            def press_key(digit: str) -> None:
                if len(entered_digits) < 4:
                    entered_digits.append(digit)
                    update_dots()
                    if len(entered_digits) == 4:
                        ui.timer(0.15, verify_pin, once=True)

            def press_backspace() -> None:
                if entered_digits:
                    entered_digits.pop()
                    update_dots()
                    err_label.set_text("")

            # 3x4 Touch Keypad
            with ui.grid(columns=3).classes("w-full gap-3 mt-2"):
                for num in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    ui.button(
                        num,
                        on_click=lambda n=num: press_key(n)
                    ).props("flat rounded").classes(
                        "h-14 text-xl font-semibold bg-gray-800/60 hover:bg-gray-700/80 active:scale-95 text-white transition"
                    )

                # Bottom row: Empty / 0 / Backspace
                ui.element("div")  # placeholder
                ui.button("0", on_click=lambda: press_key("0")).props("flat rounded").classes(
                    "h-14 text-xl font-semibold bg-gray-800/60 hover:bg-gray-700/80 active:scale-95 text-white transition"
                )
                ui.button(
                    icon="backspace",
                    on_click=press_backspace
                ).props("flat rounded").classes(
                    "h-14 text-xl bg-gray-800/60 hover:bg-gray-700/80 active:scale-95 text-gray-400 transition"
                )
