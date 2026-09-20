"""NiceGUI application entry point — wires the agent to the web UI.

:func:`launch` is the single public entry point. It handles state
initialization, agent threading, and route registration.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import ui

from unjess.gui.display import GUIDisplay
from unjess.gui.input_handler import GUIInput
from unjess.gui.state import AppState, Notification

if TYPE_CHECKING:
    from unjess.agent import Agent
    from unjess.commands import CommandContext
    from unjess.config import Settings
    from unjess.conversation_logger import ConversationLogger
    from unjess.llm.router import ProviderRouter
    from unjess.memory import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application object
# ---------------------------------------------------------------------------

class _GUIApp:
    """Internal application object — coordinates threads and state."""

    def __init__(
        self,
        agent: Agent,
        settings: Settings,
        cmd_ctx: CommandContext,
        router: ProviderRouter,
        conv_logger: ConversationLogger,
        memory_store: "MemoryStore | None" = None,
    ) -> None:
        self.agent = agent
        self.settings = settings
        self.cmd_ctx = cmd_ctx
        self.router = router
        self.conv_logger = conv_logger
        # In GUI mode, don't default to CWD or Home dir (which would expose all PC files).
        # Default Quick Chat to an isolated workspace folder inside ~/.unjess/workspaces/
        home_str = os.path.expanduser("~")
        if getattr(sys, 'frozen', False) or str(settings.workspace) in (".", "", home_str):
            default_ws = Path.home() / ".unjess" / "workspaces" / "quick_chat"
            default_ws.mkdir(parents=True, exist_ok=True)
            settings.workspace = default_ws
            _ws = str(default_ws)
        else:
            _ws = str(settings.workspace)

        # State — store settings ref so the dialog can read/write it
        self.state = AppState(
            model=settings.model,
            provider=settings.provider,
            workspace=_ws,
            settings=settings,
            memory_store=memory_store,
            subagent_manager=getattr(agent, '_subagent_manager', None),
            task_manager=getattr(agent, '_task_manager', None),
            scheduler=getattr(agent, '_scheduler', None),
            conv_logger=conv_logger,
        )

        # Infrastructure
        def _notify() -> None:
            self.state.dirty = True

        self.display = GUIDisplay(self.state, notify_fn=_notify)
        self.input_handler = GUIInput()

        # Threading
        self._agent_thread: threading.Thread | None = None

        # Wire agent to GUI display and input handler
        self.agent._display = self.display
        self.agent._input_handler = self.input_handler
        # Register ask_question tool (must be AFTER _input_handler is set)
        self.agent.register_ask_question()
        # Also wire the permission manager to use GUI input (not terminal input())
        if hasattr(self.agent, '_permissions'):
            self.agent._permissions._input_handler = self.input_handler

    def run_agent(self, message: str, images: list[dict[str, str]] | None = None) -> None:
        """Run agent turn in a background thread."""
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self.state.notifications.append(
                Notification(level="warning", message="Agent is still working… please wait.")
            )
            self.state.dirty = True
            return

        self.state.is_thinking = True
        self.state.dirty = True

        # Tag spawned subagents/tasks with this conversation ID
        conv_id = self.state.current_conversation_id
        if hasattr(self.agent, "_subagent_manager") and self.agent._subagent_manager:
            self.agent._subagent_manager.current_parent_id = conv_id
        if self.state.task_manager:
            self.state.task_manager.current_parent_id = conv_id

        def _turn() -> None:
            try:
                self.agent.run(message, images=images or [])
            except Exception as e:
                logger.exception("Agent turn failed")
                self.state.notifications.append(
                    Notification(level="negative", message=f"Error: {e}")
                )
            finally:
                self.state.is_thinking = False
                self.state.dirty = True

        self._agent_thread = threading.Thread(target=_turn, daemon=True)
        self._agent_thread.start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch(
    agent: Agent,
    settings: Settings,
    cmd_ctx: CommandContext,
    router: ProviderRouter,
    conv_logger: ConversationLogger,
    native: bool = False,
    memory_store: "MemoryStore | None" = None,
) -> None:
    """Launch the NiceGUI application.

    Args:
        agent: The Agent instance.
        settings: Current settings.
        cmd_ctx: Command context.
        router: Provider router.
        conv_logger: Logger.
        native: If True, launch in a native window instead of browser.
        memory_store: Cross-session memory store for conversation history.
    """
    from unjess.gui.pages.chat import setup_chat_page
    from unjess.gui.pages.onboarding import setup_onboarding_page
    from unjess.gui.pwa import register_pwa_routes, inject_pwa_meta
    from unjess.gui.auth import is_authenticated, render_pin_auth_screen
    from unjess.config import is_first_run

    ga = _GUIApp(agent, settings, cmd_ctx, router, conv_logger, memory_store)

    # Register PWA routes (/manifest.json, /sw.js)
    try:
        register_pwa_routes()
    except Exception as exc:
        logger.debug("Failed to register PWA routes: %s", exc)

    @ui.page("/")
    def chat_page() -> None:
        inject_pwa_meta()
        if is_first_run():
            ui.navigate.to("/onboarding")
            return

        if not is_authenticated(ga.settings):
            render_pin_auth_screen(
                settings=ga.settings,
                on_success=lambda: ui.navigate.to("/"),
            )
            return

        setup_chat_page(
            state=ga.state,
            input_handler=ga.input_handler,
            agent=ga.agent,
            cmd_ctx=ga.cmd_ctx,
            run_agent=ga.run_agent,
            settings=ga.settings,
            router=ga.router,
        )

    @ui.page("/onboarding")
    def onboarding_page() -> None:
        inject_pwa_meta()
        setup_onboarding_page(ga.state, ga.settings, ga.router)

    # Resolve favicon/icon paths
    from pathlib import Path as _P
    import sys as _sys
    _icon_candidates = [
        _P(getattr(_sys, '_MEIPASS', '')) / "assets",
        _P(_sys.executable).parent / "assets",
        _P(__file__).resolve().parent.parent.parent / "assets",
    ]
    _assets = next((c for c in _icon_candidates if c.exists()), None)
    _favicon = str(_assets / "icon_mark_white.svg") if _assets else None

    # Start server
    host = "0.0.0.0" if getattr(settings, "allow_network_access", True) else "127.0.0.1"
    target_port = getattr(settings, "network_port", 8080)

    # Check port availability to avoid crash if 8080 is held by background process
    import socket

    def _get_or_create_storage_secret() -> str:
        """Retrieve or generate a persistent local storage secret for NiceGUI."""
        import secrets

        secret_file = Path.home() / ".unjess" / ".session_secret"
        try:
            if secret_file.is_file():
                secret = secret_file.read_text(encoding="utf-8").strip()
                if secret:
                    return secret
            secret = secrets.token_hex(32)
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            secret_file.write_text(secret, encoding="utf-8")
            try:
                secret_file.chmod(0o600)
            except OSError:
                pass
            return secret
        except Exception:
            return secrets.token_hex(32)

    def _get_open_port(desired_port: int) -> int:
        for p in range(desired_port, desired_port + 50):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _sock:
                    _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    _sock.bind((host, p))
                    return p
            except OSError:
                continue
        return desired_port

    port = _get_open_port(target_port)

    run_kwargs: dict = dict(
        title="Unjess",
        dark=True,
        reload=False,
        host=host,
        port=port,
        storage_secret=_get_or_create_storage_secret(),
    )
    if _favicon:
        run_kwargs["favicon"] = _favicon

    if native:
        run_kwargs["native"] = True
        run_kwargs["window_size"] = (1280, 800)

    ui.run(**run_kwargs)
