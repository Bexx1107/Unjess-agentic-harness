"""NiceGUI application entry point — wires the agent to the web UI.

:func:`launch` is the single public entry point. It handles state
initialization, agent threading, and route registration.
"""

from __future__ import annotations

import logging
import threading
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
        # In GUI mode, don't default to CWD (which would be the exe's dir).
        # Start with no workspace — user picks one via New Conversation dialog.
        import sys
        if getattr(sys, 'frozen', False) or str(settings.workspace) == ".":
            _ws = ""
            # Set settings workspace to home dir so agent doesn't use exe dir
            import os
            settings.workspace = os.path.expanduser("~")
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
    from unjess.config import is_first_run

    ga = _GUIApp(agent, settings, cmd_ctx, router, conv_logger, memory_store)

    @ui.page("/")
    def chat_page() -> None:
        if is_first_run():
            ui.navigate.to("/onboarding")
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
    run_kwargs: dict = dict(
        title="Unjess",
        dark=True,
        reload=False,
    )
    if _favicon:
        run_kwargs["favicon"] = _favicon

    if native:
        from nicegui import native as nicegui_native
        run_kwargs["native"] = True
        run_kwargs["port"] = nicegui_native.find_open_port()
        run_kwargs["window_size"] = (1280, 800)
    else:
        run_kwargs["port"] = 8080

    ui.run(**run_kwargs)
