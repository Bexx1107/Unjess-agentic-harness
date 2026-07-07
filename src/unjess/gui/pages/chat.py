"""Main chat page — Antigravity 2.0 layout using Quasar primitives.

Uses NiceGUI's built-in layout system (ui.header, ui.left_drawer,
ui.right_drawer, ui.footer) instead of raw divs, which properly
handles fixed positioning, z-indexing, and scroll containment.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger(__name__)

from nicegui import ui

from unjess.gui.components.chat_message import render_message
from unjess.gui.components.approval_dialog import render_approval_dialog
from unjess.gui.components.project_browser import render_project_browser
from unjess.gui.components.session_panel import render_session_panel
from unjess.gui.components.new_conversation_dialog import create_new_conversation_dialog
from unjess.gui.components.spawn_dialog import render_spawn_dialog

if TYPE_CHECKING:
    from unjess.agent import Agent
    from unjess.commands import CommandContext
    from unjess.config import Settings
    from unjess.gui.input_handler import GUIInput
    from unjess.gui.state import AppState, ChatMessage
    from unjess.llm.router import ProviderRouter


def _resize_image(
    data: bytes, mime_type: str, max_size: int = 1024
) -> tuple[bytes, str]:
    """Resize an image to fit within max_size pixels (longest edge).

    Args:
        data: Raw image bytes.
        mime_type: MIME type (e.g. 'image/png').
        max_size: Maximum dimension in pixels.

    Returns:
        Tuple of (resized_bytes, mime_type). Returns original data
        if resize fails or image is already small enough.
    """
    if not mime_type.startswith("image/"):
        return data, mime_type
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if max(w, h) <= max_size:
            return data, mime_type
        # Scale down preserving aspect ratio
        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_h, new_w = max_size, int(w * max_size / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        # Use PNG for RGBA (screenshots), JPEG for photos
        if img.mode == "RGBA":
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue(), "image/png"
        else:
            img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue(), "image/jpeg"
    except ImportError:
        logger.warning("PIL not available — skipping image resize")
        return data, mime_type
    except Exception as exc:
        logger.warning("Image resize failed: %s", exc)
        return data, mime_type


def setup_chat_page(
    state: "AppState",
    input_handler: "GUIInput",
    agent: "Agent",
    cmd_ctx: "CommandContext",
    run_agent: Callable[..., None],
    settings: "Settings | None" = None,
    router: "ProviderRouter | None" = None,
) -> None:
    """Build the chat page using Quasar's QLayout system."""
    from unjess.gui.state import ChatMessage, Notification

    # ── Theme ─────────────────────────────────────────────────────────
    _accent = getattr(settings, "accent_color", "#666666")
    ui.colors(primary=_accent, dark="#141414", dark_page="#0d0d0d")
    ui.dark_mode(True)

    # Inject dynamic accent CSS overrides
    from unjess.gui.accent import accent_css
    ui.add_css(accent_css(_accent))

    # Load fonts + icons
    ui.add_head_html('''
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    ''')

    # Kill body scroll, remove default padding
    ui.query("body").classes("overflow-hidden")
    ui.query(".nicegui-content").classes("p-0")

    # Minimal CSS overrides
    ui.add_css("""
        * { font-family: 'Inter', -apple-system, sans-serif; }
        .q-scrollarea__thumb { background: #444 !important; border-radius: 4px; width: 4px !important; }
        .q-field--outlined .q-field__control:before { border-color: #333 !important; }
        .q-drawer { border-color: #222 !important; overflow-x: hidden !important; }
        /* Thin scrollbars globally */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
        /* Hide drawer scrollbar until hover */
        .q-drawer .q-scrollarea__thumb { opacity: 0; transition: opacity 0.2s; }
        .q-drawer:hover .q-scrollarea__thumb { opacity: 1; }
    """)

    # ── Settings dialog (popup) ───────────────────────────────────────
    settings_dialog = None
    if settings:
        try:
            from unjess.gui.components.settings_dialog import create_settings_dialog
            settings_dialog = create_settings_dialog(
                settings, state, router
            )
        except Exception as exc:
            logger.warning("Failed to create settings dialog: %s", exc)

    def _open_settings() -> None:
        if settings_dialog:
            settings_dialog.open()
        else:
            ui.notify("Settings not available", type="warning")

    # ── Approval dialog ───────────────────────────────────────────────
    approval_dialog = render_approval_dialog(input_handler)

    # ── Spawn dialog ──────────────────────────────────────────────────
    spawn_dialog = render_spawn_dialog(input_handler, router) if router else None

    # ── Message send handler ──────────────────────────────────────────
    def _send_message(text: str, images: list[dict[str, str]] | None = None) -> None:
        if not text or not text.strip():
            if not images:
                return

        text = text.strip()

        # ── Slash command interception ────────────────────────────
        if text.startswith("/"):
            from unjess.commands import dispatch
            # Show the command in chat as a user message
            state.append_message(ChatMessage(role="user", content=text))
            state.dirty = True

            # GUI-friendly display wrapper that captures clean output
            import re as _re

            class _GUICommandDisplay:
                """Captures command output as clean text for GUI chat."""

                def __init__(self) -> None:
                    self._lines: list[str] = []

                @property
                def console(self) -> "_GUICommandDisplay":
                    return self

                @property
                def _verbose(self) -> bool:
                    return False

                def show_info(self, text: str) -> None:
                    self._lines.append(text)

                def show_error(self, text: str) -> None:
                    self._lines.append(f"⚠️ {text}")

                def show_success(self, text: str) -> None:
                    self._lines.append(f"✅ {text}")

                def print(self, *args: object, **kwargs: object) -> None:
                    for arg in args:
                        if isinstance(arg, str):
                            # Strip rich markup tags like [bold], [dim], [/]
                            clean = _re.sub(r'\[/?[^\]]*\]', '', arg)
                            if clean.strip():
                                self._lines.append(clean.strip())
                        else:
                            # Rich renderable (Panel, Table, etc.)
                            # Render to plain text via a real Console
                            import io as _io
                            from rich.console import Console as _Console
                            _buf = _io.StringIO()
                            _c = _Console(file=_buf, force_terminal=False, no_color=True, width=80)
                            _c.print(arg)
                            rendered = _buf.getvalue().strip()
                            if rendered:
                                # Clean up box-drawing chars for GUI
                                clean_lines = []
                                for line in rendered.split("\n"):
                                    # Strip box-drawing borders
                                    stripped = line.strip("│┃║┌┐└┘├┤┬┴┼─━═╔╗╚╝╠╣╦╩╬ ")
                                    # Skip lines that are purely border
                                    if stripped and not all(c in "─━═╌╍┄┅┈┉ " for c in stripped):
                                        self._lines.append(stripped)

                def get_output(self) -> str:
                    return "\n".join(self._lines)

                # Support any attr access without crashing
                def __getattr__(self, name: str) -> object:
                    return lambda *a, **kw: None

            gui_display = _GUICommandDisplay()
            original_display = cmd_ctx.display
            cmd_ctx.display = gui_display
            try:
                dispatch(text, cmd_ctx)
            except Exception as exc:
                gui_display._lines.append(f"⚠️ Command error: {exc}")
            finally:
                cmd_ctx.display = original_display

            # Show captured output as assistant message
            output = gui_display.get_output().strip()
            if output:
                state.append_message(ChatMessage(
                    role="assistant",
                    content=output,
                ))
            state.dirty = True
            return

        # If the user is viewing a different conversation than the agent has loaded,
        # switch the agent's context now (only on actual message send — not on browse)
        if state.current_conversation_id != state._agent_conversation_id:
            agent.clear_history()
            state.is_thinking = False
            state.files_changed.clear()
            state._agent_conversation_id = state.current_conversation_id

        # Tag the message with index; checkpoint_id will be set after agent creates it
        _user_msg = ChatMessage(
            role="user",
            content=text,
            images=images or [],
            msg_index=len(state.messages),
        )
        state.messages.append(_user_msg)

        # Set checkpoint_id after undo checkpoint is created by agent
        def _tag_checkpoint() -> None:
            """Tag the user message with its checkpoint ID after agent creates it."""
            import time
            time.sleep(0.5)  # wait for agent.run() to create the checkpoint
            if hasattr(agent, '_undo_manager') and agent._undo_manager:
                entries = agent._undo_manager.list_entries(n=1)
                if entries:
                    _user_msg.checkpoint_id = entries[0].id

        import threading
        threading.Thread(target=_tag_checkpoint, daemon=True).start()

        state.is_thinking = True
        state.dirty = True
        run_agent(text, images=images or [])

    def _unsend_last() -> None:
        """Remove the last user message and any assistant response after it."""
        if not state.messages:
            return
        # Remove from end: assistant messages, then the last user message
        while state.messages and state.messages[-1].role == "assistant":
            state.messages.pop()
        if state.messages and state.messages[-1].role == "user":
            state.messages.pop()
        # Also remove from agent conversation history
        while agent.conversation and agent.conversation[-1].get("role") in ("assistant", "tool"):
            agent.conversation.pop()
        if agent.conversation and agent.conversation[-1].get("role") == "user":
            agent.conversation.pop()
        state.dirty = True
        ui.notify("Last message unsent", type="info", position="top")

    def _stop_generation() -> None:
        """Abort the running agent by setting the abort flag."""
        agent._abort_requested = True
        state.is_thinking = False
        state.dirty = True
        ui.notify("Generation stopped", type="warning", position="top")

    def _rewind_to_message(msg_index: int, checkpoint_id: str) -> None:
        """Rewind conversation and codebase to a specific message point.

        Truncates the conversation to the message before msg_index,
        and reverts the codebase using the undo manager.
        """
        if state.is_thinking:
            ui.notify("Cannot rewind while generating", type="warning", position="top")
            return

        # Revert the codebase using the undo manager
        if hasattr(agent, '_undo_manager') and agent._undo_manager:
            # Count how many checkpoints to undo
            # Each user message creates one checkpoint, so count user messages after msg_index
            user_msgs_after = sum(
                1 for m in state.messages[msg_index:]
                if m.role == "user" and m.checkpoint_id
            )
            if user_msgs_after > 0:
                success, result_msg = agent._undo_manager.undo(steps=user_msgs_after)
                if not success:
                    ui.notify(f"Undo failed: {result_msg}", type="negative", position="top")
                    return

        # Truncate UI messages — keep everything before msg_index
        state.messages = state.messages[:msg_index]

        # Truncate agent conversation history to match
        # Count the remaining user messages and keep matching entries
        remaining_user_count = sum(1 for m in state.messages if m.role == "user")
        trimmed_convo: list[dict] = []
        user_seen = 0
        for entry in agent.conversation:
            if entry.get("role") == "user":
                user_seen += 1
                if user_seen > remaining_user_count:
                    break
            trimmed_convo.append(entry)
        agent.conversation = trimmed_convo

        state.dirty = True
        ui.notify(
            f"Rewound to message {msg_index} — {user_msgs_after} change(s) undone",
            type="positive", position="top",
        )

    # Register the rewind callback for the message renderer
    from unjess.gui.components.chat_message import set_rewind_callback
    set_rewind_callback(_rewind_to_message)

    def _new_conversation() -> None:
        state.clear_messages()
        agent.clear_history()
        state.is_thinking = False
        state.files_changed.clear()
        state.dirty = True

    def _on_new_conversation(workspace_path: str | None) -> None:
        """Handle new conversation from workspace picker dialog."""
        _new_conversation()
        if workspace_path:
            state.workspace = workspace_path
            ws_name = workspace_path.rstrip("/\\").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            title = f"New chat in {ws_name}"
            if settings:
                settings.workspace = workspace_path
            # Update file tools to operate in the new workspace
            from pathlib import Path as _Path
            from unjess.tools.file_tools import update_workspace
            update_workspace(_Path(workspace_path))
        else:
            title = "Quick chat"
            # Clear workspace so the agent doesn't reference CWD
            state.workspace = ""
            if settings:
                import os
                home = os.path.expanduser("~")
                settings.workspace = home
                # Update file tools to use home dir for quick chats
                from pathlib import Path as _Path
                from unjess.tools.file_tools import update_workspace
                update_workspace(_Path(home))

        import uuid
        import time
        conv_id = str(uuid.uuid4())
        created = time.time()

        # Always add to local list for immediate sidebar display
        state.local_conversations.append({
            "conversation_id": conv_id,
            "title": title,
            "workspace": workspace_path or "",
            "model": state.model,
            "created_at": created,
        })

        # Also persist to memory store if available
        if state.memory_store:
            try:
                from unjess.memory import ConversationSummary
                summary = ConversationSummary(
                    conversation_id=conv_id,
                    title=title,
                    workspace=workspace_path or "",
                    model=state.model,
                    created_at=created,
                )
                state.memory_store.add_summary(summary)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Failed to save conversation: %s", e)
        # Track current conversation for auto-rename
        state.current_conversation_id = conv_id
        state._agent_conversation_id = conv_id  # agent context is fresh for this conversation
        state._conversation_renamed = False

        # Refresh the project browser in the sidebar
        _refresh_project_browser()

        # Persist to disk so conversations survive restarts
        state.save_conversations()

        # Auto-refresh MCP tools on new conversation
        def _auto_refresh_mcp() -> None:
            """Refresh MCP tool list in the background."""
            try:
                from unjess.commands import _get_command_context
                ctx = _get_command_context()
                if hasattr(ctx, '_mcp_manager'):
                    mgr = ctx._mcp_manager
                    if mgr.connected_servers:
                        results = mgr.refresh_tools(tool_registry=agent.tool_registry)
                        total = sum(results.values())
                        if total > 0:
                            state.notifications.append(
                                Notification(level="info", message=f"MCP tools refreshed: {total} tools")
                            )
            except Exception:
                pass  # non-critical

        import threading
        threading.Thread(target=_auto_refresh_mcp, daemon=True).start()

        ui.notify(title, type="info", position="top")

    def _show_conversation_history() -> None:
        """Open a dialog listing past conversations."""
        summaries: list = []
        if state.memory_store:
            try:
                summaries = state.memory_store.get_recent_summaries(count=50)
            except Exception as exc:
                logger.warning("Failed to load conversation summaries: %s", exc)

        with ui.dialog() as dlg, ui.card().style(
            "background: #1a1a1a; min-width: 450px; max-height: 70vh;"
        ):
            ui.label("Conversation History").classes(
                "text-sm font-semibold text-gray-300 mb-2"
            )
            if not summaries:
                ui.label("No conversation history yet").classes(
                    "text-xs text-gray-500 py-4"
                )
            else:
                with ui.scroll_area().style("max-height: 50vh; width: 100%;"):
                    for s in summaries:
                        title = (s.title or s.conversation_id)[:60]
                        ws = s.workspace.replace("\\", "/").rsplit("/", 1)[-1] if s.workspace else "Quick Chat"
                        with ui.button(
                            on_click=lambda _e, _cid=s.conversation_id, _dlg=dlg: (
                                _dlg.close(),
                                _on_select_conversation(_cid),
                            ),
                        ).props("flat dense no-caps align=left").classes("w-full").style(
                            "text-transform: none; justify-content: flex-start;"
                        ):
                            with ui.column().classes("gap-0"):
                                ui.label(title).classes("text-sm text-gray-300")
                                ui.label(f"{ws} · {s.message_count} messages").classes(
                                    "text-xs text-gray-600"
                                )
            with ui.row().classes("w-full justify-end mt-3"):
                ui.button("Close", on_click=dlg.close).props(
                    "flat dense no-caps"
                ).classes("text-gray-500")
        dlg.open()

    def _show_scheduled_tasks() -> None:
        """Show a dialog listing scheduled tasks."""
        tasks: list = []
        if state.scheduler:
            try:
                tasks = state.scheduler.list_tasks()
            except Exception as exc:
                logger.warning("Failed to list scheduled tasks: %s", exc)

        with ui.dialog() as dlg, ui.card().style(
            "background: #1a1a1a; min-width: 400px; max-height: 60vh;"
        ):
            ui.label("Scheduled Tasks").classes(
                "text-sm font-semibold text-gray-300 mb-2"
            )
            if not tasks:
                ui.label("No scheduled tasks").classes(
                    "text-xs text-gray-500 py-4"
                )
            else:
                for task in tasks:
                    name = getattr(task, "name", str(task))
                    status = getattr(task, "status", "unknown")
                    with ui.row().classes("items-center gap-2 w-full py-1"):
                        color = "#4caf50" if status == "RUNNING" else "#616161"
                        ui.icon("schedule", size="14px").style(f"color: {color}")
                        ui.label(str(name)[:50]).classes("text-sm text-gray-300 flex-1")
                        ui.label(str(status)).classes("text-xs text-gray-600")
            with ui.row().classes("w-full justify-end mt-3"):
                ui.button("Close", on_click=dlg.close).props(
                    "flat dense no-caps"
                ).classes("text-gray-500")
        dlg.open()

    def _on_select_conversation(conversation_id: str) -> None:
        """Handle clicking a past conversation — display-only, doesn't touch agent context."""
        from pathlib import Path
        conversations_dir = Path.home() / ".unjess" / "conversations"

        # Save current conversation messages to cache before switching
        if state.current_conversation_id and state.messages:
            state._message_cache[state.current_conversation_id] = list(state.messages)

        # Try multiple ID patterns since CLI/GUI use different formats
        candidates = [
            conversations_dir / conversation_id,
        ]
        if conversation_id.startswith("session-"):
            candidates.append(conversations_dir / conversation_id[8:])
        if "-" in conversation_id:
            candidates.append(conversations_dir / conversation_id.replace("-", ""))
            candidates.append(conversations_dir / conversation_id.replace("-", "")[:12])

        transcript = None
        for cand in candidates:
            t = cand / "transcript.jsonl"
            if t.exists():
                transcript = t
                break

        # Only update the DISPLAY — don't touch agent context
        state.clear_messages()
        state.current_conversation_id = conversation_id
        state._conversation_renamed = True

        def _switch_workspace(ws: str) -> None:
            """Update both display and agent workspace when switching conversations."""
            state.workspace = ws
            if settings:
                settings.workspace = ws
            # Update file tools to operate in the correct workspace
            from unjess.tools.file_tools import update_workspace
            update_workspace(Path(ws))

        # 1. Check in-memory cache first (GUI sessions)
        if conversation_id in state._message_cache:
            state.messages.extend(state._message_cache[conversation_id])
            # Also restore workspace from conversation metadata
            for lc in state.local_conversations:
                if lc.get("conversation_id") == conversation_id:
                    ws = lc.get("workspace", "")
                    if ws:
                        _switch_workspace(ws)
                    break
            ui.notify(f"Switched to: {conversation_id[:12]}…", type="info", position="top")
        # 2. Try to load from transcript file (CLI sessions)
        elif transcript:
            import json
            try:
                for line in transcript.read_text(encoding="utf-8").strip().splitlines():
                    entry = json.loads(line)
                    if entry.get("type") == "USER_INPUT":
                        state.append_message(
                            ChatMessage(role="user", content=entry.get("content", ""))
                        )
                    elif entry.get("type") == "MODEL_RESPONSE":
                        content = entry.get("content", "")
                        if content:
                            state.append_message(
                                ChatMessage(role="assistant", content=content)
                            )
                ui.notify(f"Loaded conversation {conversation_id[:8]}…", type="info", position="top")
            except Exception as e:
                ui.notify(f"Error loading transcript: {e}", type="negative")
        else:
            # New conversation — no transcript yet, just switch to it
            title = conversation_id[:8]
            # Check local conversations first, then memory store
            for lc in state.local_conversations:
                if lc.get("conversation_id") == conversation_id:
                    title = lc.get("title", title)
                    ws = lc.get("workspace", "")
                    if ws:
                        _switch_workspace(ws)
                    break
            else:
                # Try memory store for workspace/title
                if state.memory_store:
                    for s in state.memory_store.get_recent_summaries(count=50):
                        if s.conversation_id == conversation_id:
                            title = s.title or title
                            if s.workspace:
                                _switch_workspace(s.workspace)
                            break
            ui.notify(f"Switched to: {title}", type="info", position="top")

        state.dirty = True

    # Mutable ref so _refresh can find the container after it's created
    _browser_ref: dict[str, Any] = {}

    def _refresh_project_browser() -> None:
        """Clear and re-render the project browser sidebar."""
        container = _browser_ref.get("container")
        if container is None:
            return
        container.clear()
        with container:
            render_project_browser(state, _on_select_conversation, _refresh_project_browser)

    # ══════════════════════════════════════════════════════════════════
    # GLOBAL CSS OVERRIDES
    # ══════════════════════════════════════════════════════════════════
    ui.add_head_html('''
    <style>
    /* Kill horizontal scrollbar inside left drawer */
    .q-drawer--left .q-scrollarea__container {
        overflow-x: hidden !important;
    }
    .q-drawer--left .q-scrollarea__content {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    /* Drawer resize — CSS custom properties enforced over Quasar */
    :root {
        --njss-left-w: 220px;
        --njss-right-w: 260px;
    }
    .q-drawer--left {
        width: var(--njss-left-w) !important;
    }
    .q-drawer--right {
        width: var(--njss-right-w) !important;
    }
    .q-header {
        left: var(--njss-left-w) !important;
        right: var(--njss-right-w) !important;
    }
    .q-footer {
        left: var(--njss-left-w) !important;
        right: var(--njss-right-w) !important;
    }
    .q-page-container {
        padding-left: var(--njss-left-w) !important;
        padding-right: var(--njss-right-w) !important;
    }
    </style>
    ''')

    # ══════════════════════════════════════════════════════════════════
    # STATIC ASSETS — serve logo files
    # ══════════════════════════════════════════════════════════════════
    from pathlib import Path as _Path
    from nicegui import app as _app
    import sys as _sys
    # Try multiple locations: PyInstaller bundle, exe directory, source tree
    _candidates = [
        _Path(getattr(_sys, '_MEIPASS', '')) / "assets",           # PyInstaller onefile
        _Path(_sys.executable).parent / "assets",                   # PyInstaller onedir
        _Path(__file__).resolve().parent.parent.parent.parent.parent / "assets",  # dev
    ]
    _assets_dir = next((c for c in _candidates if c.exists()), None)
    if _assets_dir:
        _app.add_static_files("/assets", str(_assets_dir))

    # ══════════════════════════════════════════════════════════════════
    # HEADER — breadcrumb bar
    # ══════════════════════════════════════════════════════════════════
    with ui.header().classes(
        "items-center no-wrap px-5 py-1"
    ).style("background: #0d0d0d; border-bottom: 1px solid #222; box-shadow: none"):
        left_toggle = ui.button(icon="menu", on_click=lambda: _toggle_left()).props(
            "flat dense round"
        ).classes("text-gray-500")
        ws = state.workspace or ""
        ws_name = ws.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1] if ws else "Quick Chat"
        _header_ws_label = ui.label(ws_name).classes("text-sm text-gray-400 font-medium ml-2")
        ui.space()
        model_text = state.model or "no model"
        prov_text = f"({state.provider})" if state.provider else ""
        _header_model_label = ui.label(f"{model_text} {prov_text}").classes(
            "text-xs text-gray-600 font-mono"
        )
        right_toggle = ui.button(icon="info", on_click=lambda: _toggle_right()).props(
            "flat dense round"
        ).classes("text-gray-500")

    # Track drawer widths so we can restore them after toggle
    _left_drawer_w = 220
    _right_drawer_w = 260

    def _toggle_left() -> None:
        """Toggle left drawer and update header/footer CSS offsets."""
        nonlocal _left_drawer_w
        left_drawer.toggle()
        # After toggle, the drawer value flips
        if left_drawer.value:
            # Drawer is now visible — restore saved width
            ui.run_javascript(
                f"document.documentElement.style.setProperty('--njss-left-w', '{_left_drawer_w}px')"
            )
        else:
            # Drawer is now hidden — collapse to 0
            # Save current width first
            ui.run_javascript(
                "document.documentElement.style.setProperty('--njss-left-w', '0px')"
            )

    def _toggle_right() -> None:
        """Toggle right drawer and update header/footer CSS offsets."""
        nonlocal _right_drawer_w
        right_drawer.toggle()
        if right_drawer.value:
            ui.run_javascript(
                f"document.documentElement.style.setProperty('--njss-right-w', '{_right_drawer_w}px')"
            )
        else:
            ui.run_javascript(
                "document.documentElement.style.setProperty('--njss-right-w', '0px')"
            )

    # ══════════════════════════════════════════════════════════════════
    # LEFT DRAWER — navigation sidebar
    # ══════════════════════════════════════════════════════════════════
    left_drawer = ui.left_drawer(
        fixed=True, top_corner=True, bottom_corner=True, value=True
    ).props("width=220 bordered breakpoint=0").classes("flex flex-col p-0").style("background: #111; overflow: hidden")

    # --- Drag handle for resizing the left drawer ---
    with left_drawer:
        ui.element("div").classes("drawer-resize-handle-left").style(
            "position: absolute; top: 0; right: 0; width: 4px; height: 100%; "
            "cursor: col-resize; z-index: 9999; background: transparent;"
        ).on("mouseenter", js_handler="""
            (e) => { e.target.style.background = 'rgba(var(--accent-rgb),0.5)'; }
        """).on("mouseleave", js_handler="""
            (e) => { if (!e.target._dragging) e.target.style.background = 'transparent'; }
        """)

    # JS for drag-to-resize (both drawers)
    ui.add_body_html('''
    <script>
    (function() {
        const root = document.documentElement;
        function initHandle(h, side) {
            let dragging = false, startX = 0, startW = 0;
            const aside = h.closest('aside.q-drawer');
            h.addEventListener('mousedown', (e) => {
                dragging = true; h._dragging = true;
                startX = e.clientX;
                startW = aside ? aside.offsetWidth : (side === 'left' ? 220 : 260);
                h.style.background = 'rgba(var(--accent-rgb),0.7)';
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            });
            document.addEventListener('mousemove', (e) => {
                if (!dragging) return;
                let newW;
                if (side === 'left') {
                    newW = Math.max(160, Math.min(500, startW + e.clientX - startX));
                    root.style.setProperty('--njss-left-w', newW + 'px');
                } else {
                    newW = Math.max(160, Math.min(500, startW - (e.clientX - startX)));
                    root.style.setProperty('--njss-right-w', newW + 'px');
                }
            });
            document.addEventListener('mouseup', () => {
                if (!dragging) return;
                dragging = false; h._dragging = false;
                h.style.background = 'transparent';
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            });
        }
        const poll = setInterval(() => {
            const lh = document.querySelector('.drawer-resize-handle-left');
            const rh = document.querySelector('.drawer-resize-handle-right');
            if (lh && rh) {
                clearInterval(poll);
                initHandle(lh, 'left');
                initHandle(rh, 'right');
            }
        }, 200);
    })();
    </script>
    ''')
    with left_drawer:

        # Logo: mark + "Unjess" in sidebar header
        with ui.row().classes("items-center px-4 pt-5 pb-4 gap-2 no-wrap"):
            ui.image("/assets/icon_mark_white.svg").style(
                "width: 24px; height: 24px;"
            )
            ui.label("Unjess").classes(
                "text-lg font-extrabold tracking-wide text-gray-200"
            )

        # New Conversation dialog
        new_conv_dialog = create_new_conversation_dialog(state, _on_new_conversation)

        # Action buttons
        with ui.column().classes("w-full px-4 gap-1"):
            ui.button("New Conversation", icon="add", on_click=new_conv_dialog.open).props(
                "unelevated dense no-caps"
            ).classes("w-full mb-1 rounded-lg text-gray-400 text-sm").style("background: #1a1a1a")

            ui.button("Conversation History", icon="history", on_click=_show_conversation_history).props(
                "flat dense no-caps align=left"
            ).classes("w-full text-gray-500 text-sm justify-start")

            ui.button("Scheduled Tasks", icon="schedule", on_click=_show_scheduled_tasks).props(
                "flat dense no-caps align=left"
            ).classes("w-full text-gray-500 text-sm justify-start")

        # Divider
        ui.separator().classes("mx-3 mt-1").style("background: #222")

        # Project browser (scrollable area — vertical only)
        with ui.scroll_area().classes("flex-grow w-full").style("overflow-x: hidden"):
            project_browser_container = ui.column().classes("w-full px-1 py-2")
            _browser_ref["container"] = project_browser_container
            with project_browser_container:
                render_project_browser(state, _on_select_conversation, _refresh_project_browser)

        # Divider before settings
        ui.separator().classes("mx-3 mt-1").style("background: #222")

        # Settings
        ui.button("Settings", icon="settings", on_click=_open_settings).props(
            "flat dense no-caps align=left"
        ).classes(
            "w-full text-gray-600 text-sm justify-start px-4 py-3 rounded-none"
        ).style("border-top: 1px solid #222")

    # ══════════════════════════════════════════════════════════════════
    # RIGHT DRAWER — status & tools
    # ══════════════════════════════════════════════════════════════════
    right_drawer = ui.right_drawer(
        fixed=True, top_corner=True, bottom_corner=True, value=True
    ).props("width=260 bordered breakpoint=0").classes("flex flex-col p-0").style("background: #111; overflow: hidden")

    # --- Drag handle for resizing the right drawer ---
    with right_drawer:
        ui.element("div").classes("drawer-resize-handle-right").style(
            "position: absolute; top: 0; left: 0; width: 4px; height: 100%; "
            "cursor: col-resize; z-index: 9999; background: transparent;"
        ).on("mouseenter", js_handler="""
            (e) => { e.target.style.background = 'rgba(var(--accent-rgb),0.5)'; }
        """).on("mouseleave", js_handler="""
            (e) => { if (!e.target._dragging) e.target.style.background = 'transparent'; }
        """)
    with right_drawer:
        with ui.scroll_area().classes("w-full flex-grow"):
            session_panel_container = ui.column().classes("w-full p-4 gap-0")
            with session_panel_container:
                render_session_panel(state)

    # ══════════════════════════════════════════════════════════════════
    # FOOTER — input bar pinned to bottom
    # ══════════════════════════════════════════════════════════════════

    # Pending image attachments (shared mutable list)
    _pending_images: list[dict[str, str]] = []
    _MAX_ATTACHMENTS = 5

    with ui.footer().classes(
        "px-5 py-3"
    ).style("background: #0d0d0d; border-top: 1px solid #222; box-shadow: none"):
        with ui.column().classes("w-full max-w-3xl mx-auto gap-1"):

            # Image preview strip (shows thumbnails of pending attachments)
            preview_row = ui.row().classes(
                "w-full flex-wrap gap-2 px-1 hidden" # Start hidden
            ).style("min-height: 0px;")

            def _rebuild_preview() -> None:
                """Rebuild the thumbnail preview strip."""
                preview_row.clear()
                if not _pending_images:
                    preview_row.classes(add="hidden", remove="flex")
                    return
                
                preview_row.classes(add="flex", remove="hidden")
                with preview_row:
                    for idx, img in enumerate(_pending_images):
                        with ui.row().classes(
                            "items-center gap-2 rounded-md p-1"
                        ).style(
                            "background: #222; border: 1px solid #444; position: relative; min-width: 48px; min-height: 48px;"
                        ):
                            if img["mime_type"] == "application/pdf":
                                ui.icon("description", size="24px").classes(
                                    "text-red-400 mx-auto"
                                )
                            else:
                                data_uri = f"data:{img['mime_type']};base64,{img['data']}"
                                ui.image(data_uri).style(
                                    "width: 40px; height: 40px; border-radius: 4px;"
                                ).classes("object-cover cursor-pointer").on(
                                    "click", lambda _, u=data_uri: ui.download(u)
                                )
                            
                            # Custom Close Button (div + icon for perfect centering)
                            with ui.element("div").classes(
                                "absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full "
                                "bg-[#333] border border-[#444] cursor-pointer "
                                "flex items-center justify-center z-10"
                            ).on("click", lambda _, i=idx: (
                                _pending_images.pop(i),
                                _rebuild_preview(),
                            )):
                                ui.icon("close", size="12px").classes("text-gray-400 hover:text-white")
                        
                        # (Next element...)

            with ui.row().classes(
                "w-full no-wrap items-center rounded-xl px-3 py-1 gap-2"
            ).style("background: #1a1a1a; border: 1px solid #333"):
                ui.icon("psychology", size="20px").classes("text-gray-600")

                text_input = ui.input(
                    placeholder="Ask anything, @ to mention, / for actions..."
                ).props("borderless dense dark").classes("flex-grow text-sm")

                # ── Attach button (📎) ──────────────────────────────
                def _handle_upload(e: Any) -> None:
                    """Process uploaded file(s) from the file picker."""
                    import base64
                    try:
                        if len(_pending_images) >= _MAX_ATTACHMENTS:
                            ui.notify(
                                f"Max {_MAX_ATTACHMENTS} attachments",
                                type="warning",
                            )
                            return
                        content_bytes = e.content.read()
                        mime = e.type or "application/octet-stream"
                        name = e.name or "upload"

                        # Resize images to max 1024px
                        if mime.startswith("image/"):
                            content_bytes, mime = _resize_image(
                                content_bytes, mime, max_size=1024
                            )

                        b64 = base64.b64encode(content_bytes).decode("ascii")
                        _pending_images.append({
                            "data": b64,
                            "mime_type": mime,
                            "name": name,
                        })
                        _rebuild_preview()
                        ui.notify(f"Attached: {name}", type="info", position="top")
                    except Exception as exc:
                        ui.notify(
                            f"Upload failed: {exc}",
                            type="negative", position="top",
                        )

                # Visible attach button triggers the hidden upload
                ui.button(
                    icon="attach_file",
                    on_click=lambda: upload.run_method("pickFiles"),
                ).props("round flat dense").classes(
                    "text-gray-500"
                ).style("min-width: 32px")

                # Send/Stop toggle button — swaps between send and stop
                send_stop_btn = ui.button(icon="arrow_upward").props(
                    "round dense unelevated"
                ).classes("bg-primary text-white w-8 h-8").style("min-width: 32px")

                def _do_send_or_stop() -> None:
                    """Send message or stop generation depending on state."""
                    if state.is_thinking:
                        _stop_generation()
                    else:
                        val = text_input.value
                        images = list(_pending_images)  # snapshot
                        if val or images:
                            text_input.value = ""
                            _pending_images.clear()
                            _rebuild_preview()
                            _send_message(val or "(see attached images)", images=images)

                send_stop_btn.on_click(_do_send_or_stop)
                text_input.on("keydown.enter", _do_send_or_stop)

            # Paste handler — Ctrl+V for clipboard images
            # Uses NiceGUI's js_handler + emit() bridge pattern
            def _handle_paste(e: Any) -> None:
                """Handle pasted image from clipboard."""
                import base64
                if len(_pending_images) >= _MAX_ATTACHMENTS:
                    ui.notify(
                        f"Max {_MAX_ATTACHMENTS} attachments",
                        type="warning",
                    )
                    return
                args = e.args if hasattr(e, "args") else e
                if not isinstance(args, dict) or not args.get("data"):
                    return
                data = args["data"]
                mime = args.get("mime_type", "image/png")
                name = args.get("name", "pasted.png")

                # Resize pasted image
                raw_bytes = base64.b64decode(data)
                raw_bytes, mime = _resize_image(raw_bytes, mime, max_size=1024)
                b64 = base64.b64encode(raw_bytes).decode("ascii")

                _pending_images.append({
                    "data": b64,
                    "mime_type": mime,
                    "name": name,
                })
                _rebuild_preview()

            text_input.on(
                "paste",
                _handle_paste,
                js_handler="""async (e) => {
                    const items = e.clipboardData?.items;
                    if (!items) return;
                    for (const item of items) {
                        if (item.type.startsWith('image/')) {
                            e.preventDefault();
                            e.stopPropagation();
                            const blob = item.getAsFile();
                            const reader = new FileReader();
                            reader.onload = () => {
                                const b64 = reader.result.split(',')[1];
                                emit({
                                    data: b64,
                                    mime_type: blob.type,
                                    name: 'clipboard_' + Date.now() + '.png'
                                });
                            };
                            reader.readAsDataURL(blob);
                            return;
                        }
                    }
                }""",
            )

            # Model label — uses a ref so _refresh can update it
            _footer_model_label = ui.label(state.model or "no model").classes(
                "text-[11px] text-gray-600 text-center w-full model-status-label"
            )

    # ── Autocomplete menu (positioned above footer) ───────────────────
    # Build slash command list from the registry
    from unjess.commands import get_all_commands, register_builtin_commands
    try:
        register_builtin_commands()
    except Exception as exc:
        logger.warning("Failed to register builtin commands: %s", exc)
    all_commands = get_all_commands()
    slash_items = [
        {"label": cmd.name, "desc": cmd.description, "usage": cmd.usage}
        for cmd in sorted(all_commands.values(), key=lambda c: c.name)
    ]

    # Build subcommand items from usage strings
    # e.g. usage="[list|connect|disconnect] [name]" -> ["list", "connect", "disconnect"]
    import re as _re_sub
    _sub_items: dict[str, list[dict]] = {}
    for item in slash_items:
        usage = item.get("usage", "")
        if not usage:
            continue
        # Extract options from bracket groups like [list|connect|disconnect]
        bracket_match = _re_sub.findall(r'\[([^\]]+)\]', usage)
        subs: list[str] = []
        for group in bracket_match:
            for part in group.split("|"):
                cleaned = part.strip().lstrip("-")
                if cleaned and not cleaned.startswith("<") and cleaned not in ("N",):
                    subs.append(part.strip())
        # Also extract bare <name> style args
        bare_match = _re_sub.findall(r'<([^>]+)>', usage)
        if subs or bare_match:
            label = item["label"]
            _sub_items[label] = [
                {"label": f"{label} {s}", "desc": ""}
                for s in subs
            ]
            if bare_match and not subs:
                _sub_items[label] = [
                    {"label": f"{label} <{b}>", "desc": ""}
                    for b in bare_match
                ]

    # @ mention items
    at_items = [
        {"label": "@file", "desc": "Mention a file from the workspace"},
        {"label": "@codebase", "desc": "Search the entire codebase"},
        {"label": "@web", "desc": "Search the web"},
        {"label": "@docs", "desc": "Reference documentation"},
        {"label": "@terminal", "desc": "Reference terminal output"},
    ]

    autocomplete_container = ui.column().classes(
        "fixed bottom-[70px] left-1/2 -translate-x-1/2 hidden"
    ).style(
        "background: #1a1a1a; border: 1px solid #333; border-radius: 8px; "
        "max-height: calc(100vh - 120px); overflow-y: auto; min-width: 340px; max-width: 500px; "
        "box-shadow: 0 -4px 20px rgba(0,0,0,0.5); z-index: 9999;"
    )
    _ac_items_container = None
    with autocomplete_container:
        _ac_items_container = ui.column().classes("w-full p-1 gap-0")

    def _show_autocomplete(prefix: str, items: list[dict]) -> None:
        """Populate and show the autocomplete menu."""
        filtered = [i for i in items if prefix.lower() in i["label"].lower()]
        if not filtered:
            autocomplete_container.classes(remove="flex", add="hidden")
            return
        _ac_items_container.clear()
        with _ac_items_container:
            for item in filtered[:12]:
                _label = item["label"]
                _desc = item.get("desc", "")
                _usage = item.get("usage", "")

                def _make_click(lbl: str, has_subs: bool = False) -> Callable:
                    def _click() -> None:
                        text_input.value = lbl + " "
                        if not has_subs:
                            autocomplete_container.classes(remove="flex", add="hidden")
                        text_input.run_method("focus")
                    return _click

                has_subs = _label in _sub_items
                with ui.button(on_click=_make_click(_label, has_subs)).props(
                    "flat dense no-caps align=left"
                ).classes("w-full px-3 py-1").style(
                    "justify-content: flex-start;"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(_label).style(
                            "font-size: 0.85rem; font-weight: 500; color: #e0e0e0;"
                        )
                        ui.label(_desc).style(
                            "font-size: 0.75rem; color: #888;"
                        )
                        if _usage:
                            ui.label(_usage).style(
                                "font-size: 0.7rem; color: #555; font-family: monospace; margin-left: auto;"
                            )
        autocomplete_container.classes(remove="hidden", add="flex")

    def _on_input_change(e: Any) -> None:
        """Handle input changes for autocomplete."""
        val = e.value if hasattr(e, 'value') else (e.args if hasattr(e, 'args') else "")
        val = val or ""
        if val.startswith("/"):
            # Check if user typed a known command + space -> show subcommands
            parts = val.split(None, 1)
            cmd_name = parts[0] if parts else val
            if len(parts) > 1 and cmd_name in _sub_items:
                # Show subcommand suggestions
                sub_prefix = val
                _show_autocomplete(sub_prefix, _sub_items[cmd_name])
            else:
                _show_autocomplete(val, slash_items)
        elif val.startswith("@"):
            _show_autocomplete(val, at_items)
        else:
            autocomplete_container.classes(remove="flex", add="hidden")

    text_input.on_value_change(_on_input_change)

    # ══════════════════════════════════════════════════════════════════
    # MAIN CONTENT — scrollable chat area (fills remaining space)
    # ══════════════════════════════════════════════════════════════════
    chat_scroll = ui.scroll_area().classes("w-full").style("height: calc(100vh - 7rem)")
    with chat_scroll:
        chat_container = ui.column().classes(
            "max-w-3xl w-full mx-auto px-5 py-6 gap-4"
        )

    # ══════════════════════════════════════════════════════════════════
    # DYNAMIC DIALOGS — question, confirmation, choice
    # ══════════════════════════════════════════════════════════════════

    def _show_question_dialog(handler: Any) -> None:
        """Show a multi-option question dialog."""
        data = handler.pending_data
        question = data.get("question", "Choose an option:")
        options = data.get("options", [])

        with ui.dialog() as dlg, ui.card().style(
            "background: #1a1a1a; min-width: 380px; border: 1px solid #333;"
        ):
            ui.label(question).classes("text-sm text-gray-300 mb-3")
            for opt in options:
                ui.button(
                    opt,
                    on_click=lambda _e, _opt=opt, _dlg=dlg: (
                        handler.resolve(_opt), _dlg.close()
                    ),
                ).props("flat dense no-caps").classes(
                    "w-full text-left text-gray-400"
                ).style("justify-content: flex-start;")
            ui.button("Cancel", on_click=lambda: (
                handler.resolve(options[0] if options else ""), dlg.close()
            )).props("flat dense no-caps").classes("text-gray-600 mt-2")
        dlg.open()

    def _show_confirmation_dialog(handler: Any) -> None:
        """Show a yes/no confirmation dialog."""
        data = handler.pending_data
        message = data.get("message", "Are you sure?")

        with ui.dialog() as dlg, ui.card().style(
            "background: #1a1a1a; min-width: 350px; border: 1px solid #333;"
        ):
            ui.label(message).classes("text-sm text-gray-300 mb-4")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("No", on_click=lambda: (
                    handler.resolve(False), dlg.close()
                )).props("flat dense no-caps").classes("text-gray-500")
                ui.button("Yes", on_click=lambda: (
                    handler.resolve(True), dlg.close()
                )).props("unelevated dense no-caps").classes(
                    "text-white"
                ).style("background: var(--accent);")
        dlg.open()

    def _show_choice_dialog(handler: Any) -> None:
        """Show a labelled-choices dialog."""
        data = handler.pending_data
        title = data.get("title", "Select an option:")
        choices = data.get("choices", [])  # list of (value, label)

        with ui.dialog() as dlg, ui.card().style(
            "background: #1a1a1a; min-width: 380px; border: 1px solid #333;"
        ):
            ui.label(title).classes("text-sm font-semibold text-gray-300 mb-3")
            for value, label in choices:
                ui.button(
                    label,
                    on_click=lambda _e, _val=value, _dlg=dlg: (
                        handler.resolve(_val), _dlg.close()
                    ),
                ).props("flat dense no-caps").classes(
                    "w-full text-left text-gray-400"
                ).style("justify-content: flex-start;")
            ui.button("Cancel", on_click=lambda: (
                handler.resolve(""), dlg.close()
            )).props("flat dense no-caps").classes("text-gray-600 mt-2")
        dlg.open()

    # ══════════════════════════════════════════════════════════════════
    # REFRESH TIMER — updates chat, status, approval dialog
    # ══════════════════════════════════════════════════════════════════
    rendered_count = 0
    _approval_open = False

    def _refresh() -> None:
        nonlocal rendered_count, _approval_open

        # -- Update header labels --
        ws = state.workspace or ""
        ws_name = ws.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1] if ws else "Quick Chat"
        _header_ws_label.set_text(ws_name)
        model_text = state.model or "no model"
        prov_text = f"({state.provider})" if state.provider else ""
        _header_model_label.set_text(f"{model_text} {prov_text}")
        # Also update footer model label
        try:
            _footer_model_label.set_text(model_text)
        except Exception:
            pass  # footer label may not exist yet during init

        # Toggle send/stop button icon based on generation state
        try:
            if state.is_thinking:
                send_stop_btn.props("color=red")
                send_stop_btn._props["icon"] = "stop"
                send_stop_btn.update()
            else:
                send_stop_btn.props(remove="color=red")
                send_stop_btn._props["icon"] = "arrow_upward"
                send_stop_btn.update()
        except Exception:
            pass  # button may not exist yet during init

        msg_snapshot = state.get_messages_snapshot()
        msg_count = len(msg_snapshot)

        # -- Re-render messages if changed --
        # "Stable History + Live Tail" strategy:
        #  - Finalized messages are rendered once and stay in the DOM
        #    (preserves expansion state for trace steps).
        #  - Only the last message (live tail) is rebuilt when streaming.
        #  - A full rebuild triggers only when message count changes
        #    (new message, conversation switch, or clear).

        needs_refresh = state.dirty or msg_count != rendered_count

        if msg_count != rendered_count:
            # Message count changed: full rebuild
            chat_container.clear()
            with chat_container:
                for msg in msg_snapshot:
                    render_message(msg)
            rendered_count = msg_count
            chat_scroll.scroll_to(percent=1.0)

        elif state.dirty and msg_count > 0:
            # Same message count but content updated (streaming, tool results)
            # Remove and re-render only the last message
            children = list(chat_container)
            if children:
                chat_container.remove(children[-1])
            last_msg = msg_snapshot[-1]
            with chat_container:
                render_message(last_msg)
            chat_scroll.scroll_to(percent=1.0)

        if needs_refresh:
            state.dirty = False

            # Refresh right panel
            session_panel_container.clear()
            with session_panel_container:
                render_session_panel(state)

            # Keep message cache up to date
            if state.current_conversation_id and msg_snapshot:
                state._message_cache[state.current_conversation_id] = msg_snapshot
                # Persist to disk
                state.save_all()

            # -- Auto-rename conversation after first exchange --
            if (
                not state._conversation_renamed
                and state.current_conversation_id
                and msg_count >= 2
            ):
                # Find the first user message to derive a title
                first_user = ""
                for m in state.messages:
                    if m.role == "user" and m.content:
                        first_user = m.content.strip()
                        break
                if first_user:
                    new_title = first_user[:55]
                    if len(first_user) > 55:
                        new_title += "…"
                    cid = state.current_conversation_id
                    # Update memory store
                    if state.memory_store:
                        state.memory_store.rename_summary(cid, new_title)
                    # Update local conversations
                    for lc in state.local_conversations:
                        if lc.get("conversation_id") == cid:
                            lc["title"] = new_title
                            break
                    state._conversation_renamed = True
                    _refresh_project_browser()
                    # Persist renamed title
                    state.save_conversations()

        # -- Drain notifications --
        while state.notifications:
            notif = state.notifications.pop(0)
            level = notif.level if hasattr(notif, "level") else "info"
            message = notif.message if hasattr(notif, "message") else str(notif)
            ui.notify(message, type=level, position="top-right")

        # -- Approval / question / confirmation / choice dialogs --
        if input_handler.has_pending and not _approval_open:
            ptype = input_handler.pending_type
            if ptype == "approval":
                approval_dialog.populate()  # type: ignore[attr-defined]
                approval_dialog.open()
                _approval_open = True
            elif ptype == "question":
                _show_question_dialog(input_handler)
                _approval_open = True
            elif ptype == "confirmation":
                _show_confirmation_dialog(input_handler)
                _approval_open = True
            elif ptype == "choice":
                _show_choice_dialog(input_handler)
                _approval_open = True
            elif ptype == "spawn" and spawn_dialog:
                spawn_dialog.populate()  # type: ignore[attr-defined]
                spawn_dialog.open()
                _approval_open = True
        elif not input_handler.has_pending and _approval_open:
            _approval_open = False

    ui.timer(0.5, _refresh)
    
    # Hidden upload component (placed here so it can't affect main layout flow)
    upload = ui.upload(
        on_upload=_handle_upload,
        auto_upload=True,
        max_files=5,
    ).props(
        'accept="image/*,application/pdf" flat dense'
    ).classes("hidden")
