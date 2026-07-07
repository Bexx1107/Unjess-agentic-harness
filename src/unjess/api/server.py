"""HTTP/WebSocket server — turns the CLI agent into a network-accessible platform.

Uses aiohttp (zero-config, stdlib-compatible) instead of FastAPI to keep
dependencies minimal. Can be swapped for FastAPI if needed.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from aiohttp import web
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


class AgentAPI:
    """HTTP + WebSocket server for the agent.

    Exposes REST endpoints for chat, approval, config, and status.
    WebSocket endpoint for real-time streaming.

    Args:
        host: Bind address (default "127.0.0.1").
        port: Port number (default 9757).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9757,
    ) -> None:
        self._host = host
        self._port = port
        self._app: Optional[Any] = None
        self._agent: Optional[Any] = None
        self._settings: Optional[Any] = None
        self._conversations: dict[str, dict[str, Any]] = {}
        self._pending_approvals: dict[str, asyncio.Future] = {}
        self._ws_clients: dict[str, list[Any]] = {}  # conv_id -> [ws connections]

    @property
    def is_available(self) -> bool:
        """Whether the server dependencies are installed."""
        return _HAS_AIOHTTP

    def configure(
        self,
        agent: Any = None,
        settings: Any = None,
    ) -> None:
        """Configure the API with agent and settings references.

        Args:
            agent: The Agent instance.
            settings: The Settings instance.
        """
        self._agent = agent
        self._settings = settings

    def create_app(self) -> Any:
        """Create the aiohttp application with all routes.

        Returns:
            aiohttp.web.Application instance.
        """
        if not _HAS_AIOHTTP:
            raise RuntimeError(
                "aiohttp is required for the API server. "
                "Install it with: pip install aiohttp"
            )

        app = web.Application()
        self._app = app

        # Register routes
        from unjess.api.routes import register_routes
        register_routes(app, self)

        return app

    async def start(self) -> None:
        """Start the server."""
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info("API server started on http://%s:%d", self._host, self._port)

    def run(self) -> None:
        """Run the server (blocking)."""
        app = self.create_app()
        web.run_app(app, host=self._host, port=self._port, print=None)

    # ----- Conversation management -----

    def create_conversation(self) -> str:
        """Create a new conversation.

        Returns:
            Conversation ID.
        """
        conv_id = uuid.uuid4().hex[:16]
        self._conversations[conv_id] = {
            "id": conv_id,
            "messages": [],
            "created_at": None,
        }
        return conv_id

    def get_conversation(self, conv_id: str) -> Optional[dict[str, Any]]:
        """Get a conversation by ID."""
        return self._conversations.get(conv_id)

    def list_conversations(self) -> list[dict[str, Any]]:
        """List all conversations."""
        return list(self._conversations.values())

    # ----- Approval handling -----

    async def request_approval(self, tool_call_id: str) -> bool:
        """Request approval for a tool call (blocks until approved/denied).

        Args:
            tool_call_id: The tool call to approve.

        Returns:
            True if approved.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_approvals[tool_call_id] = future

        # Notify WebSocket clients
        await self._broadcast_event("approval_required", {
            "tool_call_id": tool_call_id,
        })

        return await future

    def resolve_approval(self, tool_call_id: str, approved: bool) -> bool:
        """Resolve a pending approval.

        Args:
            tool_call_id: The tool call ID.
            approved: Whether to approve.

        Returns:
            True if the approval was pending and resolved.
        """
        future = self._pending_approvals.pop(tool_call_id, None)
        if future and not future.done():
            future.set_result(approved)
            return True
        return False

    # ----- WebSocket streaming -----

    def register_ws(self, conv_id: str, ws: Any) -> None:
        """Register a WebSocket connection for a conversation."""
        if conv_id not in self._ws_clients:
            self._ws_clients[conv_id] = []
        self._ws_clients[conv_id].append(ws)

    def unregister_ws(self, conv_id: str, ws: Any) -> None:
        """Unregister a WebSocket connection."""
        if conv_id in self._ws_clients:
            self._ws_clients[conv_id] = [
                w for w in self._ws_clients[conv_id] if w is not ws
            ]

    async def stream_event(self, conv_id: str, event_type: str, data: Any) -> None:
        """Send an event to all WebSocket clients for a conversation.

        Args:
            conv_id: Conversation ID.
            event_type: Event type (e.g. "token", "tool_call", "done").
            data: Event data.
        """
        message = json.dumps({"type": event_type, "data": data})
        clients = self._ws_clients.get(conv_id, [])

        for ws in clients:
            try:
                await ws.send_str(message)
            except Exception:
                pass  # client disconnected

    async def _broadcast_event(self, event_type: str, data: Any) -> None:
        """Broadcast an event to ALL WebSocket clients."""
        message = json.dumps({"type": event_type, "data": data})
        for clients in self._ws_clients.values():
            for ws in clients:
                try:
                    await ws.send_str(message)
                except Exception:
                    pass
