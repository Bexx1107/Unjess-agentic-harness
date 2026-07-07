"""API routes — REST + WebSocket endpoints.

Endpoints:
- POST /api/chat            — send a message
- GET  /api/conversations    — list conversations
- GET  /api/conversations/:id — get conversation history
- POST /api/approve/:id     — approve/deny a tool call
- GET  /api/config           — get current config
- PUT  /api/config           — update config
- GET  /api/status           — server/agent status
- WS   /api/stream/:id      — real-time streaming
"""

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import aiohttp
    from aiohttp import web
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False
    web = None  # type: ignore[assignment]


def register_routes(app: Any, api: Any) -> None:
    """Register all API routes on the aiohttp app.

    Args:
        app: aiohttp.web.Application.
        api: AgentAPI instance.
    """
    if not _HAS_AIOHTTP:
        return

    # REST endpoints
    app.router.add_post("/api/chat", _make_chat_handler(api))
    app.router.add_get("/api/conversations", _make_list_conversations_handler(api))
    app.router.add_get("/api/conversations/{conv_id}", _make_get_conversation_handler(api))
    app.router.add_post("/api/approve/{tool_call_id}", _make_approve_handler(api))
    app.router.add_get("/api/config", _make_get_config_handler(api))
    app.router.add_put("/api/config", _make_update_config_handler(api))
    app.router.add_get("/api/status", _make_status_handler(api))

    # WebSocket
    app.router.add_get("/api/stream/{conv_id}", _make_ws_handler(api))

    # Health check
    app.router.add_get("/api/health", _make_health_handler())


# ---------------------------------------------------------------------------
# Route handlers (factory functions to capture `api` reference)
# ---------------------------------------------------------------------------

def _make_health_handler() -> Any:
    """Health check endpoint."""
    async def handler(request: Any) -> Any:
        return web.json_response({"status": "ok", "timestamp": time.time()})
    return handler


def _make_chat_handler(api: Any) -> Any:
    """POST /api/chat — send a message."""
    async def handler(request: Any) -> Any:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        message = body.get("message", "")
        conv_id = body.get("conversation_id", "")

        if not message:
            return web.json_response({"error": "message is required"}, status=400)

        # Create conversation if needed
        if not conv_id:
            conv_id = api.create_conversation()

        conv = api.get_conversation(conv_id)
        if not conv:
            return web.json_response({"error": "Conversation not found"}, status=404)

        # Add user message
        conv["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": time.time(),
        })

        # Run agent (non-blocking in production, sync for now)
        response_text = ""
        if api._agent:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, api._agent.run, message)
                response_text = "Message processed"
            except Exception as exc:
                response_text = f"Error: {exc}"
        else:
            response_text = "Agent not configured"

        return web.json_response({
            "conversation_id": conv_id,
            "response": response_text,
        })
    return handler


def _make_list_conversations_handler(api: Any) -> Any:
    """GET /api/conversations — list all conversations."""
    async def handler(request: Any) -> Any:
        convs = api.list_conversations()
        return web.json_response({
            "conversations": [
                {"id": c["id"], "message_count": len(c["messages"])}
                for c in convs
            ]
        })
    return handler


def _make_get_conversation_handler(api: Any) -> Any:
    """GET /api/conversations/:id — get conversation history."""
    async def handler(request: Any) -> Any:
        conv_id = request.match_info["conv_id"]
        conv = api.get_conversation(conv_id)
        if not conv:
            return web.json_response({"error": "Not found"}, status=404)
        return web.json_response(conv)
    return handler


def _make_approve_handler(api: Any) -> Any:
    """POST /api/approve/:id — approve or deny a tool call."""
    async def handler(request: Any) -> Any:
        tool_call_id = request.match_info["tool_call_id"]
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        approved = body.get("approved", False)
        resolved = api.resolve_approval(tool_call_id, approved)

        if not resolved:
            return web.json_response({"error": "No pending approval"}, status=404)

        return web.json_response({"status": "resolved", "approved": approved})
    return handler


def _make_get_config_handler(api: Any) -> Any:
    """GET /api/config — get current configuration."""
    async def handler(request: Any) -> Any:
        if api._settings:
            config = {
                "model": api._settings.model,
                "provider": getattr(api._settings, "provider", ""),
                "verbosity": api._settings.verbosity,
            }
        else:
            config = {}
        return web.json_response(config)
    return handler


def _make_update_config_handler(api: Any) -> Any:
    """PUT /api/config — update configuration."""
    async def handler(request: Any) -> Any:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        if api._settings:
            if "model" in body:
                api._settings.model = body["model"]
            if "verbose" in body:
                api._settings.verbosity = body["verbose"]

        return web.json_response({"status": "updated"})
    return handler


def _make_status_handler(api: Any) -> Any:
    """GET /api/status — server and agent status."""
    async def handler(request: Any) -> Any:
        status = {
            "server": "running",
            "conversations": len(api.list_conversations()),
            "pending_approvals": len(api._pending_approvals),
            "ws_clients": sum(len(v) for v in api._ws_clients.values()),
        }
        if api._settings:
            status["model"] = api._settings.model
        return web.json_response(status)
    return handler


def _make_ws_handler(api: Any) -> Any:
    """WS /api/stream/:id — real-time streaming WebSocket."""
    async def handler(request: Any) -> Any:
        conv_id = request.match_info["conv_id"]
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        api.register_ws(conv_id, ws)
        logger.info("WebSocket client connected for conversation %s", conv_id[:8])

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Client can send messages via WebSocket too
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "message":
                            # Process message
                            await api.stream_event(conv_id, "ack", {"received": True})
                    except json.JSONDecodeError:
                        pass
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        finally:
            api.unregister_ws(conv_id, ws)
            logger.info("WebSocket client disconnected from %s", conv_id[:8])

        return ws
    return handler
