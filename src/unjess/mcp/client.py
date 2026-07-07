"""MCP JSON-RPC protocol client — handles message framing and request/response."""

import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_request_id_counter = itertools.count(1)


def _next_id() -> int:
    """Generate a unique request ID."""
    return next(_request_id_counter)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MCPToolSchema:
    """Schema for a single MCP tool."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""

    def to_agent_schema(self) -> dict[str, Any]:
        """Convert to the agent's tool schema format."""
        return {
            "name": f"mcp_{self.server_name}_{self.name}" if self.server_name else self.name,
            "description": f"[MCP:{self.server_name}] {self.description}",
            "parameters": self.input_schema or {"type": "object", "properties": {}},
        }


@dataclass
class MCPResource:
    """An MCP resource."""

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""


@dataclass
class MCPResponse:
    """Response from an MCP server."""

    id: int | str | None = None
    result: Any = None
    error: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        """Whether this response is an error."""
        return self.error is not None

    @property
    def error_message(self) -> str:
        """Human-readable error message."""
        if self.error:
            return self.error.get("message", str(self.error))
        return ""


# ---------------------------------------------------------------------------
# JSON-RPC message builders
# ---------------------------------------------------------------------------

def build_request(method: str, params: dict[str, Any] | None = None) -> tuple[int, str]:
    """Build a JSON-RPC 2.0 request.

    Returns (request_id, json_string).
    """
    req_id = _next_id()
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return req_id, json.dumps(msg)


def build_notification(method: str, params: dict[str, Any] | None = None) -> str:
    """Build a JSON-RPC 2.0 notification (no ID, no response expected)."""
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def parse_response(raw: str) -> MCPResponse:
    """Parse a JSON-RPC 2.0 response."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return MCPResponse(error={"code": -32700, "message": f"Parse error: {exc}"})

    return MCPResponse(
        id=data.get("id"),
        result=data.get("result"),
        error=data.get("error"),
    )


# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------

class MCPClient:
    """High-level MCP client that wraps a transport.

    Provides methods for the MCP protocol operations:
    initialize, list tools, call tools, list resources, read resources.

    Args:
        transport: A transport object with send(str) -> str interface.
        server_name: Name identifier for this server.
    """

    def __init__(self, transport: Any, server_name: str = "") -> None:
        self._transport = transport
        self._server_name = server_name
        self._initialized = False
        self._server_capabilities: dict[str, Any] = {}

    @property
    def server_name(self) -> str:
        """Name of the connected server."""
        return self._server_name

    @property
    def is_initialized(self) -> bool:
        """Whether the MCP handshake has completed."""
        return self._initialized

    # ----- Protocol methods -----

    def initialize(self) -> MCPResponse:
        """Send the MCP initialize handshake."""
        resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "unjess",
                "version": "0.1.0",
            },
        })

        if not resp.is_error and resp.result:
            self._server_capabilities = resp.result.get("capabilities", {})
            self._initialized = True

            # Send initialized notification
            self._notify("notifications/initialized")

        return resp

    def list_tools(self) -> list[MCPToolSchema]:
        """List available tools from the server."""
        resp = self._request("tools/list")

        if resp.is_error or not resp.result:
            logger.warning("Failed to list tools: %s", resp.error_message)
            return []

        tools: list[MCPToolSchema] = []
        for tool_data in resp.result.get("tools", []):
            tools.append(MCPToolSchema(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self._server_name,
            ))
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPResponse:
        """Call a tool on the server.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            MCPResponse with the tool result.
        """
        return self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

    def list_resources(self) -> list[MCPResource]:
        """List available resources from the server."""
        resp = self._request("resources/list")

        if resp.is_error or not resp.result:
            return []

        resources: list[MCPResource] = []
        for res_data in resp.result.get("resources", []):
            resources.append(MCPResource(
                uri=res_data.get("uri", ""),
                name=res_data.get("name", ""),
                description=res_data.get("description", ""),
                mime_type=res_data.get("mimeType", ""),
            ))
        return resources

    def read_resource(self, uri: str) -> MCPResponse:
        """Read a resource by URI."""
        return self._request("resources/read", {"uri": uri})

    def close(self) -> None:
        """Close the transport connection."""
        if hasattr(self._transport, "close"):
            self._transport.close()
        self._initialized = False

    # ----- Internal -----

    def _request(self, method: str, params: dict[str, Any] | None = None) -> MCPResponse:
        """Send a request and wait for the response."""
        req_id, message = build_request(method, params)

        try:
            raw_response = self._transport.send(message)
            return parse_response(raw_response)
        except Exception as exc:
            logger.error("MCP request failed (%s): %s", method, exc)
            return MCPResponse(error={"code": -1, "message": str(exc)})

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification (no response expected)."""
        message = build_notification(method, params)
        try:
            self._transport.send_notification(message)
        except Exception as exc:
            logger.debug("MCP notification failed (%s): %s", method, exc)
