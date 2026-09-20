"""MCP server manager — discovery, lifecycle, and tool registry integration."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from unjess.mcp.client import MCPClient, MCPToolSchema
from unjess.mcp.transport import StdioTransport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config data
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    eager_tools: list[str] = field(default_factory=list)  # tools to load immediately
    enabled: bool = False


# ---------------------------------------------------------------------------
# Server Manager
# ---------------------------------------------------------------------------

class ServerManager:
    """Manages MCP server connections, tool discovery, and lifecycle.

    Discovers servers from the config file, starts them on demand,
    and exposes their tools to the agent's tool registry.

    Args:
        config_path: Path to the MCP config file (YAML).
        config_data: Pre-loaded config dict (e.g. from main config.yaml).
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config_data: Optional[dict] = None,
        workspace_dir: Optional[Path] = None,
    ) -> None:
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._all_tools: dict[str, MCPToolSchema] = {}  # keyed by prefixed name
        self._workspace_dir = workspace_dir

        if config_data:
            self._load_from_dict(config_data)
        elif config_path and config_path.exists():
            self._load_config(config_path)

    def _resolve_path(self, path_str: str) -> str:
        """Resolve placeholders, relative paths, and incorrect paths against workspace."""
        if not path_str:
            return path_str

        # 1. Determine workspace root
        ws = self._workspace_dir
        if not ws:
            try:
                from unjess.workspace import detect_project_root
                ws = detect_project_root(Path.cwd())
            except Exception:
                ws = Path.cwd()

        ws_str = str(ws.resolve())

        # 2. Replace placeholders
        resolved = path_str.replace("${workspaceRoot}", ws_str).replace("${workspace}", ws_str)

        # 3. Handle relative paths
        norm = resolved.replace("\\", "/")
        if norm.startswith("./") or norm.startswith("../"):
            try:
                resolved = str((ws / resolved).resolve())
            except Exception:
                pass

        # 4. Self-healing for absolute paths from other machines
        try:
            p = Path(resolved)
            if not p.exists() and p.is_absolute():
                parts = p.parts
                if "mcp-servers" in parts:
                    idx = parts.index("mcp-servers")
                    subpath = Path(*parts[idx:])
                    candidate = ws / subpath
                    if candidate.exists():
                        resolved = str(candidate.resolve())
        except Exception:
            pass

        return resolved

    @property
    def server_names(self) -> list[str]:
        """Names of configured servers."""
        return list(self._configs.keys())

    @property
    def connected_servers(self) -> list[str]:
        """Names of currently connected servers."""
        return [name for name, client in self._clients.items() if client.is_initialized]

    @property
    def available_tools(self) -> list[MCPToolSchema]:
        """All discovered tools across all servers."""
        return list(self._all_tools.values())

    # ----- Config loading -----

    def _load_config(self, path: Path) -> None:
        """Load server configurations from a file (JSON or YAML)."""
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read MCP config: %s", exc)
            return

        # Parse based on extension
        try:
            if path.suffix == ".json":
                import json
                data = json.loads(raw)
            else:
                data = yaml.safe_load(raw)
        except Exception as exc:
            logger.warning("Failed to parse MCP config %s: %s", path, exc)
            return

        if isinstance(data, dict):
            self._load_from_dict(data)

    def _load_from_dict(self, data: dict) -> None:
        """Load server configurations from a dict.

        Supports standard MCP JSON format (``mcpServers``) and
        our YAML format (``mcp_servers``).
        """
        # Standard JSON: mcpServers (camelCase)
        # Our YAML: mcp_servers (snake_case)
        # Fallback: servers
        servers = (
            data.get("mcpServers")
            or data.get("mcp_servers")
            or data.get("servers")
            or {}
        )
        if isinstance(servers, dict):
            for name, cfg in servers.items():
                if isinstance(cfg, dict):
                    self._configs[name] = MCPServerConfig(
                        name=name,
                        command=cfg.get("command", ""),
                        args=cfg.get("args", []),
                        cwd=cfg.get("cwd"),
                        env=cfg.get("env", {}),
                        eager_tools=cfg.get("eager_tools", cfg.get("eagerTools", [])),
                        enabled=cfg.get("enabled", False),
                    )

    # ----- Server lifecycle -----

    def connect(self, server_name: str) -> bool:
        """Start and initialize an MCP server.

        Args:
            server_name: Name of the server from config.

        Returns:
            True if successfully connected.
        """
        config = self._configs.get(server_name)
        if not config:
            logger.error("Unknown MCP server: %s", server_name)
            return False

        if not config.enabled:
            logger.info("MCP server '%s' is disabled", server_name)
            return False

        if server_name in self._clients and self._clients[server_name].is_initialized:
            return True  # already connected

        # Resolve path variables and self-heal incorrect absolute paths
        resolved_cmd = self._resolve_path(config.command)
        resolved_args = [self._resolve_path(arg) for arg in config.args]
        resolved_cwd = self._resolve_path(config.cwd) if config.cwd else None

        # Create transport and client
        transport = StdioTransport(
            command=resolved_cmd,
            args=resolved_args,
            cwd=resolved_cwd,
            env=config.env or None,
        )

        try:
            transport.start()
        except RuntimeError as exc:
            logger.error("Failed to start MCP server '%s': %s", server_name, exc)
            return False

        client = MCPClient(transport, server_name=server_name)

        # Initialize handshake
        resp = client.initialize()
        if resp.is_error:
            logger.error("MCP init failed for '%s': %s", server_name, resp.error_message)
            transport.close()
            return False

        self._clients[server_name] = client
        logger.info("Connected to MCP server: %s", server_name)

        # Discover tools
        self._discover_tools(server_name)

        return True

    def connect_all(self) -> int:
        """Connect to all enabled servers.

        Returns:
            Number of servers successfully connected.
        """
        connected = 0
        for name, config in self._configs.items():
            if config.enabled and self.connect(name):
                connected += 1
        return connected

    def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server."""
        client = self._clients.pop(server_name, None)
        if client:
            client.close()
            # Remove tools for this server
            to_remove = [k for k, v in self._all_tools.items() if v.server_name == server_name]
            for key in to_remove:
                del self._all_tools[key]
            logger.info("Disconnected from MCP server: %s", server_name)

    def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._clients.keys()):
            self.disconnect(name)

    # ----- Tool discovery -----

    def _discover_tools(self, server_name: str) -> None:
        """Discover and cache tools from a server."""
        client = self._clients.get(server_name)
        if not client:
            return

        tools = client.list_tools()
        for tool in tools:
            prefixed_name = f"mcp_{server_name}_{tool.name}"
            self._all_tools[prefixed_name] = tool
            logger.debug("Discovered MCP tool: %s/%s", server_name, tool.name)

        logger.info("Discovered %d tools from '%s'", len(tools), server_name)

    def refresh_tools(self, tool_registry: Any = None) -> dict[str, int]:
        """Re-discover tools from all connected servers.

        Clears the cached tool list and re-fetches from each connected
        server.  Optionally re-registers tools on the given registry.

        Args:
            tool_registry: If provided, re-register tools on this registry.

        Returns:
            Dict mapping server name to number of tools discovered.
        """
        # Clear cached tools
        self._all_tools.clear()

        results: dict[str, int] = {}
        for name, client in self._clients.items():
            if client.is_initialized:
                before = len(self._all_tools)
                self._discover_tools(name)
                results[name] = len(self._all_tools) - before

        # Re-register on tool registry if provided
        if tool_registry is not None:
            self.register_tools(tool_registry)

        return results

    # ----- Tool execution -----

    def call_tool(self, prefixed_name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool by its prefixed name.

        Args:
            prefixed_name: Tool name in format ``mcp_{server}_{tool}``.
            arguments: Tool arguments.

        Returns:
            Result text from the server.
        """
        tool = self._all_tools.get(prefixed_name)
        if not tool:
            return f"Error: Unknown MCP tool '{prefixed_name}'"

        client = self._clients.get(tool.server_name)
        if not client or not client.is_initialized:
            return f"Error: MCP server '{tool.server_name}' is not connected"

        resp = client.call_tool(tool.name, arguments)

        if resp.is_error:
            return f"Error: MCP tool call failed: {resp.error_message}"

        # Extract text content from MCP response
        if isinstance(resp.result, dict):
            content = resp.result.get("content", [])
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        text_parts.append(item.get("text", str(item)))
                    else:
                        text_parts.append(str(item))
                return "\n".join(text_parts)
            return str(content)

        return str(resp.result) if resp.result is not None else ""

    # ----- Agent integration -----

    def register_tools(self, tool_registry: Any) -> int:
        """Register all discovered MCP tools with the agent's tool registry.

        Args:
            tool_registry: The ToolRegistry instance.

        Returns:
            Number of tools registered.
        """
        count = 0
        for prefixed_name, tool in self._all_tools.items():
            schema = tool.to_agent_schema()

            # Create a closure for the tool handler
            def make_handler(name: str):  # noqa: E306
                def handler(**kwargs: Any) -> str:
                    return self.call_tool(name, kwargs)
                return handler

            tool_registry.register(
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"],
                handler=make_handler(prefixed_name),
            )
            count += 1

        return count

    # ----- Context info -----

    def context_summary(self) -> str:
        """Build MCP server summary for the system prompt.

        Includes per-server instruction files from
        ``~/.unjess/mcp_instructions/<server_name>.md`` if they exist.
        """
        if not self._clients:
            return ""

        lines = ["MCP Servers:"]
        for name, client in self._clients.items():
            tools = [t for t in self._all_tools.values() if t.server_name == name]
            tool_names = ", ".join(t.name for t in tools[:10])
            status = "connected" if client.is_initialized else "disconnected"
            lines.append(f"  - {name} ({status}): {tool_names}")

        # Load per-server instruction files
        instructions_dir = Path.home() / ".unjess" / "mcp_instructions"
        for name in self._clients:
            instr_file = instructions_dir / f"{name}.md"
            if instr_file.exists():
                try:
                    content = instr_file.read_text(encoding="utf-8").strip()
                    if content:
                        lines.append(f"\n--- Instructions for {name} ---")
                        lines.append(content)
                except Exception as exc:
                    logger.debug("Failed to load MCP instructions for %s: %s", name, exc)

        return "\n".join(lines)
