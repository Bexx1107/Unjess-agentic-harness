"""Tool registry — register, list, validate, execute tools."""

from dataclasses import dataclass, field
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """A registered tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    handler: Callable[..., str]  # function(**kwargs) -> str

    def to_schema(self) -> dict[str, Any]:
        """Return the JSON schema dict for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Central registry for all agent tools.

    Handles registration, schema generation, argument validation,
    and execution routing.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., str],
    ) -> None:
        """Register a new tool.

        Args:
            name: Unique tool name (e.g. 'read_file').
            description: Human-readable description for the LLM.
            parameters: JSON Schema dict describing the parameters.
            handler: Callable that takes **kwargs and returns a string result.
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Return JSON schemas for all registered tools (for LLM function calling)."""
        return [tool.to_schema() for tool in self._tools.values()]

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments.

        Includes argument normalization to handle LLM parameter name
        hallucination (e.g. a model sending 'pattern' instead of 'query').

        Returns the string result, or an error message.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'. Available tools: {', '.join(self._tools.keys())}"

        # Normalize arguments — fix hallucinated parameter names
        arguments = self._normalize_arguments(tool, arguments)

        try:
            return tool.handler(**arguments)
        except TypeError as exc:
            return f"Error: Invalid arguments for tool '{name}': {exc}"
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", name)
            return f"Error executing '{name}': {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # Argument normalization
    # ------------------------------------------------------------------

    # Common aliases that weaker models hallucinate for standard parameter names.
    # Maps hallucinated_name → list of canonical names it could mean.
    _PARAM_ALIASES: dict[str, list[str]] = {
        # Search-related
        "pattern": ["query", "search", "term"],
        "search_pattern": ["query", "search"],
        "search_term": ["query", "search"],
        "search_query": ["query", "search"],
        "text": ["query", "content", "code"],
        "term": ["query"],
        "keyword": ["query"],
        "expression": ["query"],
        "find": ["query"],
        # Path-related
        "file": ["path", "file_path", "target_file"],
        "filepath": ["path", "file_path", "target_file"],
        "file_name": ["path", "file_path", "target_file"],
        "filename": ["path", "file_path", "target_file"],
        "directory": ["path", "dir_path"],
        "dir": ["path", "dir_path"],
        "folder": ["path", "dir_path"],
        "target": ["path", "file_path", "target_file"],
        # Content-related
        "code": ["content", "new_content", "replacement"],
        "new_code": ["content", "new_content", "replacement"],
        "source": ["content", "code"],
        # Command-related
        "cmd": ["command"],
        "shell_command": ["command"],
        "run": ["command"],
        # Boolean flags
        "ignore_case": ["case_insensitive"],
        "case_sensitive": ["case_insensitive"],
        "is_regex": ["regex"],
        "use_regex": ["regex"],
        "recursive": ["recurse"],
        # Browser-related
        "link": ["url"],
        "href": ["url"],
        "uri": ["url"],
        "address": ["url"],
        "page_url": ["url"],
        "target_url": ["url"],
        "site": ["url"],
        "css_selector": ["selector"],
        "css": ["selector"],
        "element": ["selector"],
        "locator": ["selector"],
        "input_text": ["text"],
        "value": ["text"],
        "input": ["text"],
        "js": ["expression"],
        "script": ["expression"],
    }

    def _normalize_arguments(
        self, tool: ToolDefinition, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Remap hallucinated parameter names to match the tool's schema.

        If the LLM sends an argument name not in the tool's schema, try to
        map it to a valid parameter using the alias table. Only remaps if
        the target parameter exists in the schema and hasn't already been
        provided.

        Args:
            tool: The tool definition with its parameter schema.
            arguments: The raw arguments from the LLM.

        Returns:
            Corrected arguments dict (may be the same object if no fixes needed).
        """
        schema_props = tool.parameters.get("properties", {})
        if not schema_props:
            return arguments

        valid_names = set(schema_props.keys())

        # Check if any argument names are invalid
        invalid = {k for k in arguments if k not in valid_names}
        if not invalid:
            return arguments  # all good, fast path

        # Build corrected arguments
        corrected = dict(arguments)
        for bad_name in invalid:
            # Check alias table first
            candidates = self._PARAM_ALIASES.get(bad_name.lower(), [])
            matched = False
            for candidate in candidates:
                if candidate in valid_names and candidate not in corrected:
                    corrected[candidate] = corrected.pop(bad_name)
                    logger.info(
                        "Tool '%s': remapped arg '%s' → '%s'",
                        tool.name, bad_name, candidate,
                    )
                    matched = True
                    break

            if not matched:
                # Fallback: if there's exactly one required param missing,
                # assume the bad name was meant for it
                required = set(tool.parameters.get("required", []))
                missing_required = required - set(corrected.keys())
                if len(missing_required) == 1:
                    target = missing_required.pop()
                    corrected[target] = corrected.pop(bad_name)
                    logger.info(
                        "Tool '%s': remapped arg '%s' → '%s' (missing required)",
                        tool.name, bad_name, target,
                    )

        return corrected

    @property
    def tool_names(self) -> list[str]:
        """List of all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
