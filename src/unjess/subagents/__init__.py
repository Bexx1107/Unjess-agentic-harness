"""Subagent system package — spawn, manage, and communicate with child agents."""

from unjess.subagents.manager import SubagentManager
from unjess.subagents.types import AgentType, BUILTIN_TYPES

__all__ = ["SubagentManager", "AgentType", "BUILTIN_TYPES"]
