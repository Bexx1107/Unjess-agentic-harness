"""Model router — intelligent routing of requests to cheap vs expensive models.

Implements:
- Complexity assessment (heuristic-based)
- Task classification (edit, explain, plan, debug)
- Architect mode (plan with expensive, execute with cheap)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from unjess.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    """Result of model routing — which model to use and why."""

    model: str
    reason: str
    task_type: str = "general"  # edit, explain, plan, debug, general
    complexity: str = "medium"  # simple, medium, complex

    def __str__(self) -> str:
        return f"{self.model} ({self.reason})"


@dataclass
class ModelTier:
    """A model with its tier classification."""

    name: str
    tier: str  # "cheap", "standard", "expensive", "reasoning"
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

    @property
    def is_cheap(self) -> bool:
        """Whether this is a budget model."""
        return self.tier == "cheap"


# ---------------------------------------------------------------------------
# Known model tiers
# ---------------------------------------------------------------------------

_MODEL_TIERS: dict[str, ModelTier] = {
    # Cheap / fast
    "gpt-4o-mini": ModelTier("gpt-4o-mini", "cheap", 0.15, 0.60),
    "gemini-3.1-flash": ModelTier("gemini-3.1-flash", "cheap", 0.15, 0.60),
    # Standard
    "gpt-4o": ModelTier("gpt-4o", "standard", 2.50, 10.00),
    "claude-sonnet-4-20250514": ModelTier("claude-sonnet-4-20250514", "standard", 3.00, 15.00),
    "gemini-3.1-pro": ModelTier("gemini-3.1-pro", "standard", 1.25, 10.00),
    # Expensive / reasoning
    "o3": ModelTier("o3", "reasoning", 2.00, 8.00),
    "o4-mini": ModelTier("o4-mini", "reasoning", 1.10, 4.40),
    "claude-opus-4-20250514": ModelTier("claude-opus-4-20250514", "expensive", 15.00, 75.00),
}

# Task classification keywords
_SIMPLE_KEYWORDS = {
    "rename", "fix typo", "add comment", "remove", "delete",
    "update import", "change name", "add type hint", "format",
}
_PLAN_KEYWORDS = {
    "refactor", "redesign", "architect", "plan", "restructure",
    "migrate", "rewrite", "overhaul", "design", "strategy",
    "build me", "create a", "make a", "implement a", "set up",
    "build a", "create me", "make me", "write a", "develop",
}
_DEBUG_KEYWORDS = {
    "debug", "fix", "error", "bug", "broken", "crash", "exception",
    "traceback", "failing", "doesn't work", "issue",
}
_EXPLAIN_KEYWORDS = {
    "explain", "how does", "what does", "why does", "describe",
    "understand", "walk me through", "what is",
}


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """Routes user requests to the appropriate model based on complexity.

    Args:
        settings: Runtime settings with model and provider config.
        cheap_model: Override for the cheap model name.
        expensive_model: Override for the expensive model name.
    """

    def __init__(
        self,
        settings: Settings,
        cheap_model: Optional[str] = None,
        expensive_model: Optional[str] = None,
    ) -> None:
        self._settings = settings
        self._cheap_model = cheap_model or self._detect_cheap_model()
        self._expensive_model = expensive_model or settings.model
        self._routing_enabled = self._cheap_model != self._expensive_model

    @property
    def routing_enabled(self) -> bool:
        """Whether dual-model routing is active."""
        return self._routing_enabled

    def route(self, user_message: str) -> RouteDecision:
        """Decide which model to use for a given message.

        Args:
            user_message: The user's input.

        Returns:
            RouteDecision with model name, reason, task type, complexity.
        """
        if not self._routing_enabled:
            return RouteDecision(
                model=self._settings.model,
                reason="single model mode",
                task_type="general",
            )

        task_type = self._classify_task(user_message)
        complexity = self._assess_complexity(user_message)

        # Routing rules
        if task_type == "explain" and complexity == "simple":
            return RouteDecision(
                model=self._cheap_model,
                reason="simple explanation",
                task_type=task_type,
                complexity=complexity,
            )

        if task_type == "edit" and complexity == "simple":
            return RouteDecision(
                model=self._cheap_model,
                reason="simple edit",
                task_type=task_type,
                complexity=complexity,
            )

        if task_type == "plan" or complexity == "complex":
            return RouteDecision(
                model=self._expensive_model,
                reason="complex task requires powerful model",
                task_type=task_type,
                complexity=complexity,
            )

        # Default to current model
        return RouteDecision(
            model=self._settings.model,
            reason="standard routing",
            task_type=task_type,
            complexity=complexity,
        )

    # ----- Classification -----

    def _classify_task(self, message: str) -> str:
        """Classify the task type from the user's message."""
        msg_lower = message.lower()

        if any(kw in msg_lower for kw in _PLAN_KEYWORDS):
            return "plan"
        if any(kw in msg_lower for kw in _DEBUG_KEYWORDS):
            return "debug"
        if any(kw in msg_lower for kw in _EXPLAIN_KEYWORDS):
            return "explain"
        if any(kw in msg_lower for kw in _SIMPLE_KEYWORDS):
            return "edit"

        return "general"

    def _assess_complexity(self, message: str) -> str:
        """Assess the complexity of a request.

        Heuristics:
        - Short messages with simple keywords → simple
        - Long messages with planning keywords → complex
        - Multiple files mentioned → complex
        - Everything else → medium
        """
        msg_lower = message.lower()
        word_count = len(message.split())

        # Simple indicators
        if word_count < 15 and any(kw in msg_lower for kw in _SIMPLE_KEYWORDS):
            return "simple"

        # Complex indicators
        complex_score = 0
        if word_count > 100:
            complex_score += 1
        if any(kw in msg_lower for kw in _PLAN_KEYWORDS):
            complex_score += 2

        # Multiple file references
        file_refs = re.findall(r'(?:[\w/\\]+/)?[\w-]+\.(?:py|js|ts|jsx|tsx|css|html|json|yaml|yml|md|txt|go|rs|java|c|cpp|h|rb|sh)\b', message)
        if len(file_refs) > 3:
            complex_score += 1

        # Code blocks
        if "```" in message:
            complex_score += 1

        if complex_score >= 3:
            return "complex"
        elif complex_score >= 1:
            return "medium"

        return "simple" if word_count < 20 else "medium"

    # ----- Helpers -----

    def _detect_cheap_model(self) -> str:
        """Auto-detect a cheap model based on the current provider."""
        model = self._settings.model.lower()

        if "gpt" in model or "o3" in model or "o4" in model:
            return "gpt-4o-mini"
        elif "claude" in model:
            return "claude-3-5-haiku-20241022"
        elif "gemini" in model:
            return "gemini-3.1-flash"

        return self._settings.model  # fallback to same model


# ---------------------------------------------------------------------------
# Architect Mode
# ---------------------------------------------------------------------------

@dataclass
class ArchitectPlan:
    """A plan produced by the architect model."""

    description: str
    files_to_edit: list[str]
    approach: str
    raw_plan: str


class ArchitectMode:
    """Dual-model pipeline: plan with expensive model, execute with cheap.

    The expensive model produces a high-level plan, then the cheap
    model executes the mechanical edits.

    Args:
        settings: Runtime settings.
        model_router: Model router for model selection.
    """

    def __init__(self, settings: Settings, model_router: ModelRouter) -> None:
        self._settings = settings
        self._router = model_router
        self.enabled = True  # plan-first by default

    def should_use(self, task_type: str, complexity: str) -> bool:
        """Whether architect mode should be used for this task."""
        if not self.enabled:
            return False
        # Always plan for complex tasks of relevant types
        if complexity == "complex" and task_type in ("plan", "edit", "debug"):
            return True
        # Also plan for medium tasks that are explicitly plan-type
        # (e.g. "build me an app" is short but clearly needs a plan)
        if task_type == "plan" and complexity in ("medium", "complex"):
            return True
        return False

    def build_planning_prompt(self, user_request: str, repo_context: str = "") -> str:
        """Build a prompt for the architect model.

        Args:
            user_request: What the user wants.
            repo_context: Repo map or relevant files.

        Returns:
            Planning prompt for the expensive model.
        """
        return (
            "You are an architect. Plan the changes needed, but do NOT write code.\n\n"
            "List:\n"
            "1. Which files to modify/create/delete\n"
            "2. What changes to make in each file\n"
            "3. The overall approach and reasoning\n\n"
            f"Project context:\n{repo_context}\n\n"
            f"User request:\n{user_request}"
        )

    def build_execution_prompt(self, plan: str, user_request: str) -> str:
        """Build a prompt for the editor model to execute the plan.

        Args:
            plan: The architect's plan.
            user_request: Original user request.

        Returns:
            Execution prompt for the cheap model.
        """
        return (
            f"Execute the following plan. Make the exact edits described.\n\n"
            f"Plan:\n{plan}\n\n"
            f"Original request:\n{user_request}"
        )
