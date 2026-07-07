"""Tests for unjess.model_router — routing, classification, complexity, architect mode."""

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from unjess.config import Settings
from unjess.model_router import (
    ArchitectMode,
    ArchitectPlan,
    ModelRouter,
    ModelTier,
    RouteDecision,
    _DEBUG_KEYWORDS,
    _EXPLAIN_KEYWORDS,
    _MODEL_TIERS,
    _PLAN_KEYWORDS,
    _SIMPLE_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(model: str = "gpt-4o", **kwargs: Any) -> Settings:
    """Build a Settings instance with the given model."""
    return Settings(model=model, **kwargs)


# ===========================================================================
# RouteDecision
# ===========================================================================

class TestRouteDecision:
    """RouteDecision dataclass and __str__."""

    def test_defaults(self) -> None:
        rd = RouteDecision(model="gpt-4o", reason="test")
        assert rd.task_type == "general"
        assert rd.complexity == "medium"

    def test_str(self) -> None:
        rd = RouteDecision(model="gpt-4o", reason="simple edit")
        assert str(rd) == "gpt-4o (simple edit)"

    def test_custom_fields(self) -> None:
        rd = RouteDecision(model="o3", reason="complex", task_type="plan", complexity="complex")
        assert rd.model == "o3"
        assert rd.task_type == "plan"
        assert rd.complexity == "complex"


# ===========================================================================
# ModelTier
# ===========================================================================

class TestModelTier:
    """ModelTier dataclass with is_cheap property."""

    def test_cheap_tier(self) -> None:
        mt = ModelTier(name="gpt-4o-mini", tier="cheap", cost_per_1m_input=0.15, cost_per_1m_output=0.60)
        assert mt.is_cheap is True

    def test_standard_tier_not_cheap(self) -> None:
        mt = ModelTier(name="gpt-4o", tier="standard")
        assert mt.is_cheap is False

    def test_expensive_tier_not_cheap(self) -> None:
        mt = ModelTier(name="claude-opus-4-20250514", tier="expensive")
        assert mt.is_cheap is False

    def test_reasoning_tier_not_cheap(self) -> None:
        mt = ModelTier(name="o3", tier="reasoning")
        assert mt.is_cheap is False

    def test_default_costs(self) -> None:
        mt = ModelTier(name="test", tier="cheap")
        assert mt.cost_per_1m_input == 0.0
        assert mt.cost_per_1m_output == 0.0


# ===========================================================================
# ArchitectPlan
# ===========================================================================

class TestArchitectPlan:
    """ArchitectPlan dataclass."""

    def test_fields(self) -> None:
        plan = ArchitectPlan(
            description="Add logging",
            files_to_edit=["app.py", "utils.py"],
            approach="Wrap functions with decorators",
            raw_plan="Full plan text...",
        )
        assert plan.description == "Add logging"
        assert len(plan.files_to_edit) == 2
        assert plan.approach == "Wrap functions with decorators"
        assert plan.raw_plan == "Full plan text..."


# ===========================================================================
# ModelRouter — construction
# ===========================================================================

class TestModelRouterConstruction:
    """ModelRouter initialization and auto-detection."""

    def test_explicit_cheap_and_expensive(self) -> None:
        router = ModelRouter(_settings("gpt-4o"), cheap_model="gpt-4o-mini", expensive_model="o3")
        assert router._cheap_model == "gpt-4o-mini"
        assert router._expensive_model == "o3"
        assert router.routing_enabled is True

    def test_same_model_disables_routing(self) -> None:
        router = ModelRouter(_settings("gpt-4o"), cheap_model="gpt-4o", expensive_model="gpt-4o")
        assert router.routing_enabled is False

    def test_auto_detect_openai(self) -> None:
        router = ModelRouter(_settings("gpt-4o"))
        assert router._cheap_model == "gpt-4o-mini"

    def test_auto_detect_anthropic(self) -> None:
        router = ModelRouter(_settings("claude-sonnet-4-20250514"))
        assert router._cheap_model == "claude-3-5-haiku-20241022"

    def test_auto_detect_gemini(self) -> None:
        router = ModelRouter(_settings("gemini-2.5-pro"))
        assert router._cheap_model == "gemini-2.5-flash"

    def test_auto_detect_o3(self) -> None:
        router = ModelRouter(_settings("o3"))
        assert router._cheap_model == "gpt-4o-mini"

    def test_auto_detect_o4(self) -> None:
        router = ModelRouter(_settings("o4-mini"))
        assert router._cheap_model == "gpt-4o-mini"

    def test_auto_detect_unknown_fallback(self) -> None:
        s = _settings("totally-unknown-model")
        router = ModelRouter(s)
        assert router._cheap_model == s.model
        # Same model → routing disabled
        assert router.routing_enabled is False

    def test_expensive_defaults_to_settings_model(self) -> None:
        router = ModelRouter(_settings("gpt-4o"))
        assert router._expensive_model == "gpt-4o"


# ===========================================================================
# ModelRouter — route (single-model mode)
# ===========================================================================

class TestModelRouterSingleModel:
    """Routing when dual-model is disabled."""

    def test_single_model_always_returns_settings_model(self) -> None:
        router = ModelRouter(_settings("my-local-model"), cheap_model="my-local-model", expensive_model="my-local-model")
        result = router.route("refactor the entire codebase")
        assert result.model == "my-local-model"
        assert result.reason == "single model mode"
        assert result.task_type == "general"


# ===========================================================================
# ModelRouter — route (dual-model mode)
# ===========================================================================

class TestModelRouterDualRoute:
    """Routing decisions in dual-model mode."""

    def _router(self) -> ModelRouter:
        return ModelRouter(_settings("gpt-4o"), cheap_model="gpt-4o-mini", expensive_model="o3")

    def test_simple_explain_routes_cheap(self) -> None:
        router = self._router()
        result = router.route("explain this function")
        assert result.model == "gpt-4o-mini"
        assert result.task_type == "explain"
        assert result.complexity == "simple"

    def test_simple_edit_routes_cheap(self) -> None:
        router = self._router()
        result = router.route("rename this variable")
        assert result.model == "gpt-4o-mini"
        assert result.task_type == "edit"

    def test_plan_task_routes_expensive(self) -> None:
        router = self._router()
        result = router.route("refactor the authentication module")
        assert result.model == "o3"
        assert result.task_type == "plan"

    def test_complex_task_routes_expensive(self) -> None:
        router = self._router()
        # Long message with multiple file refs and code blocks → complex
        long_msg = (
            "Please refactor these files: auth.py, routes.py, models.py, utils.py, config.py "
            "and here's the current code:\n```python\ndef foo(): pass\n```\n" + "details " * 50
        )
        result = router.route(long_msg)
        assert result.model == "o3"
        assert result.complexity == "complex"

    def test_debug_medium_routes_default(self) -> None:
        router = self._router()
        result = router.route("debug this error: KeyError in the handler module")
        assert result.task_type == "debug"
        # medium complexity debug → default model (settings.model)
        assert result.model == "gpt-4o"

    def test_general_short_routes_default(self) -> None:
        router = self._router()
        result = router.route("hello there")
        assert result.model == "gpt-4o"
        assert result.reason == "standard routing"


# ===========================================================================
# ModelRouter — _classify_task
# ===========================================================================

class TestClassifyTask:
    """Task classification by keyword matching."""

    def _router(self) -> ModelRouter:
        return ModelRouter(_settings("gpt-4o"), cheap_model="gpt-4o-mini", expensive_model="o3")

    def test_plan_keywords(self) -> None:
        router = self._router()
        for kw in ("refactor", "redesign", "architect", "plan", "restructure",
                    "migrate", "build me", "create a", "implement a"):
            assert router._classify_task(f"please {kw} the code") == "plan", f"Failed for: {kw}"

    def test_debug_keywords(self) -> None:
        router = self._router()
        for kw in ("debug", "fix", "error", "bug", "broken", "crash"):
            assert router._classify_task(f"there's a {kw} here") == "debug", f"Failed for: {kw}"

    def test_explain_keywords(self) -> None:
        router = self._router()
        for kw in ("explain", "how does", "what does", "why does", "describe"):
            assert router._classify_task(f"{kw} this work") == "explain", f"Failed for: {kw}"

    def test_edit_keywords(self) -> None:
        router = self._router()
        # 'fix typo' excluded: 'fix' matches debug keywords first
        for kw in ("rename", "add comment", "remove", "delete",
                    "add type hint", "format"):
            assert router._classify_task(kw) == "edit", f"Failed for: {kw}"

    def test_fix_typo_classified_as_debug(self) -> None:
        router = self._router()
        # 'fix typo' contains 'fix' which is a debug keyword, checked before edit
        assert router._classify_task("fix typo") == "debug"

    def test_general_fallback(self) -> None:
        router = self._router()
        assert router._classify_task("hello world") == "general"

    def test_case_insensitive(self) -> None:
        router = self._router()
        assert router._classify_task("REFACTOR everything") == "plan"
        assert router._classify_task("EXPLAIN this") == "explain"

    def test_plan_takes_priority_over_debug(self) -> None:
        # "fix" is debug, but "refactor" is plan — plan keywords checked first
        router = self._router()
        assert router._classify_task("refactor and fix the module") == "plan"


# ===========================================================================
# ModelRouter — _assess_complexity
# ===========================================================================

class TestAssessComplexity:
    """Complexity assessment heuristics."""

    def _router(self) -> ModelRouter:
        return ModelRouter(_settings("gpt-4o"), cheap_model="gpt-4o-mini", expensive_model="o3")

    def test_simple_short_message(self) -> None:
        router = self._router()
        assert router._assess_complexity("rename this variable") == "simple"

    def test_simple_no_keywords_short(self) -> None:
        router = self._router()
        # < 20 words, no plan keywords, no complex indicators
        assert router._assess_complexity("hello there") == "simple"

    def test_medium_moderate_length(self) -> None:
        router = self._router()
        msg = " ".join(["word"] * 25)
        assert router._assess_complexity(msg) == "medium"

    def test_complex_long_with_plan_keywords(self) -> None:
        router = self._router()
        msg = "please refactor " + " ".join(["detail"] * 110)
        assert router._assess_complexity(msg) == "complex"

    def test_complex_many_file_refs(self) -> None:
        router = self._router()
        msg = "refactor these files: auth.py routes.py models.py utils.py config.py "
        result = router._assess_complexity(msg)
        # 5 file refs (>3) + plan keyword = complex_score >= 3
        assert result == "complex"

    def test_complex_code_blocks_plus_plan(self) -> None:
        router = self._router()
        msg = "refactor this:\n```python\ndef foo(): pass\n```\n" + "details " * 60
        assert router._assess_complexity(msg) == "complex"

    def test_medium_with_code_block_only(self) -> None:
        router = self._router()
        msg = "look at this:\n```\nprint('hi')\n```"
        result = router._assess_complexity(msg)
        # code block adds 1, short message → score=1 → medium
        assert result == "medium"

    def test_medium_plan_keyword_only(self) -> None:
        router = self._router()
        # Plan keyword (+2) but short message → score=2 → medium
        msg = "create a module"
        result = router._assess_complexity(msg)
        assert result == "medium"

    def test_file_extension_detection(self) -> None:
        router = self._router()
        # Various extensions
        msg = "update auth.py, routes.js, config.yaml, styles.css"
        result = router._assess_complexity(msg)
        # 4 file refs → +1, but no plan keyword → score=1 → medium
        assert result in ("medium", "simple")  # 4 files > 3 → +1


# ===========================================================================
# ModelRouter — _detect_cheap_model
# ===========================================================================

class TestDetectCheapModel:
    """Auto-detection of cheap model based on provider."""

    def test_gpt_family(self) -> None:
        router = ModelRouter(_settings("gpt-4.1"))
        assert router._cheap_model == "gpt-4o-mini"

    def test_claude_family(self) -> None:
        router = ModelRouter(_settings("claude-opus-4-20250514"))
        assert router._cheap_model == "claude-3-5-haiku-20241022"

    def test_gemini_family(self) -> None:
        router = ModelRouter(_settings("gemini-2.0-flash"))
        assert router._cheap_model == "gemini-2.5-flash"

    def test_unknown_family_same_model(self) -> None:
        router = ModelRouter(_settings("llama3"))
        assert router._cheap_model == "llama3"


# ===========================================================================
# _MODEL_TIERS
# ===========================================================================

class TestModelTiers:
    """Known model tier registry."""

    def test_known_tiers_exist(self) -> None:
        assert "gpt-4o-mini" in _MODEL_TIERS
        assert "gpt-4o" in _MODEL_TIERS
        assert "o3" in _MODEL_TIERS

    def test_cheap_models_are_cheap(self) -> None:
        for name, tier in _MODEL_TIERS.items():
            if tier.tier == "cheap":
                assert tier.is_cheap is True

    def test_all_tiers_have_names(self) -> None:
        for name, tier in _MODEL_TIERS.items():
            assert tier.name == name


# ===========================================================================
# Keyword sets
# ===========================================================================

class TestKeywordSets:
    """Keyword sets are non-empty and contain expected entries."""

    def test_simple_keywords_nonempty(self) -> None:
        assert len(_SIMPLE_KEYWORDS) > 0
        assert "rename" in _SIMPLE_KEYWORDS

    def test_plan_keywords_nonempty(self) -> None:
        assert len(_PLAN_KEYWORDS) > 0
        assert "refactor" in _PLAN_KEYWORDS

    def test_debug_keywords_nonempty(self) -> None:
        assert len(_DEBUG_KEYWORDS) > 0
        assert "debug" in _DEBUG_KEYWORDS

    def test_explain_keywords_nonempty(self) -> None:
        assert len(_EXPLAIN_KEYWORDS) > 0
        assert "explain" in _EXPLAIN_KEYWORDS


# ===========================================================================
# ArchitectMode
# ===========================================================================

class TestArchitectMode:
    """Architect mode: plan with expensive, execute with cheap."""

    def _architect(self, enabled: bool = True) -> ArchitectMode:
        s = _settings("gpt-4o")
        router = ModelRouter(s, cheap_model="gpt-4o-mini", expensive_model="o3")
        am = ArchitectMode(s, router)
        am.enabled = enabled
        return am

    # --- should_use ---

    def test_should_use_complex_plan(self) -> None:
        am = self._architect()
        assert am.should_use("plan", "complex") is True

    def test_should_use_complex_edit(self) -> None:
        am = self._architect()
        assert am.should_use("edit", "complex") is True

    def test_should_use_complex_debug(self) -> None:
        am = self._architect()
        assert am.should_use("debug", "complex") is True

    def test_should_use_medium_plan(self) -> None:
        am = self._architect()
        assert am.should_use("plan", "medium") is True

    def test_should_not_use_simple_plan(self) -> None:
        am = self._architect()
        assert am.should_use("plan", "simple") is False

    def test_should_not_use_complex_explain(self) -> None:
        am = self._architect()
        assert am.should_use("explain", "complex") is False

    def test_should_not_use_complex_general(self) -> None:
        am = self._architect()
        assert am.should_use("general", "complex") is False

    def test_should_not_use_simple_edit(self) -> None:
        am = self._architect()
        assert am.should_use("edit", "simple") is False

    def test_should_not_use_when_disabled(self) -> None:
        am = self._architect(enabled=False)
        assert am.should_use("plan", "complex") is False

    def test_enabled_by_default(self) -> None:
        s = _settings("gpt-4o")
        router = ModelRouter(s, cheap_model="gpt-4o-mini", expensive_model="o3")
        am = ArchitectMode(s, router)
        assert am.enabled is True

    # --- build_planning_prompt ---

    def test_planning_prompt_contains_user_request(self) -> None:
        am = self._architect()
        prompt = am.build_planning_prompt("Add OAuth login")
        assert "Add OAuth login" in prompt

    def test_planning_prompt_contains_repo_context(self) -> None:
        am = self._architect()
        prompt = am.build_planning_prompt("Add logging", repo_context="src/app.py\nsrc/utils.py")
        assert "src/app.py" in prompt
        assert "src/utils.py" in prompt

    def test_planning_prompt_instructs_no_code(self) -> None:
        am = self._architect()
        prompt = am.build_planning_prompt("anything")
        assert "do NOT write code" in prompt.lower() or "NOT write code" in prompt

    def test_planning_prompt_asks_for_file_list(self) -> None:
        am = self._architect()
        prompt = am.build_planning_prompt("anything")
        assert "files" in prompt.lower()

    # --- build_execution_prompt ---

    def test_execution_prompt_contains_plan(self) -> None:
        am = self._architect()
        prompt = am.build_execution_prompt("1. Edit app.py\n2. Add logging", "Add logging")
        assert "1. Edit app.py" in prompt

    def test_execution_prompt_contains_original_request(self) -> None:
        am = self._architect()
        prompt = am.build_execution_prompt("the plan", "Add OAuth login")
        assert "Add OAuth login" in prompt

    def test_execution_prompt_instructs_exact_edits(self) -> None:
        am = self._architect()
        prompt = am.build_execution_prompt("plan", "request")
        assert "exact" in prompt.lower() or "Execute" in prompt
