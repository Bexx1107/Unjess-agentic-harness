"""Tests for UserProfile and continuous Knowledge Graph memory."""

import pytest
from pathlib import Path
from unjess.memory import MemoryStore, UserProfile
from unjess.knowledge_graph import KnowledgeGraph
from unjess.system_prompt import build_system_prompt
from unjess.config import Settings
from unjess.tools import ToolRegistry


def test_user_profile_dataclass():
    profile = UserProfile(
        role="C++ Developer",
        tech_stack=["C++", "Photoshop SDK"],
        active_goals=["Build Jobhunt desktop app"],
        preferences=["Clean code"],
    )
    context = profile.to_context()
    assert "- **User Role:** C++ Developer" in context
    assert "- **Tech Stack:** C++, Photoshop SDK" in context
    assert "- **Active Goals:** Build Jobhunt desktop app" in context
    assert "- **Preferences:** Clean code" in context


def test_memory_store_user_profile(tmp_path: Path):
    store = MemoryStore(storage_dir=tmp_path)
    
    # Initially empty
    prof = store.get_user_profile()
    assert prof.role == ""

    # Update profile
    store.update_user_profile(
        role="Graphics Engineer",
        tech_stack=["WebGL", "OpenGL"],
        active_goals=["Game Engine"],
    )

    # Verify memory store holds values
    updated_prof = store.get_user_profile()
    assert updated_prof.role == "Graphics Engineer"
    assert "WebGL" in updated_prof.tech_stack

    # Reload from disk
    store2 = MemoryStore(storage_dir=tmp_path)
    reloaded_prof = store2.get_user_profile()
    assert reloaded_prof.role == "Graphics Engineer"
    assert "OpenGL" in reloaded_prof.tech_stack


def test_knowledge_graph_user_fact_and_fallback(tmp_path: Path):
    kg_file = tmp_path / "knowledge_graph.json"
    kg = KnowledgeGraph(storage_path=kg_file)

    kg.record_user_fact(
        subject="User",
        relation="uses_tech",
        target="Photoshop SDK",
        subject_type="user_trait",
        target_type="tech_stack",
    )
    kg.save()

    # Verify entities exist
    assert kg.entity_count == 2

    # Query matching specific term
    ctx = kg.get_context_for("Photoshop")
    assert "Photoshop SDK" in ctx

    # Query with generic terms (fallback test)
    fallback_ctx = kg.get_context_for("what do I do")
    assert "Photoshop SDK" in fallback_ctx or "User" in fallback_ctx


def test_system_prompt_user_profile(tmp_path: Path):
    settings = Settings(workspace=tmp_path)
    registry = ToolRegistry()

    profile_ctx = "- **User Role:** C++ Developer\n- **Tech Stack:** OpenGL"
    prompt = build_system_prompt(
        settings=settings,
        tool_registry=registry,
        user_profile_context=profile_ctx,
    )

    assert "<user_information>" in prompt
    assert "User Persona / Cross-Session Profile:" in prompt
    assert "- **User Role:** C++ Developer" in prompt
