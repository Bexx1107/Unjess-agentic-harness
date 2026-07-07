"""Tests for unjess.memory — ConversationSummary, LearnedRule, and MemoryStore."""

import json
import time
from pathlib import Path

import pytest

from unjess.memory import ConversationSummary, LearnedRule, MemoryStore


# ---------------------------------------------------------------------------
# ConversationSummary dataclass
# ---------------------------------------------------------------------------

class TestConversationSummary:
    """Tests for the ConversationSummary dataclass and its methods."""

    def test_minimal_construction(self) -> None:
        s = ConversationSummary(conversation_id="conv-1")
        assert s.conversation_id == "conv-1"
        assert s.title == ""
        assert s.summary == ""
        assert s.workspace == ""
        assert s.model == ""
        assert s.message_count == 0
        assert s.key_topics == []
        assert s.files_modified == []

    def test_created_at_auto_set(self) -> None:
        before = time.time()
        s = ConversationSummary(conversation_id="conv-1")
        after = time.time()
        assert before <= s.created_at <= after

    def test_created_at_explicit_preserved(self) -> None:
        s = ConversationSummary(conversation_id="conv-1", created_at=12345.0)
        assert s.created_at == 12345.0

    def test_to_context_title_only(self) -> None:
        s = ConversationSummary(
            conversation_id="conv-1", title="My Title", created_at=0.1,
        )
        ctx = s.to_context()
        assert "**My Title**" in ctx
        # No summary/topics/files lines beyond the header
        assert ctx.count("\n") == 0

    def test_to_context_with_summary(self) -> None:
        s = ConversationSummary(
            conversation_id="conv-1", title="T", summary="Did stuff", created_at=1.0,
        )
        ctx = s.to_context()
        assert "Did stuff" in ctx

    def test_to_context_with_topics(self) -> None:
        s = ConversationSummary(
            conversation_id="conv-1", title="T", key_topics=["a", "b"], created_at=1.0,
        )
        ctx = s.to_context()
        assert "Topics: a, b" in ctx

    def test_to_context_with_files(self) -> None:
        s = ConversationSummary(
            conversation_id="conv-1", title="T",
            files_modified=["x.py", "y.py"], created_at=1.0,
        )
        ctx = s.to_context()
        assert "Files: x.py, y.py" in ctx

    def test_to_context_files_truncated_to_five(self) -> None:
        files = [f"file{i}.py" for i in range(10)]
        s = ConversationSummary(
            conversation_id="conv-1", title="T", files_modified=files, created_at=1.0,
        )
        ctx = s.to_context()
        # Only first 5 should appear
        assert "file4.py" in ctx
        assert "file5.py" not in ctx

    def test_to_context_full(self) -> None:
        s = ConversationSummary(
            conversation_id="conv-1", title="Full",
            summary="A summary", key_topics=["testing"],
            files_modified=["main.py"], created_at=1.0,
        )
        ctx = s.to_context()
        lines = ctx.split("\n")
        assert len(lines) == 4
        assert "**Full**" in lines[0]
        assert lines[1] == "A summary"
        assert "Topics: testing" in lines[2]
        assert "Files: main.py" in lines[3]

    def test_key_topics_default_is_independent(self) -> None:
        a = ConversationSummary(conversation_id="a")
        b = ConversationSummary(conversation_id="b")
        a.key_topics.append("x")
        assert b.key_topics == []

    def test_files_modified_default_is_independent(self) -> None:
        a = ConversationSummary(conversation_id="a")
        b = ConversationSummary(conversation_id="b")
        a.files_modified.append("f.py")
        assert b.files_modified == []


# ---------------------------------------------------------------------------
# LearnedRule dataclass
# ---------------------------------------------------------------------------

class TestLearnedRule:
    """Tests for the LearnedRule dataclass."""

    def test_defaults(self) -> None:
        r = LearnedRule(id="r-1", rule="Use tabs")
        assert r.id == "r-1"
        assert r.rule == "Use tabs"
        assert r.source == "user"
        assert r.workspace == ""

    def test_created_at_auto_set(self) -> None:
        before = time.time()
        r = LearnedRule(id="r-1", rule="x")
        after = time.time()
        assert before <= r.created_at <= after

    def test_created_at_explicit_preserved(self) -> None:
        r = LearnedRule(id="r-1", rule="x", created_at=99.0)
        assert r.created_at == 99.0

    def test_custom_source(self) -> None:
        r = LearnedRule(id="r-1", rule="x", source="correction")
        assert r.source == "correction"


# ---------------------------------------------------------------------------
# MemoryStore — summaries
# ---------------------------------------------------------------------------

class TestMemoryStoreSummaries:
    """Tests for MemoryStore conversation summary CRUD operations."""

    def test_add_and_retrieve_summary(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        s = ConversationSummary(
            conversation_id="c1", title="First", summary="hello",
            created_at=100.0,
        )
        store.add_summary(s)
        result = store.get_recent_summaries(count=10)
        assert len(result) == 1
        assert result[0].conversation_id == "c1"
        assert result[0].title == "First"

    def test_add_summary_replaces_same_id(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(conversation_id="c1", title="Old", created_at=1.0))
        store.add_summary(ConversationSummary(conversation_id="c1", title="New", created_at=2.0))
        result = store.get_recent_summaries(count=10)
        assert len(result) == 1
        assert result[0].title == "New"

    def test_get_recent_ordering(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        base = 1_000_000.0  # large base so all are clearly > 0
        for i in range(5):
            store.add_summary(
                ConversationSummary(
                    conversation_id=f"c{i}", title=f"T{i}",
                    created_at=base + float(i),
                ),
            )
        recent = store.get_recent_summaries(count=3)
        assert len(recent) == 3
        # Most recent first (highest created_at)
        ids = [s.conversation_id for s in recent]
        assert ids == ["c4", "c3", "c2"]

    def test_get_recent_count_larger_than_store(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(conversation_id="c1", created_at=1.0))
        result = store.get_recent_summaries(count=100)
        assert len(result) == 1

    def test_get_recent_empty_store(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        result = store.get_recent_summaries()
        assert result == []

    def test_delete_summary_exists(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(conversation_id="c1", created_at=1.0))
        assert store.delete_summary("c1") is True
        assert store.get_recent_summaries() == []

    def test_delete_summary_not_found(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.delete_summary("nonexistent") is False

    def test_rename_summary_exists(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(conversation_id="c1", title="Old", created_at=1.0))
        assert store.rename_summary("c1", "Renamed") is True
        result = store.get_recent_summaries()
        assert result[0].title == "Renamed"

    def test_rename_summary_not_found(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.rename_summary("nonexistent", "New") is False

    def test_rename_preserves_other_fields(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="Old", summary="details",
            model="gpt-4", message_count=10, created_at=1.0,
        ))
        store.rename_summary("c1", "Updated")
        result = store.get_recent_summaries()[0]
        assert result.title == "Updated"
        assert result.summary == "details"
        assert result.model == "gpt-4"
        assert result.message_count == 10


# ---------------------------------------------------------------------------
# MemoryStore — workspace filtering
# ---------------------------------------------------------------------------

class TestMemoryStoreWorkspaceFiltering:
    """Tests for workspace-scoped summary retrieval."""

    def test_workspace_filter(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", workspace="/project/a", created_at=1.0,
        ))
        store.add_summary(ConversationSummary(
            conversation_id="c2", workspace="/project/b", created_at=2.0,
        ))
        result = store.get_recent_summaries(count=10, workspace="/project/a")
        assert len(result) == 1
        assert result[0].conversation_id == "c1"

    def test_workspace_filter_empty_returns_all(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", workspace="/a", created_at=1.0,
        ))
        store.add_summary(ConversationSummary(
            conversation_id="c2", workspace="/b", created_at=2.0,
        ))
        result = store.get_recent_summaries(count=10, workspace="")
        assert len(result) == 2

    def test_workspace_filter_case_insensitive(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", workspace="C:/Project/App", created_at=1.0,
        ))
        result = store.get_recent_summaries(count=10, workspace="c:/project/app")
        assert len(result) == 1

    def test_workspace_filter_slash_normalization(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", workspace="C:\\Users\\project", created_at=1.0,
        ))
        result = store.get_recent_summaries(count=10, workspace="C:/Users/project")
        assert len(result) == 1

    def test_workspace_filter_trailing_slash(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", workspace="/project/a/", created_at=1.0,
        ))
        result = store.get_recent_summaries(count=10, workspace="/project/a")
        assert len(result) == 1

    def test_workspace_no_match(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", workspace="/project/a", created_at=1.0,
        ))
        result = store.get_recent_summaries(count=10, workspace="/project/z")
        assert result == []


# ---------------------------------------------------------------------------
# MemoryStore — rules
# ---------------------------------------------------------------------------

class TestMemoryStoreRules:
    """Tests for MemoryStore learned-rule operations."""

    def test_learn_and_get(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        rule = store.learn("Always use type hints")
        assert rule.rule == "Always use type hints"
        assert rule.source == "user"
        assert rule.id.startswith("rule-")
        rules = store.get_rules()
        assert len(rules) == 1
        assert rules[0].id == rule.id

    def test_learn_with_workspace_and_source(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        rule = store.learn("Use tabs", workspace="/proj", source="correction")
        assert rule.workspace == "/proj"
        assert rule.source == "correction"

    def test_get_rules_no_workspace_returns_all(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.learn("Global rule")
        store.learn("Scoped rule", workspace="/proj")
        assert len(store.get_rules()) == 2

    def test_get_rules_workspace_includes_global(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.learn("Global rule")
        store.learn("Scoped rule", workspace="/proj")
        result = store.get_rules(workspace="/proj")
        assert len(result) == 2

    def test_get_rules_workspace_excludes_other_scoped(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.learn("Global rule")
        store.learn("Proj-A rule", workspace="/proj-a")
        store.learn("Proj-B rule", workspace="/proj-b")
        result = store.get_rules(workspace="/proj-a")
        assert len(result) == 2
        rule_texts = {r.rule for r in result}
        assert "Global rule" in rule_texts
        assert "Proj-A rule" in rule_texts
        assert "Proj-B rule" not in rule_texts

    def test_forget_existing_rule(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        rule = store.learn("Delete me")
        assert store.forget(rule.id) is True
        assert store.get_rules() == []

    def test_forget_nonexistent_rule(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.forget("rule-nope") is False

    def test_get_rules_empty_store(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.get_rules() == []


# ---------------------------------------------------------------------------
# MemoryStore — persistence / roundtrip
# ---------------------------------------------------------------------------

class TestMemoryStorePersistence:
    """Tests for JSON persistence — data survives across MemoryStore instances."""

    def test_summaries_persist_across_instances(self, tmp_path: Path) -> None:
        store1 = MemoryStore(tmp_path)
        store1.add_summary(ConversationSummary(
            conversation_id="c1", title="Persisted", summary="data",
            workspace="/w", model="gpt-4", message_count=5,
            key_topics=["a", "b"], files_modified=["f.py"],
            created_at=42.0,
        ))

        store2 = MemoryStore(tmp_path)
        result = store2.get_recent_summaries(count=10)
        assert len(result) == 1
        s = result[0]
        assert s.conversation_id == "c1"
        assert s.title == "Persisted"
        assert s.summary == "data"
        assert s.workspace == "/w"
        assert s.model == "gpt-4"
        assert s.message_count == 5
        assert s.key_topics == ["a", "b"]
        assert s.files_modified == ["f.py"]
        assert s.created_at == 42.0

    def test_rules_persist_across_instances(self, tmp_path: Path) -> None:
        store1 = MemoryStore(tmp_path)
        rule = store1.learn("Persist me", workspace="/w", source="auto")

        store2 = MemoryStore(tmp_path)
        rules = store2.get_rules()
        assert len(rules) == 1
        r = rules[0]
        assert r.id == rule.id
        assert r.rule == "Persist me"
        assert r.workspace == "/w"
        assert r.source == "auto"

    def test_delete_summary_persists(self, tmp_path: Path) -> None:
        store1 = MemoryStore(tmp_path)
        store1.add_summary(ConversationSummary(conversation_id="c1", created_at=1.0))
        store1.delete_summary("c1")

        store2 = MemoryStore(tmp_path)
        assert store2.get_recent_summaries() == []

    def test_rename_summary_persists(self, tmp_path: Path) -> None:
        store1 = MemoryStore(tmp_path)
        store1.add_summary(ConversationSummary(conversation_id="c1", title="Old", created_at=1.0))
        store1.rename_summary("c1", "New")

        store2 = MemoryStore(tmp_path)
        result = store2.get_recent_summaries()
        assert result[0].title == "New"

    def test_forget_rule_persists(self, tmp_path: Path) -> None:
        store1 = MemoryStore(tmp_path)
        rule = store1.learn("Forget me")
        store1.forget(rule.id)

        store2 = MemoryStore(tmp_path)
        assert store2.get_rules() == []

    def test_ensure_loaded_idempotent(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(conversation_id="c1", created_at=1.0))
        store.ensure_loaded()
        store.ensure_loaded()
        assert len(store.get_recent_summaries()) == 1

    def test_storage_dir_created_on_save(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        store = MemoryStore(nested)
        store.add_summary(ConversationSummary(conversation_id="c1", created_at=1.0))
        assert nested.exists()
        assert (nested / "conversation_summaries.json").exists()

    def test_storage_dir_created_for_rules(self, tmp_path: Path) -> None:
        nested = tmp_path / "rules" / "dir"
        store = MemoryStore(nested)
        store.learn("a rule")
        assert (nested / "learned_rules.json").exists()


# ---------------------------------------------------------------------------
# MemoryStore — corrupt / missing file handling
# ---------------------------------------------------------------------------

class TestMemoryStoreErrorHandling:
    """Tests for graceful handling of corrupt or missing persistence files."""

    def test_missing_summaries_file_loads_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.get_recent_summaries() == []

    def test_missing_rules_file_loads_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.get_rules() == []

    def test_corrupt_summaries_file_loads_empty(self, tmp_path: Path) -> None:
        (tmp_path / "conversation_summaries.json").write_text(
            "NOT VALID JSON{{{", encoding="utf-8",
        )
        store = MemoryStore(tmp_path)
        result = store.get_recent_summaries()
        assert result == []

    def test_corrupt_rules_file_loads_empty(self, tmp_path: Path) -> None:
        (tmp_path / "learned_rules.json").write_text(
            "NOT VALID JSON{{{", encoding="utf-8",
        )
        store = MemoryStore(tmp_path)
        result = store.get_rules()
        assert result == []

    def test_invalid_summary_schema_loads_empty(self, tmp_path: Path) -> None:
        # Valid JSON but wrong schema — missing required field conversation_id
        (tmp_path / "conversation_summaries.json").write_text(
            json.dumps([{"title": "no id"}]), encoding="utf-8",
        )
        store = MemoryStore(tmp_path)
        result = store.get_recent_summaries()
        assert result == []

    def test_invalid_rule_schema_loads_empty(self, tmp_path: Path) -> None:
        (tmp_path / "learned_rules.json").write_text(
            json.dumps([{"not_a_rule": True}]), encoding="utf-8",
        )
        store = MemoryStore(tmp_path)
        result = store.get_rules()
        assert result == []


# ---------------------------------------------------------------------------
# MemoryStore — context / rules block building
# ---------------------------------------------------------------------------

class TestMemoryStoreBuildBlocks:
    """Tests for build_context_block and build_rules_block."""

    def test_build_context_block_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.build_context_block() == ""

    def test_build_context_block_with_summaries(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="Session 1", summary="Did things",
            workspace="/w", created_at=1.0,
        ))
        block = store.build_context_block(workspace="/w")
        assert "## Recent Conversation History (this workspace)" in block
        assert "**Session 1**" in block
        assert "Did things" in block

    def test_build_context_block_respects_workspace(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", workspace="/a", title="A", created_at=1.0,
        ))
        store.add_summary(ConversationSummary(
            conversation_id="c2", workspace="/b", title="B", created_at=2.0,
        ))
        block = store.build_context_block(workspace="/a")
        assert "**A**" in block
        assert "**B**" not in block

    def test_build_context_block_respects_max(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        base = 1_000_000.0
        for i in range(10):
            store.add_summary(ConversationSummary(
                conversation_id=f"c{i}", title=f"T{i}",
                created_at=base + float(i),
            ))
        block = store.build_context_block(max_summaries=2)
        assert "**T9**" in block
        assert "**T8**" in block
        assert "**T7**" not in block

    def test_build_rules_block_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.build_rules_block() == ""

    def test_build_rules_block_with_global_rule(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.learn("Always lint")
        block = store.build_rules_block()
        assert "## Learned Rules" in block
        assert "- Always lint (global)" in block

    def test_build_rules_block_with_scoped_rule(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.learn("Use black", workspace="/projects/myapp")
        block = store.build_rules_block(workspace="/projects/myapp")
        assert "(myapp)" in block

    def test_build_rules_block_filters_by_workspace(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.learn("Global rule")
        store.learn("App rule", workspace="/app")
        store.learn("Other rule", workspace="/other")
        block = store.build_rules_block(workspace="/app")
        assert "Global rule" in block
        assert "App rule" in block
        assert "Other rule" not in block


# ---------------------------------------------------------------------------
# MemoryStore — search_summaries
# ---------------------------------------------------------------------------

class TestMemoryStoreSearch:
    """Tests for the search_summaries keyword search."""

    def test_search_by_title(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="Refactoring the parser",
            created_at=1.0,
        ))
        store.add_summary(ConversationSummary(
            conversation_id="c2", title="Writing tests",
            created_at=2.0,
        ))
        results = store.search_summaries("parser")
        assert len(results) == 1
        assert results[0].conversation_id == "c1"

    def test_search_by_summary_text(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="Session",
            summary="Fixed a bug in the database layer", created_at=1.0,
        ))
        results = store.search_summaries("database")
        assert len(results) == 1

    def test_search_by_topic(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="Session",
            key_topics=["typescript", "react"], created_at=1.0,
        ))
        results = store.search_summaries("typescript")
        assert len(results) == 1

    def test_search_case_insensitive(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="Docker Setup", created_at=1.0,
        ))
        results = store.search_summaries("DOCKER")
        assert len(results) == 1

    def test_search_no_results(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="Hello", created_at=1.0,
        ))
        results = store.search_summaries("nonexistentxyz")
        assert results == []

    def test_search_respects_limit(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        for i in range(10):
            store.add_summary(ConversationSummary(
                conversation_id=f"c{i}", title=f"keyword session {i}",
                created_at=float(i),
            ))
        results = store.search_summaries("keyword", limit=3)
        assert len(results) == 3

    def test_search_ranks_by_word_match_count(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="python testing",
            summary="unit tests", created_at=1.0,
        ))
        store.add_summary(ConversationSummary(
            conversation_id="c2", title="python",
            summary="language overview", created_at=2.0,
        ))
        results = store.search_summaries("python testing")
        assert len(results) == 2
        # c1 matches both words, c2 only matches one
        assert results[0].conversation_id == "c1"

    def test_search_empty_store(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.search_summaries("anything") == []


# ---------------------------------------------------------------------------
# MemoryStore — JSON file content verification
# ---------------------------------------------------------------------------

class TestMemoryStoreFileContents:
    """Tests that verify the actual JSON written to disk."""

    def test_summaries_json_structure(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.add_summary(ConversationSummary(
            conversation_id="c1", title="T", summary="S",
            workspace="/w", model="m", message_count=3,
            key_topics=["k"], files_modified=["f.py"],
            created_at=100.0,
        ))
        data = json.loads(
            (tmp_path / "conversation_summaries.json").read_text(encoding="utf-8"),
        )
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["conversation_id"] == "c1"
        assert entry["title"] == "T"
        assert entry["summary"] == "S"
        assert entry["workspace"] == "/w"
        assert entry["model"] == "m"
        assert entry["message_count"] == 3
        assert entry["key_topics"] == ["k"]
        assert entry["files_modified"] == ["f.py"]
        assert entry["created_at"] == 100.0

    def test_rules_json_structure(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        rule = store.learn("Use spaces", workspace="/w", source="auto")
        data = json.loads(
            (tmp_path / "learned_rules.json").read_text(encoding="utf-8"),
        )
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["id"] == rule.id
        assert entry["rule"] == "Use spaces"
        assert entry["source"] == "auto"
        assert entry["workspace"] == "/w"
        assert isinstance(entry["created_at"], float)


# ---------------------------------------------------------------------------
# MemoryStore — learn generates unique IDs
# ---------------------------------------------------------------------------

class TestMemoryStoreLearnIds:
    """Tests that learned rules get unique IDs."""

    def test_unique_rule_ids(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        ids = set()
        for i in range(20):
            r = store.learn(f"Rule {i}")
            ids.add(r.id)
        assert len(ids) == 20

    def test_rule_id_prefix(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        r = store.learn("x")
        assert r.id.startswith("rule-")
        assert len(r.id) == len("rule-") + 8  # rule- + 8 hex chars
