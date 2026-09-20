"""Conversation memory — cross-session persistence, summaries, and /learn.

Stores conversation summaries and learned rules so the agent
retains knowledge across sessions.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConversationSummary:
    """A summary of a past conversation."""

    conversation_id: str
    title: str = ""
    summary: str = ""
    workspace: str = ""
    model: str = ""
    created_at: float = 0.0
    message_count: int = 0
    key_topics: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()

    def to_context(self) -> str:
        """Format as context for injection into the system prompt."""
        parts = [f"**{self.title}** ({time.strftime('%Y-%m-%d', time.localtime(self.created_at))})"]
        if self.summary:
            parts.append(self.summary)
        if self.key_topics:
            parts.append(f"Topics: {', '.join(self.key_topics)}")
        if self.files_modified:
            parts.append(f"Files: {', '.join(self.files_modified[:5])}")
        return "\n".join(parts)


@dataclass
class LearnedRule:
    """A rule learned via /learn — persisted for future sessions."""

    id: str
    rule: str
    source: str = "user"  # "user", "auto", "correction"
    created_at: float = 0.0
    workspace: str = ""  # empty = global

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class UserProfile:
    """User profile persona facts auto-extracted across sessions."""

    role: str = ""
    tech_stack: list[str] = field(default_factory=list)
    active_goals: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    last_updated: float = 0.0

    def __post_init__(self) -> None:
        if not self.last_updated:
            self.last_updated = time.time()

    def to_context(self) -> str:
        """Format persona facts for prompt injection."""
        parts = []
        if self.role:
            parts.append(f"- **User Role:** {self.role}")
        if self.tech_stack:
            parts.append(f"- **Tech Stack:** {', '.join(self.tech_stack)}")
        if self.active_goals:
            parts.append(f"- **Active Goals:** {', '.join(self.active_goals)}")
        if self.preferences:
            parts.append(f"- **Preferences:** {', '.join(self.preferences)}")
        return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """Persistent memory across conversations.

    Stores conversation summaries and learned rules in a JSON file.
    Loaded at startup, saved after each mutation.

    Args:
        storage_dir: Directory for memory files.
    """

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._summaries: list[ConversationSummary] = []
        self._rules: list[LearnedRule] = []
        self._profile: UserProfile = UserProfile()
        self._loaded = False

    def ensure_loaded(self) -> None:
        """Load from disk if not already loaded."""
        if self._loaded:
            return
        self._load_summaries()
        self._load_rules()
        self._load_profile()
        self._loaded = True

    # ----- User Profile -----

    def get_user_profile(self) -> UserProfile:
        """Get the current user profile persona."""
        self.ensure_loaded()
        return self._profile

    def update_user_profile(
        self,
        role: str = "",
        tech_stack: Optional[list[str]] = None,
        active_goals: Optional[list[str]] = None,
        preferences: Optional[list[str]] = None,
    ) -> None:
        """Update the user profile facts."""
        self.ensure_loaded()
        updated = False

        if role and role != self._profile.role:
            self._profile.role = role
            updated = True

        if tech_stack:
            for item in tech_stack:
                if item and item not in self._profile.tech_stack:
                    self._profile.tech_stack.append(item)
                    updated = True

        if active_goals:
            for goal in active_goals:
                if goal and goal not in self._profile.active_goals:
                    self._profile.active_goals.append(goal)
                    updated = True

        if preferences:
            for pref in preferences:
                if pref and pref not in self._profile.preferences:
                    self._profile.preferences.append(pref)
                    updated = True

        if updated:
            self._profile.last_updated = time.time()
            self._save_profile()

    # ----- Conversation summaries -----

    def add_summary(self, summary: ConversationSummary) -> None:
        """Store a conversation summary.

        Args:
            summary: The summary to store.
        """
        self.ensure_loaded()

        # Replace if same conversation_id exists
        existing = next((s for s in self._summaries if s.conversation_id == summary.conversation_id), None)
        if existing and existing.created_at:
            summary.created_at = existing.created_at

        self._summaries = [
            s for s in self._summaries if s.conversation_id != summary.conversation_id
        ]
        self._summaries.append(summary)
        self._save_summaries()

    def delete_summary(self, conversation_id: str) -> bool:
        """Delete a conversation summary by ID.

        Args:
            conversation_id: The conversation to delete.

        Returns:
            True if deleted, False if not found.
        """
        self.ensure_loaded()
        before = len(self._summaries)
        self._summaries = [
            s for s in self._summaries if s.conversation_id != conversation_id
        ]
        if len(self._summaries) < before:
            self._save_summaries()
            return True
        return False

    def rename_summary(self, conversation_id: str, new_title: str) -> bool:
        """Rename a conversation summary.

        Args:
            conversation_id: The conversation to rename.
            new_title: The new title.

        Returns:
            True if renamed, False if not found.
        """
        self.ensure_loaded()
        for s in self._summaries:
            if s.conversation_id == conversation_id:
                s.title = new_title
                self._save_summaries()
                return True
        return False

    def get_recent_summaries(
        self, count: int = 5, workspace: str = "",
    ) -> list[ConversationSummary]:
        """Get the most recent conversation summaries.

        Args:
            count: Max number of summaries.
            workspace: If provided, only return summaries from this workspace.

        Returns:
            Most recent summaries.
        """
        self.ensure_loaded()
        pool = self._summaries
        if workspace:
            # Normalize for comparison (case-insensitive on Windows)
            ws_norm = workspace.replace("\\", "/").rstrip("/").lower()
            pool = [
                s for s in pool
                if s.workspace.replace("\\", "/").rstrip("/").lower() == ws_norm
            ]
        sorted_sums = sorted(pool, key=lambda s: s.created_at, reverse=True)
        return sorted_sums[:count]

    def search_summaries(self, query: str, limit: int = 5) -> list[ConversationSummary]:
        """Search conversation summaries by keyword.

        Args:
            query: Search query.
            limit: Max results.

        Returns:
            Matching summaries.
        """
        self.ensure_loaded()
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[int, ConversationSummary]] = []
        for summary in self._summaries:
            score = 0
            searchable = f"{summary.title} {summary.summary} {' '.join(summary.key_topics)}".lower()
            for word in query_words:
                if word in searchable:
                    score += 1
            if score > 0:
                scored.append((score, summary))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def build_context_block(
        self, max_summaries: int = 3, workspace: str = "",
    ) -> str:
        """Build a context block from recent conversation summaries.

        For injection into the system prompt. Only includes summaries
        from the current workspace to prevent cross-project context leaking.

        Args:
            max_summaries: Max summaries to include.
            workspace: Current workspace path — only summaries from this
                workspace are included.

        Returns:
            Formatted context string.
        """
        recent = self.get_recent_summaries(max_summaries, workspace=workspace)
        if not recent:
            return ""

        lines = ["## Recent Conversation History (this workspace)", ""]
        for summary in recent:
            lines.append(summary.to_context())
            lines.append("")

        return "\n".join(lines)

    # ----- Learned rules -----

    def learn(self, rule: str, workspace: str = "", source: str = "user") -> LearnedRule:
        """Store a learned rule.

        Args:
            rule: The rule text (e.g., "Always use TypeScript for new files").
            workspace: Workspace path (empty for global rules).
            source: How the rule was learned.

        Returns:
            The created LearnedRule.
        """
        self.ensure_loaded()

        learned = LearnedRule(
            id=f"rule-{uuid.uuid4().hex[:8]}",
            rule=rule,
            source=source,
            workspace=workspace,
        )
        self._rules.append(learned)
        self._save_rules()

        logger.info("Learned rule: %s", rule[:80])
        return learned

    def get_rules(self, workspace: str = "") -> list[LearnedRule]:
        """Get learned rules, optionally filtered by workspace.

        Args:
            workspace: Workspace path to filter by (empty = all).

        Returns:
            Matching rules.
        """
        self.ensure_loaded()
        if workspace:
            return [r for r in self._rules if r.workspace in ("", workspace)]
        return list(self._rules)

    def forget(self, rule_id: str) -> bool:
        """Remove a learned rule.

        Args:
            rule_id: The rule ID to remove.

        Returns:
            True if removed.
        """
        self.ensure_loaded()
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.id != rule_id]
        if len(self._rules) < original_len:
            self._save_rules()
            return True
        return False

    def build_rules_block(self, workspace: str = "") -> str:
        """Build a rules block for the system prompt.

        Args:
            workspace: Current workspace path.

        Returns:
            Formatted rules string.
        """
        rules = self.get_rules(workspace)
        if not rules:
            return ""

        lines = ["## Learned Rules", ""]
        for rule in rules:
            scope = "(global)" if not rule.workspace else f"({Path(rule.workspace).name})"
            lines.append(f"- {rule.rule} {scope}")

        return "\n".join(lines)

    # ----- Persistence -----

    def _load_summaries(self) -> None:
        """Load summaries from disk."""
        path = self._storage_dir / "conversation_summaries.json"
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._summaries = []
            for s_data in data:
                s = ConversationSummary(**s_data)
                if not s.created_at or s.created_at == 0.0:
                    t_path = Path.home() / ".unjess" / "conversations" / s.conversation_id / "transcript.jsonl"
                    if t_path.exists():
                        s.created_at = t_path.stat().st_mtime
                    else:
                        s.created_at = 0.0
                self._summaries.append(s)
            logger.debug("Loaded %d conversation summaries", len(self._summaries))
        except Exception as exc:
            logger.warning("Failed to load summaries: %s", exc)

    def _save_summaries(self) -> None:
        """Save summaries to disk."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / "conversation_summaries.json"

        data = []
        for s in self._summaries:
            data.append({
                "conversation_id": s.conversation_id,
                "title": s.title,
                "summary": s.summary,
                "workspace": s.workspace,
                "model": s.model,
                "created_at": s.created_at,
                "message_count": s.message_count,
                "key_topics": s.key_topics,
                "files_modified": s.files_modified,
            })

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_rules(self) -> None:
        """Load rules from disk."""
        path = self._storage_dir / "learned_rules.json"
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._rules = [LearnedRule(**r) for r in data]
            logger.debug("Loaded %d learned rules", len(self._rules))
        except Exception as exc:
            logger.warning("Failed to load rules: %s", exc)

    def _save_rules(self) -> None:
        """Save rules to disk."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / "learned_rules.json"

        data = []
        for r in self._rules:
            data.append({
                "id": r.id,
                "rule": r.rule,
                "source": r.source,
                "created_at": r.created_at,
                "workspace": r.workspace,
            })

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_profile(self) -> None:
        """Load user profile from disk."""
        path = self._storage_dir / "user_profile.json"
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._profile = UserProfile(
                role=data.get("role", ""),
                tech_stack=data.get("tech_stack", []),
                active_goals=data.get("active_goals", []),
                preferences=data.get("preferences", []),
                last_updated=data.get("last_updated", 0.0),
            )
            logger.debug("Loaded user profile persona")
        except Exception as exc:
            logger.warning("Failed to load user profile: %s", exc)

    def _save_profile(self) -> None:
        """Save user profile to disk."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / "user_profile.json"

        data = {
            "role": self._profile.role,
            "tech_stack": self._profile.tech_stack,
            "active_goals": self._profile.active_goals,
            "preferences": self._profile.preferences,
            "last_updated": self._profile.last_updated,
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
