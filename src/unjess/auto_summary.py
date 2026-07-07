"""Auto-summary — generate conversation summaries on session end.

Calls the LLM to produce a concise summary of the session,
then stores it in the MemoryStore for future context injection.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from unjess.memory import ConversationSummary, MemoryStore

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Summarize this coding session in 3-5 sentences. Include:
1. What the user wanted to accomplish
2. Key changes made (files, features)
3. Any important decisions or patterns established

Keep it concise — this summary will be injected into future sessions as context.

Conversation:
{conversation}

Respond with ONLY the summary text, no headers or formatting."""


def _extract_heuristic_summary(
    conversation: list[dict[str, Any]],
    workspace: str = "",
    conversation_id: str = "",
) -> ConversationSummary:
    """Build a summary from conversation metadata without calling the LLM.
    
    Used as a fallback when the LLM is unavailable or fails.
    
    Args:
        conversation: The conversation message list.
        workspace: Current workspace path.
        
    Returns:
        A ConversationSummary with heuristic data.
    """
    user_messages: list[str] = []
    files_mentioned: set[str] = set()
    tool_names: set[str] = set()
    
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user" and isinstance(content, str):
            user_messages.append(content[:200])
        
        if role == "assistant" and isinstance(content, str):
            # Extract file paths mentioned in responses
            for word in content.split():
                if any(word.endswith(ext) for ext in (".py", ".js", ".ts", ".html", ".css", ".yaml", ".json", ".md", ".toml")):
                    clean = word.strip("'\"`,.:;()[]{}")
                    if "/" in clean or "\\" in clean:
                        files_mentioned.add(Path(clean).name)
                    elif clean:
                        files_mentioned.add(clean)
        
        # Track tool usage from tool role messages
        if role == "tool":
            name = msg.get("name", "")
            if name:
                tool_names.add(name)
    
    # Build title from first user message
    title = user_messages[0][:60] if user_messages else "Untitled session"
    if len(title) == 60:
        title += "..."
    
    # Build summary
    summary_parts: list[str] = []
    if user_messages:
        summary_parts.append(f"User request: {user_messages[0][:100]}")
    if files_mentioned:
        summary_parts.append(f"Files: {', '.join(sorted(files_mentioned)[:10])}")
    if tool_names:
        summary_parts.append(f"Tools used: {', '.join(sorted(tool_names))}")
    
    # Extract key topics from user messages
    all_words = " ".join(user_messages).lower().split()
    # Simple keyword extraction: words that appear 2+ times and are 4+ chars
    from collections import Counter
    word_counts = Counter(w for w in all_words if len(w) >= 4)
    key_topics = [w for w, c in word_counts.most_common(5) if c >= 2]
    
    return ConversationSummary(
        conversation_id=conversation_id or f"session-{uuid.uuid4().hex[:8]}",
        title=title,
        summary="\n".join(summary_parts) if summary_parts else "(no summary)",
        workspace=workspace,
        message_count=len(conversation),
        key_topics=key_topics,
        files_modified=sorted(files_mentioned)[:10],
    )


def generate_summary(
    conversation: list[dict[str, Any]],
    router: Any,  # ProviderRouter — avoid circular import
    settings: Any,  # Settings
    memory_store: Optional[MemoryStore] = None,
    conversation_id: str = "",
) -> Optional[ConversationSummary]:
    """Generate a conversation summary and save it.
    
    Tries to use the LLM for a high-quality summary. Falls back to
    heuristic extraction if the LLM call fails.
    
    Args:
        conversation: The full conversation message list.
        router: LLM provider router for generating the summary.
        settings: Current settings.
        memory_store: Optional memory store to save the summary to.
        
    Returns:
        The generated ConversationSummary, or None if conversation too short.
    """
    # Skip trivial conversations (less than 2 exchanges)
    user_msgs = [m for m in conversation if m.get("role") == "user"]
    if len(user_msgs) < 1:
        return None
    
    workspace = str(settings.workspace) if hasattr(settings, "workspace") else ""
    
    # Build condensed conversation text for the LLM
    condensed: list[str] = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str):
            # Truncate long messages
            text = content[:300]
            if len(content) > 300:
                text += "..."
            condensed.append(f"{role}: {text}")
    
    conv_text = "\n".join(condensed[-20:])  # Last 20 messages max
    
    # Try LLM summary
    summary_text = ""
    try:
        prompt = _SUMMARY_PROMPT.format(conversation=conv_text)
        response = router.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
        )
        summary_text = response.content.strip() if response.content else ""
    except Exception as exc:
        logger.debug("LLM summary failed, using heuristic: %s", exc)
    
    if summary_text:
        # LLM summary succeeded — build the full ConversationSummary
        heuristic = _extract_heuristic_summary(conversation, workspace, conversation_id)
        summary = ConversationSummary(
            conversation_id=heuristic.conversation_id,
            title=heuristic.title,
            summary=summary_text,
            workspace=workspace,
            model=str(settings.model) if hasattr(settings, "model") else "",
            message_count=len(conversation),
            key_topics=heuristic.key_topics,
            files_modified=heuristic.files_modified,
        )
    else:
        # Fallback to heuristic
        summary = _extract_heuristic_summary(conversation, workspace, conversation_id)
    
    # Save to memory store
    if memory_store:
        try:
            memory_store.add_summary(summary)
            logger.info("Session summary saved: %s", summary.title[:50])
        except Exception as exc:
            logger.warning("Failed to save summary: %s", exc)
    
    return summary
