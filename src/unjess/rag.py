"""RAG engine — semantic context retrieval from past sessions and code.

Indexes conversation chunks, session summaries, and code snippets
into the vector store, then retrieves relevant context for injection
into the system prompt.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 500  # chars per chunk
_CHUNK_OVERLAP = 100  # overlap between chunks


@dataclass
class RAGResult:
    """A retrieval result with context."""
    text: str
    score: float
    source: str  # "conversation", "summary", "file", "rule"
    metadata: dict


class RAGEngine:
    """Retrieval-Augmented Generation engine.

    Indexes conversations, summaries, and code into a vector store,
    then retrieves relevant context for system prompt injection.

    Args:
        vector_store: VectorStore instance for storage.
        embedding_provider: EmbeddingProvider for generating embeddings.
    """

    def __init__(
        self,
        vector_store: Any,  # VectorStore — avoid circular import at module level
        embedding_provider: Any,  # EmbeddingProvider
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_provider

    @property
    def is_available(self) -> bool:
        """Check if RAG is functional (has both store and embedder)."""
        return self._store is not None and self._embedder is not None

    # ----- Indexing -----

    def index_conversation(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
    ) -> int:
        """Index a conversation by chunking and embedding it.

        Args:
            conversation_id: Unique conversation identifier.
            messages: Conversation messages list.

        Returns:
            Number of chunks indexed.
        """
        if not self.is_available:
            return 0

        # Extract text from conversation
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str):
                parts.append(f"{role}: {content}")

        full_text = "\n".join(parts)
        chunks = _chunk_text(full_text, _CHUNK_SIZE, _CHUNK_OVERLAP)

        if not chunks:
            return 0

        # Embed all chunks
        try:
            embeddings = self._embedder.embed(chunks)
        except Exception as exc:
            logger.warning("RAG embedding failed: %s", exc)
            return 0

        if len(embeddings) != len(chunks):
            logger.warning("Embedding count mismatch: %d chunks, %d embeddings",
                          len(chunks), len(embeddings))
            return 0

        # Store each chunk
        count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding:
                continue
            doc_id = f"conv:{conversation_id}:{i}"
            self._store.add(
                doc_id=doc_id,
                text=chunk,
                embedding=embedding,
                metadata={
                    "source": "conversation",
                    "conversation_id": conversation_id,
                    "chunk_index": i,
                    "indexed_at": time.time(),
                },
            )
            count += 1

        self._store.save()
        logger.info("Indexed %d chunks from conversation %s", count, conversation_id[:12])
        return count

    def index_summary(
        self,
        summary_id: str,
        title: str,
        summary_text: str,
        key_topics: list[str] | None = None,
    ) -> bool:
        """Index a conversation summary.

        Args:
            summary_id: Summary identifier.
            title: Session title.
            summary_text: Summary content.
            key_topics: Optional list of topics.

        Returns:
            True if indexed successfully.
        """
        if not self.is_available:
            return False

        text = f"{title}\n{summary_text}"
        if key_topics:
            text += f"\nTopics: {', '.join(key_topics)}"

        try:
            embeddings = self._embedder.embed([text])
            if not embeddings or not embeddings[0]:
                return False
        except Exception as exc:
            logger.warning("Summary embedding failed: %s", exc)
            return False

        doc_id = f"summary:{summary_id}"
        self._store.add(
            doc_id=doc_id,
            text=text,
            embedding=embeddings[0],
            metadata={
                "source": "summary",
                "summary_id": summary_id,
                "title": title,
                "indexed_at": time.time(),
            },
        )
        self._store.save()
        return True

    def index_file(self, file_path: str, content: str) -> int:
        """Index a code file by chunking it.

        Args:
            file_path: Path to the file.
            content: File content.

        Returns:
            Number of chunks indexed.
        """
        if not self.is_available or not content.strip():
            return 0

        # Use content hash to avoid re-indexing unchanged files
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        file_id = f"file:{Path(file_path).name}:{content_hash}"

        # Check if already indexed (same hash)
        if self._store.count > 0:
            # Simple check — if doc with same hash prefix exists, skip
            self._store.ensure_loaded()
            if file_id in self._store._documents:
                return 0

        chunks = _chunk_text(content, _CHUNK_SIZE, _CHUNK_OVERLAP)
        if not chunks:
            return 0

        try:
            embeddings = self._embedder.embed(chunks)
        except Exception as exc:
            logger.warning("File embedding failed for %s: %s", file_path, exc)
            return 0

        count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding:
                continue
            doc_id = f"{file_id}:{i}"
            self._store.add(
                doc_id=doc_id,
                text=chunk,
                embedding=embedding,
                metadata={
                    "source": "file",
                    "file_path": file_path,
                    "chunk_index": i,
                    "content_hash": content_hash,
                    "indexed_at": time.time(),
                },
            )
            count += 1

        if count > 0:
            self._store.save()
        return count

    def index_rule(self, rule_id: str, rule_text: str) -> bool:
        """Index a learned rule.

        Args:
            rule_id: Rule identifier.
            rule_text: Rule content.

        Returns:
            True if indexed successfully.
        """
        if not self.is_available:
            return False

        try:
            embeddings = self._embedder.embed([rule_text])
            if not embeddings or not embeddings[0]:
                return False
        except Exception as exc:
            logger.warning("Rule embedding failed: %s", exc)
            return False

        doc_id = f"rule:{rule_id}"
        self._store.add(
            doc_id=doc_id,
            text=rule_text,
            embedding=embeddings[0],
            metadata={
                "source": "rule",
                "rule_id": rule_id,
                "indexed_at": time.time(),
            },
        )
        self._store.save()
        return True

    # ----- Retrieval -----

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[RAGResult]:
        """Retrieve relevant context for a query.

        Args:
            query: The user's query text.
            top_k: Number of results to return.
            min_score: Minimum similarity threshold.

        Returns:
            List of RAGResult sorted by relevance.
        """
        if not self.is_available:
            return []

        try:
            query_embedding = self._embedder.embed_query(query)
        except Exception as exc:
            logger.debug("RAG query embedding failed: %s", exc)
            return []

        if not query_embedding:
            return []

        results = self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        return [
            RAGResult(
                text=r.text,
                score=r.score,
                source=r.metadata.get("source", "unknown"),
                metadata=r.metadata,
            )
            for r in results
        ]

    def build_context_block(
        self,
        query: str,
        max_tokens: int = 500,
        top_k: int = 5,
    ) -> str:
        """Build a context block for system prompt injection.

        Args:
            query: User query to find relevant context.
            max_tokens: Approximate token budget (chars / 4).
            top_k: Max results to consider.

        Returns:
            Formatted context string, or empty string.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        max_chars = max_tokens * 4  # rough estimate
        lines = ["## Relevant Context (from past sessions)", ""]
        used_chars = 50  # header overhead

        for result in results:
            source_label = {
                "conversation": "Past conversation",
                "summary": "Session summary",
                "file": "Code",
                "rule": "Learned rule",
            }.get(result.source, result.source)

            entry = f"**[{source_label}]** (relevance: {result.score:.0%})\n{result.text}\n"

            if used_chars + len(entry) > max_chars:
                break

            lines.append(entry)
            used_chars += len(entry)

        if len(lines) <= 2:
            return ""  # Only header, no actual results

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk.
        chunk_size: Target chunk size in characters.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a newline
        if end < len(text):
            newline_pos = text.rfind("\n", start + chunk_size // 2, end + 50)
            if newline_pos > start:
                end = newline_pos + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks
