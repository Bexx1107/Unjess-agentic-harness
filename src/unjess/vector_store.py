"""Vector store — pure Python embedding storage and similarity search.

No external dependencies. Uses cosine similarity for search.
Optionally uses numpy if available for faster batch operations.
"""

import json
import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try numpy for faster ops, fall back to pure Python
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


@dataclass
class Document:
    """A stored document with its embedding."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    # Embedding stored separately in binary for efficiency


@dataclass
class SearchResult:
    """A search result with similarity score."""
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """Pure Python vector store with cosine similarity search.
    
    Stores document text + metadata in a JSON file and embeddings
    in a compact binary file. Supports search with cosine similarity.
    
    Args:
        storage_dir: Directory for storage files.
        name: Store name (used for filenames).
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        name: str = "default",
    ) -> None:
        self._dir = storage_dir or (Path.home() / ".unjess" / "memory")
        self._name = name
        self._meta_path = self._dir / f"vectors_{name}.json"
        self._bin_path = self._dir / f"vectors_{name}.bin"
        self._documents: dict[str, Document] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._dimension: int = 0
        self._loaded = False
        self._dirty = False

    def ensure_loaded(self) -> None:
        """Load from disk if not already loaded."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    @property
    def count(self) -> int:
        """Number of documents in the store."""
        self.ensure_loaded()
        return len(self._documents)

    def add(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a document with its embedding.
        
        If a document with the same ID exists, it is replaced.
        
        Args:
            doc_id: Unique document identifier.
            text: Document text content.
            embedding: Embedding vector.
            metadata: Optional metadata dict.
        """
        self.ensure_loaded()
        
        if not embedding:
            return
        
        # Set dimension from first embedding
        if self._dimension == 0:
            self._dimension = len(embedding)
        elif len(embedding) != self._dimension:
            logger.warning(
                "Dimension mismatch: expected %d, got %d for doc %s",
                self._dimension, len(embedding), doc_id,
            )
            return
        
        self._documents[doc_id] = Document(
            id=doc_id,
            text=text,
            metadata=metadata or {},
        )
        self._embeddings[doc_id] = embedding
        self._dirty = True

    def remove(self, doc_id: str) -> bool:
        """Remove a document by ID.
        
        Returns:
            True if removed, False if not found.
        """
        self.ensure_loaded()
        if doc_id in self._documents:
            del self._documents[doc_id]
            del self._embeddings[doc_id]
            self._dirty = True
            return True
        return False

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            min_score: Minimum similarity score (0-1).
            
        Returns:
            List of SearchResult sorted by similarity (highest first).
        """
        self.ensure_loaded()
        
        if not self._embeddings or not query_embedding:
            return []
        
        # Compute similarities
        if _HAS_NUMPY and len(self._embeddings) > 10:
            # Use numpy for batch cosine similarity
            scores = self._search_numpy(query_embedding)
        else:
            # Pure Python
            scores = self._search_python(query_embedding)
        
        # Filter and sort
        results: list[SearchResult] = []
        for doc_id, score in scores:
            if score >= min_score:
                doc = self._documents[doc_id]
                results.append(SearchResult(
                    id=doc.id,
                    text=doc.text,
                    score=score,
                    metadata=doc.metadata,
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _search_python(self, query: list[float]) -> list[tuple[str, float]]:
        """Pure Python cosine similarity search."""
        scores: list[tuple[str, float]] = []
        for doc_id, emb in self._embeddings.items():
            score = _cosine_similarity(query, emb)
            scores.append((doc_id, score))
        return scores

    def _search_numpy(self, query: list[float]) -> list[tuple[str, float]]:
        """Numpy-accelerated cosine similarity search."""
        doc_ids = list(self._embeddings.keys())
        matrix = np.array([self._embeddings[did] for did in doc_ids])
        q = np.array(query)
        
        # Cosine similarity: dot(q, matrix) / (|q| * |matrix|)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [(did, 0.0) for did in doc_ids]
        
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0  # avoid division by zero
        
        similarities = matrix.dot(q) / (norms * q_norm)
        
        return [(doc_ids[i], float(similarities[i])) for i in range(len(doc_ids))]

    # ----- Persistence -----

    def save(self) -> None:
        """Save documents and embeddings to disk."""
        if not self._dirty:
            return
        
        self._dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata (documents) as JSON
        meta = {
            "dimension": self._dimension,
            "documents": {
                doc_id: {
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }
                for doc_id, doc in self._documents.items()
            },
        }
        self._meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        # Save embeddings as compact binary (float32)
        with open(self._bin_path, "wb") as f:
            for doc_id in self._documents:
                emb = self._embeddings.get(doc_id, [])
                f.write(struct.pack(f"{len(emb)}f", *emb))
        
        self._dirty = False
        logger.debug(
            "Saved vector store '%s': %d documents, dim=%d",
            self._name, len(self._documents), self._dimension,
        )

    def _load(self) -> None:
        """Load documents and embeddings from disk."""
        if not self._meta_path.exists():
            return
        
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._dimension = meta.get("dimension", 0)
            
            for doc_id, doc_data in meta.get("documents", {}).items():
                self._documents[doc_id] = Document(**doc_data)
            
            # Load binary embeddings
            if self._bin_path.exists() and self._dimension > 0:
                with open(self._bin_path, "rb") as f:
                    for doc_id in self._documents:
                        raw = f.read(self._dimension * 4)  # 4 bytes per float32
                        if len(raw) == self._dimension * 4:
                            emb = list(struct.unpack(f"{self._dimension}f", raw))
                            self._embeddings[doc_id] = emb
            
            logger.debug(
                "Loaded vector store '%s': %d documents, dim=%d",
                self._name, len(self._documents), self._dimension,
            )
        except Exception as exc:
            logger.warning("Failed to load vector store '%s': %s", self._name, exc)
