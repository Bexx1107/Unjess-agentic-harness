"""Knowledge graph — tracks entities and relationships across sessions.

Stores a simple directed graph of entities (files, concepts, libraries,
patterns) and their relationships. Persisted as JSON.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A node in the knowledge graph."""
    name: str
    entity_type: str  # "file", "function", "concept", "library", "pattern", "rule"
    metadata: dict = field(default_factory=dict)
    last_seen: float = 0.0
    mention_count: int = 0

    def __post_init__(self) -> None:
        if not self.last_seen:
            self.last_seen = time.time()


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source: str  # entity name
    target: str  # entity name
    relation: str  # "uses", "modifies", "depends_on", "related_to", "contains"
    weight: float = 1.0
    last_seen: float = 0.0

    def __post_init__(self) -> None:
        if not self.last_seen:
            self.last_seen = time.time()


class KnowledgeGraph:
    """Cross-session knowledge graph.
    
    Tracks entities (files, functions, concepts, libraries) and their
    relationships. Persisted as a JSON file for cross-session memory.
    
    Args:
        storage_path: Path to the JSON storage file.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._path = storage_path or (Path.home() / ".unjess" / "memory" / "knowledge_graph.json")
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []
        self._loaded = False
        self._dirty = False

    def ensure_loaded(self) -> None:
        """Load from disk if not already loaded."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    # ----- Entity operations -----

    def add_entity(
        self,
        name: str,
        entity_type: str,
        metadata: Optional[dict] = None,
    ) -> Entity:
        """Add or update an entity in the graph.
        
        If the entity already exists, increments its mention count
        and updates last_seen.
        
        Args:
            name: Entity name (e.g. file path, function name, concept).
            entity_type: One of: file, function, concept, library, pattern, rule.
            metadata: Optional metadata dict.
            
        Returns:
            The created or updated Entity.
        """
        self.ensure_loaded()
        key = name.lower()
        
        if key in self._entities:
            entity = self._entities[key]
            entity.mention_count += 1
            entity.last_seen = time.time()
            if metadata:
                entity.metadata.update(metadata)
        else:
            entity = Entity(
                name=name,
                entity_type=entity_type,
                metadata=metadata or {},
                mention_count=1,
            )
            self._entities[key] = entity
        
        self._dirty = True
        return entity

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        self.ensure_loaded()
        return self._entities.get(name.lower())

    # ----- Relationship operations -----

    def add_relationship(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> Relationship:
        """Add a relationship between two entities.
        
        If the same relationship exists, increments its weight.
        Creates entities if they don't exist.
        
        Args:
            source: Source entity name.
            target: Target entity name.
            relation: Relationship type.
            
        Returns:
            The created or updated Relationship.
        """
        self.ensure_loaded()
        
        # Find existing relationship
        for rel in self._relationships:
            if (
                rel.source.lower() == source.lower()
                and rel.target.lower() == target.lower()
                and rel.relation == relation
            ):
                rel.weight += 1.0
                rel.last_seen = time.time()
                self._dirty = True
                return rel
        
        # Create new
        rel = Relationship(
            source=source,
            target=target,
            relation=relation,
        )
        self._relationships.append(rel)
        self._dirty = True
        return rel

    def get_related(
        self,
        entity_name: str,
        depth: int = 1,
        max_results: int = 20,
    ) -> list[tuple[Entity, Relationship]]:
        """Get entities related to the given entity.
        
        Args:
            entity_name: Entity to find relations for.
            depth: How many hops to traverse (1 = direct only).
            max_results: Maximum results.
            
        Returns:
            List of (entity, relationship) tuples.
        """
        self.ensure_loaded()
        key = entity_name.lower()
        results: list[tuple[Entity, Relationship]] = []
        visited: set[str] = {key}
        frontier: set[str] = {key}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for name in frontier:
                for rel in self._relationships:
                    # Check both directions
                    other = None
                    if rel.source.lower() == name:
                        other = rel.target.lower()
                    elif rel.target.lower() == name:
                        other = rel.source.lower()
                    
                    if other and other not in visited:
                        entity = self._entities.get(other)
                        if entity:
                            results.append((entity, rel))
                            visited.add(other)
                            next_frontier.add(other)
                            
                            if len(results) >= max_results:
                                return results
            frontier = next_frontier
        
        # Sort by relationship weight (strongest first)
        results.sort(key=lambda x: x[1].weight, reverse=True)
        return results

    # ----- Context building -----

    def get_context_for(self, query: str, max_entities: int = 10) -> str:
        """Build a context string for the system prompt.
        
        Finds entities matching the query and their relationships.
        
        Args:
            query: User query to find relevant entities.
            max_entities: Max entities to include.
            
        Returns:
            Formatted context string, or empty string.
        """
        self.ensure_loaded()
        if not self._entities:
            return ""
        
        query_words = set(query.lower().split())
        scored: list[tuple[float, Entity]] = []
        
        for entity in self._entities.values():
            score = 0.0
            name_lower = entity.name.lower()
            
            # Name match
            for word in query_words:
                if word in name_lower:
                    score += 5.0
            
            # Type boost for frequently mentioned entities
            score += min(entity.mention_count * 0.5, 5.0)
            
            # Recency boost (decay over 7 days)
            age_days = (time.time() - entity.last_seen) / 86400
            if age_days < 7:
                score += (7 - age_days) * 0.3
            
            if score > 0:
                scored.append((score, entity))
        
        if not scored:
            return ""
        
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_entities]
        
        lines = ["## Knowledge Graph Context", ""]
        for _score, entity in top:
            related = self.get_related(entity.name, depth=1, max_results=3)
            rel_str = ""
            if related:
                rel_parts = [f"{r.relation} {e.name}" for e, r in related]
                rel_str = f" → {', '.join(rel_parts)}"
            lines.append(f"- **{entity.name}** ({entity.entity_type}){rel_str}")
        
        return "\n".join(lines)

    # ----- Auto-population from tool calls -----

    def record_file_read(self, file_path: str) -> None:
        """Record that a file was read."""
        name = Path(file_path).name
        self.add_entity(name, "file", {"path": file_path})

    def record_file_write(self, file_path: str) -> None:
        """Record that a file was written/modified."""
        name = Path(file_path).name
        self.add_entity(name, "file", {"path": file_path, "modified": True})

    # Common stopwords to reject from concept recording
    _STOPWORDS: set[str] = {
        "this", "that", "these", "those", "what", "which", "who", "whom",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "and", "but", "or", "nor", "not", "so", "yet", "for", "of", "in",
        "on", "at", "to", "by", "with", "from", "up", "about", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such",
        "no", "any", "only", "own", "same", "than", "too", "very",
        "just", "also", "now", "new", "old", "big", "small", "good", "bad",
        "it", "its", "he", "she", "they", "them", "we", "us", "my", "your",
        "his", "her", "our", "their", "me", "him", "if", "ok", "okay",
        "yes", "yeah", "no", "nope", "sure", "well", "like", "get", "got",
        "let", "make", "made", "see", "look", "go", "going", "went",
        "thing", "things", "stuff", "way", "something", "anything",
        "nothing", "everything", "thought", "running", "projects",
    }

    def record_concept(self, concept: str) -> None:
        """Record a concept or topic.

        Filters out stopwords, very short strings, and generic filler words
        to keep the knowledge graph meaningful.
        """
        cleaned = concept.strip().lower()
        # Reject: too short, stopword, or purely numeric
        if len(cleaned) < 3 or cleaned in self._STOPWORDS or cleaned.isdigit():
            return
        self.add_entity(concept, "concept")

    def record_dependency(self, source_file: str, target: str, relation: str = "uses") -> None:
        """Record a dependency relationship."""
        self.add_entity(Path(source_file).name, "file", {"path": source_file})
        self.add_entity(target, "library")
        self.add_relationship(Path(source_file).name, target, relation)

    # ----- Stats -----

    @property
    def entity_count(self) -> int:
        """Number of entities in the graph."""
        self.ensure_loaded()
        return len(self._entities)

    @property
    def relationship_count(self) -> int:
        """Number of relationships in the graph."""
        self.ensure_loaded()
        return len(self._relationships)

    # ----- Persistence -----

    def save(self) -> None:
        """Save the graph to disk (only if dirty)."""
        if not self._dirty:
            return
        
        self._path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "entities": {
                key: {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "metadata": e.metadata,
                    "last_seen": e.last_seen,
                    "mention_count": e.mention_count,
                }
                for key, e in self._entities.items()
            },
            "relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "weight": r.weight,
                    "last_seen": r.last_seen,
                }
                for r in self._relationships
            ],
        }
        
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._dirty = False
        logger.debug(
            "Saved knowledge graph: %d entities, %d relationships",
            len(self._entities), len(self._relationships),
        )

    def _load(self) -> None:
        """Load the graph from disk."""
        if not self._path.exists():
            return
        
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            
            for key, edata in data.get("entities", {}).items():
                self._entities[key] = Entity(**edata)
            
            for rdata in data.get("relationships", []):
                self._relationships.append(Relationship(**rdata))
            
            logger.debug(
                "Loaded knowledge graph: %d entities, %d relationships",
                len(self._entities), len(self._relationships),
            )
        except Exception as exc:
            logger.warning("Failed to load knowledge graph: %s", exc)
