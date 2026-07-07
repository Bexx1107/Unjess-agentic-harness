"""Result aggregation — strategies for combining outputs from multiple agents.

Provides 4 strategies:
- merge: Concatenate all results
- best_pick: Select the best result by score
- synthesize: Combine results into a summary
- vote: Majority vote for discrete choices
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class AggregatedResult:
    """Result from aggregating multiple agent outputs."""

    strategy: str
    output: str
    source_count: int = 0
    metadata: dict[str, Any] | None = None

    def summary(self) -> str:
        """One-line summary."""
        return f"Aggregated ({self.strategy}) from {self.source_count} sources"


def merge(results: dict[str, str], separator: str = "\n\n---\n\n") -> AggregatedResult:
    """Merge strategy — concatenate all results.

    Args:
        results: Dict of agent_id -> result text.
        separator: Separator between results.

    Returns:
        AggregatedResult with merged output.
    """
    parts: list[str] = []
    for agent_id, result in results.items():
        parts.append(f"**[{agent_id[:8]}]:**\n{result}")

    return AggregatedResult(
        strategy="merge",
        output=separator.join(parts),
        source_count=len(results),
    )


def best_pick(
    results: dict[str, str],
    scorer: Optional[Callable[[str], float]] = None,
) -> AggregatedResult:
    """Best-pick strategy — select the highest-scoring result.

    Args:
        results: Dict of agent_id -> result text.
        scorer: Optional scoring function. If None, picks the longest result.

    Returns:
        AggregatedResult with the best result.
    """
    if not results:
        return AggregatedResult(strategy="best_pick", output="", source_count=0)

    if scorer is None:
        # Default: longest result wins
        scorer = lambda text: float(len(text))

    scored: list[tuple[float, str, str]] = []
    for agent_id, result in results.items():
        try:
            score = scorer(result)
        except Exception:
            score = 0.0
        scored.append((score, agent_id, result))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_id, best_result = scored[0]

    return AggregatedResult(
        strategy="best_pick",
        output=best_result,
        source_count=len(results),
        metadata={"winner": best_id, "score": best_score},
    )


def synthesize(
    results: dict[str, str],
    synthesizer: Optional[Callable[[list[str]], str]] = None,
) -> AggregatedResult:
    """Synthesize strategy — combine multiple results into one.

    If a synthesizer function is provided (e.g., an LLM call),
    it's used to combine the results. Otherwise, falls back to
    a simple deduplication merge.

    Args:
        results: Dict of agent_id -> result text.
        synthesizer: Optional function that takes a list of texts
                     and returns a synthesized output.

    Returns:
        AggregatedResult with synthesized output.
    """
    texts = list(results.values())

    if synthesizer and callable(synthesizer):
        try:
            output = synthesizer(texts)
        except Exception as exc:
            logger.warning("Synthesizer failed: %s", exc)
            output = "\n\n".join(texts)
    else:
        # Simple dedup merge
        seen_lines: set[str] = set()
        unique_lines: list[str] = []
        for text in texts:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped and stripped not in seen_lines:
                    seen_lines.add(stripped)
                    unique_lines.append(line)
        output = "\n".join(unique_lines)

    return AggregatedResult(
        strategy="synthesize",
        output=output,
        source_count=len(results),
    )


def vote(results: dict[str, str]) -> AggregatedResult:
    """Vote strategy — majority vote for discrete choices.

    Each result is treated as a "vote" for a choice. The choice
    with the most votes wins.

    Best for binary or categorical decisions.

    Args:
        results: Dict of agent_id -> result text.

    Returns:
        AggregatedResult with the winning choice.
    """
    if not results:
        return AggregatedResult(strategy="vote", output="", source_count=0)

    # Count votes (normalize by stripping whitespace)
    votes: dict[str, int] = {}
    for result in results.values():
        key = result.strip().lower()
        votes[key] = votes.get(key, 0) + 1

    # Find winner
    winner = max(votes, key=lambda k: votes[k])
    count = votes[winner]

    # Find original-cased version
    for result in results.values():
        if result.strip().lower() == winner:
            winner = result.strip()
            break

    return AggregatedResult(
        strategy="vote",
        output=winner,
        source_count=len(results),
        metadata={"votes": votes, "winner_count": count},
    )
