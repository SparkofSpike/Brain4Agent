"""Entity co-occurrence graph.

D-023 left a specific gap. Distilled memories recall well (hit@10 1.0) but rank
poorly, because every fact is a clean, topic-focused sentence and facts about
the same subject share most of their vocabulary. IDF falls, and the top result
becomes a coin flip between memories that are all "about" the question.

Word matching cannot fix that, because the information is not in the words — it
is in which memory is *connected* to the question through what it is about.

This module supplies that connection cheaply and without a model:

- each memory records its entities (nouns and names, extracted at write time)
- an entity shared by two memories is an edge between them
- retrieval expands from the best word-matches along those edges, with decay,
  so a memory that shares a specific entity with a strong hit gets a boost even
  if it does not share the query's exact wording

Deliberately not Personalized PageRank. PPR needs a converged power iteration
over the whole graph per query, and the measured problem here is one hop deep:
"the fact I want shares an entity with a fact that already matched". One hop
plus decay is measurable, explainable, and fast. If a two-hop gain ever shows up
in a benchmark, this is the place to extend.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# An entity appearing in more than this share of memories carries no signal --
# it is the corpus's background, not a distinction. Keeping it would let every
# memory reach every other and turn the boost into a constant.
COMMON_ENTITY_RATIO = 0.25


@dataclass(slots=True)
class EntityGraph:
    """Memory <-> entity index with one-hop expansion."""

    _by_entity: dict[str, set[str]] = field(default_factory=dict)
    _by_memory: dict[str, set[str]] = field(default_factory=dict)
    _n: int = 0

    def add(self, memory_id: str, entities: list[str] | tuple[str, ...]) -> None:
        if not entities:
            return
        unique = {e for e in entities if e}
        if not unique:
            return
        self._by_memory[memory_id] = unique
        for entity in unique:
            self._by_entity.setdefault(entity, set()).add(memory_id)

    def finish(self) -> None:
        """Record corpus size so the common-entity cut-off can be applied."""
        self._n = len(self._by_memory)

    def _too_common(self, entity: str) -> bool:
        if self._n < 8:  # too small to have a meaningful background
            return False
        return len(self._by_entity.get(entity, ())) > max(8, self._n * COMMON_ENTITY_RATIO)

    def expand(
        self,
        seeds: list[tuple[str, float]],
        *,
        decay: float = 0.35,
        max_seeds: int = 5,
    ) -> dict[str, float]:
        """Boost memories that share a specific entity with a strong hit.

        ``seeds`` is the ranked BM25 output. Only the head of it is expanded:
        propagating from a weak match would spread noise rather than signal.

        Returns memory_id -> boost, excluding the seeds themselves.
        """
        boosts: dict[str, float] = {}
        seed_ids = {mid for mid, _ in seeds}

        for memory_id, score in seeds[:max_seeds]:
            for entity in self._by_memory.get(memory_id, ()):
                if self._too_common(entity):
                    continue
                for other in self._by_entity.get(entity, ()):
                    if other in seed_ids:
                        continue
                    candidate = score * decay
                    if candidate > boosts.get(other, 0.0):
                        boosts[other] = candidate
        return boosts

    def stats(self) -> dict[str, int]:
        sizes = Counter(len(v) for v in self._by_memory.values())
        return {
            "memories": len(self._by_memory),
            "entities": len(self._by_entity),
            "edges": sum(len(v) - 1 for v in self._by_entity.values() if len(v) > 1),
            "isolated": sizes.get(0, 0) + sum(1 for v in self._by_memory.values() if len(v) == 1),
        }
