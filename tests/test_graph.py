"""Tests for the entity graph.

Not wired into search — D-024 measured it as neutral there — but it is a real
structure with real edge cases, and unused code that is wrong is worse than no
code.

The behaviour that matters: expansion must not leak through entities so common
they connect everything, and it must never return a seed as its own neighbour.
"""

from __future__ import annotations

from brain4agent.graph import COMMON_ENTITY_RATIO, EntityGraph


def build(memories: dict[str, list[str]]) -> EntityGraph:
    g = EntityGraph()
    for mid, ents in memories.items():
        g.add(mid, ents)
    g.finish()
    return g


class TestExpansion:
    def test_shares_entity(self) -> None:
        g = build({"m1": ["杭州", "部署"], "m2": ["杭州", "磁盘"]})
        boosts = g.expand([("m1", 10.0)])
        assert "m2" in boosts
        assert boosts["m2"] == 10.0 * 0.35

    def test_seeds_never_returned_as_neighbours(self) -> None:
        g = build({"m1": ["杭州"], "m2": ["杭州"]})
        boosts = g.expand([("m1", 10.0), ("m2", 9.0)])
        assert boosts == {}

    def test_no_shared_entity_means_no_boost(self) -> None:
        g = build({"m1": ["杭州"], "m2": ["成都"]})
        assert g.expand([("m1", 10.0)]) == {}

    def test_decay_applies(self) -> None:
        g = build({"a": ["x"], "b": ["x"]})
        assert g.expand([("a", 1.0)], decay=0.1)["b"] == 0.1

    def test_only_the_head_is_expanded(self) -> None:
        """Propagating from weak matches would spread noise, not signal."""
        mems = {f"m{i}": [f"e{i}"] for i in range(10)}
        mems["m9"] = ["e0"]
        g = build(mems)
        boosts = g.expand([(f"m{i}", 10.0 - i) for i in range(10)], max_seeds=2)
        assert "m9" not in boosts

    def test_boost_is_the_max_not_the_sum(self) -> None:
        """Two shared entities must not double-count into a runaway score."""
        g = build({"seed": ["a", "b"], "other": ["a", "b"]})
        assert g.expand([("seed", 10.0)])["other"] == 10.0 * 0.35


class TestCommonEntityFilter:
    def test_ubiquitous_entity_is_ignored(self) -> None:
        """An entity in most memories is background, not a connection."""
        n = 40
        mems = {f"m{i}": ["年度总结"] for i in range(n)}
        mems["seed"] = ["年度总结", "独特实体"]
        mems["target"] = ["独特实体"]
        g = build(mems)
        boosts = g.expand([("seed", 10.0)])
        # Reached through 独特实体, but not through the ubiquitous one.
        assert "target" in boosts
        non_targets = [k for k in boosts if k != "target"]
        assert non_targets == []

    def test_small_corpus_has_no_filter(self) -> None:
        """With few memories there is no meaningful background to subtract."""
        g = build({"a": ["x"], "b": ["x"], "c": ["x"]})
        assert g.expand([("a", 1.0)])

    def test_ratio_is_sane(self) -> None:
        assert 0.0 < COMMON_ENTITY_RATIO < 1.0


class TestRobustness:
    def test_empty_graph(self) -> None:
        assert EntityGraph().expand([("a", 1.0)]) == {}

    def test_empty_entities_ignored(self) -> None:
        g = build({"m1": []})
        assert g.expand([("m1", 1.0)]) == {}
        assert g.stats()["entities"] == 0

    def test_duplicate_entities_collapsed(self) -> None:
        g = build({"m1": ["x", "x", "x"]})
        assert g.stats()["entities"] == 1

    def test_unknown_seed_is_harmless(self) -> None:
        g = build({"m1": ["x"]})
        assert g.expand([("nonexistent", 1.0)]) == {}

    def test_stats_shape(self) -> None:
        stats = build({"m1": ["a", "b"], "m2": ["b", "c"]}).stats()
        assert stats["memories"] == 2
        assert stats["entities"] == 3
        assert stats["edges"] >= 1
