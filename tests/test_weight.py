"""Tests for reinforcement, decay, weight-based pruning, and weight blending.

The design constraint these pin down: **nothing changes by default**. Every
published benchmark number was measured with `weight_blend=0`, and a corpus that
never calls `reinforce()` must score exactly as it did before weights existed.
The mechanism is opt-in because it has not been measured to help ranking; only
pruning has a reason to use it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain4agent import MemoryStore


class TestReinforce:
    def test_starts_at_importance(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "r.db") as s:
            mid = s.add("主库是 MySQL", importance=0.4)
            assert s.memory(mid).weight == pytest.approx(0.4)  # type: ignore[union-attr]

    def test_increments(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "r.db") as s:
            mid = s.add("主库是 MySQL")
            assert s.reinforce(mid) == pytest.approx(0.53)
            assert s.reinforce(mid) == pytest.approx(0.56)

    def test_capped_at_one(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "r.db") as s:
            mid = s.add("主库是 MySQL", importance=0.99)
            for _ in range(10):
                s.reinforce(mid)
            assert s.memory(mid).weight == pytest.approx(1.0)  # type: ignore[union-attr]

    def test_unknown_id_returns_none(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "r.db") as s:
            assert s.reinforce("nope") is None

    def test_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "r.db"
        with MemoryStore(path) as s:
            mid = s.add("主库是 MySQL")
            s.reinforce(mid)
        with MemoryStore(path) as s:
            assert s.memory(mid).weight == pytest.approx(0.53)  # type: ignore[union-attr]


class TestDecay:
    def test_multiplies(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "d.db") as s:
            mid = s.add("主库是 MySQL", importance=0.8)
            assert s.decay(factor=0.5) == 1
            assert s.memory(mid).weight == pytest.approx(0.4)  # type: ignore[union-attr]

    def test_floor_clamps(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "d.db") as s:
            mid = s.add("主库是 MySQL", importance=0.1)
            for _ in range(20):
                s.decay(factor=0.5, floor=0.05)
            assert s.memory(mid).weight == pytest.approx(0.05)  # type: ignore[union-attr]

    def test_invalid_factor_rejected(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "d.db") as s:
            with pytest.raises(ValueError, match="factor"):
                s.decay(factor=1.5)
            with pytest.raises(ValueError, match="factor"):
                s.decay(factor=0.0)

    def test_no_op_returns_zero(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "d.db") as s:
            assert s.decay(factor=1.0) == 0


class TestPruneByWeight:
    def test_dry_run_counts_without_deleting(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "p.db") as s:
            mid = s.add("主库是 MySQL", importance=0.1)
            assert s.prune_by_weight(floor=0.5, dry_run=True) == 1
            assert s.memory(mid) is not None

    def test_deletes_below_floor(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "p.db") as s:
            low = s.add("低权重", importance=0.1)
            high = s.add("高权重", importance=0.9)
            assert s.prune_by_weight(floor=0.5) == 1
            assert s.memory(low) is None
            assert s.memory(high) is not None

    def test_reinforced_memory_survives(self, tmp_path: Path) -> None:
        """The point of the whole exercise: use protects a memory from pruning."""
        with MemoryStore(tmp_path / "p.db") as s:
            mid = s.add("常用的那条", importance=0.1)
            for _ in range(5):
                s.reinforce(mid)  # 0.1 -> 0.25
            assert s.prune_by_weight(floor=0.2) == 0
            assert s.memory(mid) is not None


class TestWeightBlend:
    def test_zero_is_a_no_op(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "b.db") as s:
            s.add("主库是 MySQL 8.0", session_id="a")
            b = s.add("主库是 MySQL 8.0", session_id="b")
            for _ in range(5):
                s.reinforce(b)
            default = [e.session_id for e in s.search("主库 MySQL", top_k=2).evidence]
            explicit = [
                e.session_id for e in s.search("主库 MySQL", top_k=2, weight_blend=0.0).evidence
            ]
            assert default == explicit

    def test_blend_reorders_by_weight(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "b.db") as s:
            s.add("主库是 MySQL 8.0", session_id="a")
            b = s.add("主库是 MySQL 8.0", session_id="b")
            for _ in range(5):
                s.reinforce(b)
            blended = [
                e.session_id for e in s.search("主库 MySQL", top_k=2, weight_blend=1.0).evidence
            ]
            assert blended[0] == "b"

    def test_neutral_weight_keeps_score(self, tmp_path: Path) -> None:
        """A memory at the default 0.5 is unaffected even at full blend."""
        with MemoryStore(tmp_path / "b.db") as s:
            s.add("主库是 MySQL 8.0")
            plain = s.search("主库", top_k=1).evidence[0].score
            blended = s.search("主库", top_k=1, weight_blend=1.0).evidence[0].score
            assert plain == pytest.approx(blended)
