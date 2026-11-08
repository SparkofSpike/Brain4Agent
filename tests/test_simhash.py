"""Tests for SimHash near-duplicate detection.

The store reports duplicates; it does not merge them. A merge is irreversible and
the caller is an agent with a model that can read both and decide. What the store
owes the caller is noticing that two memories might be the same statement, which
is what these tests pin down.
"""

from __future__ import annotations

from pathlib import Path

from brain4agent import MemoryStore
from brain4agent.simhash import DEFAULT_THRESHOLD, fingerprint, hamming, near_duplicate


class TestFingerprint:
    def test_deterministic(self) -> None:
        text = "主库是 MySQL 8.0"
        assert fingerprint(text) == fingerprint(text)

    def test_empty_is_zero(self) -> None:
        assert fingerprint("") == 0
        assert fingerprint("   \n\t ") == 0

    def test_identical_text_matches(self) -> None:
        a = fingerprint("主库是 MySQL 8.0，部署在 192.0.2.10")
        assert near_duplicate(a, a)

    def test_one_character_difference_is_near(self) -> None:
        """A changed IP should still look like the same statement."""
        a = fingerprint("主库是 MySQL 8.0，部署在 192.0.2.10")
        b = fingerprint("主库是 MySQL 8.0，部署在 192.0.2.11")
        assert hamming(a, b) < DEFAULT_THRESHOLD
        assert near_duplicate(a, b)

    def test_different_topic_is_far(self) -> None:
        a = fingerprint("主库是 MySQL 8.0，部署在 192.0.2.10")
        b = fingerprint("缓存用 Redis，单机部署")
        assert hamming(a, b) > DEFAULT_THRESHOLD
        assert not near_duplicate(a, b)

    def test_empty_never_matches(self) -> None:
        """Two empty texts are absence, not duplication."""
        assert not near_duplicate(0, 0)
        assert not near_duplicate(0, fingerprint("something"))

    def test_bit_count_is_bounded(self) -> None:
        a = fingerprint("中文内容测试")
        assert 0 <= hamming(a, a) <= 64


class TestStoreIntegration:
    def test_duplicates_reported(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "s.db") as s:
            a = s.add("主库是 MySQL 8.0，部署在 192.0.2.10", session_id="jan")
            b = s.add("主库是 MySQL 8.0，部署在 192.0.2.11", session_id="feb")
            s.add("缓存用 Redis，单机", session_id="mar")
            assert s.duplicates_of(a) == [b]
            assert s.duplicates_of(b) == [a]

    def test_unrelated_has_no_duplicates(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "s.db") as s:
            a = s.add("主库是 MySQL")
            s.add("前端用 React")
            assert s.duplicates_of(a) == []

    def test_superseded_reports_nothing(self, tmp_path: Path) -> None:
        """A replaced memory is not the current statement of anything."""
        with MemoryStore(tmp_path / "s.db") as s:
            old = s.add("主库是 MySQL 8.0，部署在 192.0.2.10", session_id="a")
            new = s.add("主库是 MySQL 8.0，部署在 192.0.2.11", session_id="b")
            s.supersede(old, new)
            assert s.duplicates_of(old) == []
            # The replacement does not flag itself against its own predecessor,
            # because that predecessor is superseded and therefore skipped.
            assert s.duplicates_of(new) == []

    def test_surfaces_in_search(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "s.db") as s:
            s.add("主库是 MySQL 8.0，部署在 192.0.2.10", session_id="jan")
            s.add("主库是 MySQL 8.0，部署在 192.0.2.11", session_id="feb")
            hits = s.search("主库", top_k=5)
            assert any(e.possible_duplicates for e in hits.evidence)

    def test_no_false_positive_on_one_memory(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "s.db") as s:
            s.add("唯一的一条记忆")
            hits = s.search("记忆", top_k=5)
            assert all(not e.possible_duplicates for e in hits.evidence)


class TestAccessCount:
    def test_increments_on_each_search(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "a.db") as s:
            mid = s.add("主库是 MySQL")
            assert s.memory(mid).accesses == 0  # type: ignore[union-attr]
            s.search("主库")
            assert s.memory(mid).accesses == 1  # type: ignore[union-attr]
            s.search("主库")
            assert s.memory(mid).accesses == 2  # type: ignore[union-attr]

    def test_not_counted_when_not_returned(self, tmp_path: Path) -> None:
        with MemoryStore(tmp_path / "a.db") as s:
            mid = s.add("主库是 MySQL")
            s.search("完全无关的词")
            assert s.memory(mid).accesses == 0  # type: ignore[union-attr]

    def test_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "p.db"
        with MemoryStore(path) as s:
            mid = s.add("主库是 MySQL")
            s.search("主库")
        with MemoryStore(path) as s:
            assert s.memory(mid).accesses == 1  # type: ignore[union-attr]
