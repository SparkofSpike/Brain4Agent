"""Tests for MemoryStore.remember() -- the write path that distils.

The behaviour that matters is what happens on each of three outcomes, because
they are deliberately different and easy to collapse by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain4agent import MemoryStore
from brain4agent.extract import ExtractedMemory, ExtractionResult


class FakeExtractor:
    def __init__(self, result: ExtractionResult) -> None:
        self.result = result
        self.calls = 0

    def extract(self, text: str, **kwargs: object) -> ExtractionResult:
        self.calls += 1
        return self.result


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "r.db") as s:
        yield s


class TestOutcomes:
    def test_no_extractor_stores_raw(self, store: MemoryStore) -> None:
        """No key configured must still be a working store."""
        ids = store.remember("没有模型时的原文", extractor=None)
        assert len(ids) == 1
        mem = store._by_session_memory(ids[0])
        assert mem is not None
        assert mem.kind == "raw"
        assert mem.content == "没有模型时的原文"

    def test_failed_extraction_stores_raw(self, store: MemoryStore) -> None:
        """An outage degrades quality; it must not lose the turn."""
        ext = FakeExtractor(ExtractionResult((), ok=False))
        ids = store.remember("API 挂了也要存下来", extractor=ext)
        assert len(ids) == 1
        mem = store._by_session_memory(ids[0])
        assert mem is not None and mem.kind == "raw"

    def test_empty_extraction_stores_nothing(self, store: MemoryStore) -> None:
        """'Nothing durable here' must actually drop the turn.

        Storing it anyway would defeat the purpose: chit-chat would compete with
        real memories at search time.
        """
        ext = FakeExtractor(ExtractionResult((), ok=True))
        ids = store.remember("好的，谢谢", extractor=ext)
        assert ids == []
        assert len(store) == 0

    def test_multiple_memories_stored(self, store: MemoryStore) -> None:
        ext = FakeExtractor(
            ExtractionResult(
                (
                    ExtractedMemory("error", "根因在 keywords.py:443", ("keywords.py",), 0.9),
                    ExtractedMemory("fact", "下游无需改动", ("FTS",), 0.6),
                ),
                ok=True,
            )
        )
        ids = store.remember("原始对话内容", extractor=ext)
        assert len(ids) == 2
        kinds = {store._by_session_memory(i).kind for i in ids}
        assert kinds == {"error", "fact"}


class TestStructure:
    def test_metadata_round_trips(self, store: MemoryStore) -> None:
        ext = FakeExtractor(
            ExtractionResult(
                (ExtractedMemory("decision", "用 bigram 而非 jieba", ("bigram", "jieba"), 0.85),),
                ok=True,
            )
        )
        ids = store.remember("我们决定用 bigram", extractor=ext)
        mem = store._by_session_memory(ids[0])
        assert mem is not None
        assert mem.kind == "decision"
        assert mem.importance == pytest.approx(0.85)
        assert mem.entities == ("bigram", "jieba")

    def test_source_text_kept(self, store: MemoryStore) -> None:
        """The distilled sentence is indexed; the original stays for audit."""
        original = "user: 杭州测试机地址？\nassistant: 203.0.113.42，部署在 /opt/demo-app"
        ext = FakeExtractor(
            ExtractionResult((ExtractedMemory("fact", "杭州测试机地址 203.0.113.42"),), ok=True)
        )
        ids = store.remember(original, extractor=ext)
        mem = store._by_session_memory(ids[0])
        assert mem is not None
        assert mem.source_text == original
        assert mem.source_text != mem.content

    def test_all_memories_share_one_session(self, store: MemoryStore) -> None:
        ext = FakeExtractor(
            ExtractionResult((ExtractedMemory("fact", "a"), ExtractedMemory("fact", "b")), ok=True)
        )
        ids = store.remember("x", extractor=ext)
        sessions = {store._by_session_memory(i).session_id for i in ids}
        assert len(sessions) == 1

    def test_persists_across_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.db"
        ext = FakeExtractor(
            ExtractionResult((ExtractedMemory("fact", "重启后还在", (), 0.7),), ok=True)
        )
        with MemoryStore(db) as s:
            ids = s.remember("x", extractor=ext)
        with MemoryStore(db) as s2:
            mem = s2._by_session_memory(ids[0])
            assert mem is not None
            assert mem.content == "重启后还在"
            assert mem.importance == pytest.approx(0.7)


class TestSearchAfterRemember:
    def test_distilled_memory_is_findable(self, store: MemoryStore) -> None:
        ext = FakeExtractor(
            ExtractionResult(
                (ExtractedMemory("fact", "杭州测试机的地址是 203.0.113.42", ("杭州测试机",)),),
                ok=True,
            )
        )
        store.remember("杭州测试机地址？203.0.113.42", extractor=ext)
        assert store.search("杭州测试机地址").best() is not None

    def test_dropped_turn_is_not_findable(self, store: MemoryStore) -> None:
        ext = FakeExtractor(ExtractionResult((), ok=True))
        store.remember("好的谢谢", extractor=ext)
        assert store.search("好的谢谢").empty


class TestMigration:
    def test_old_database_gains_columns(self, tmp_path: Path) -> None:
        """A database written before memory structure existed must still open."""
        import sqlite3

        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, created_at TEXT NOT NULL,
                tokens TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.execute(
            "INSERT INTO memories VALUES ('x','s','user','旧数据','2026-01-01','旧 数 据')"
        )
        conn.commit()
        conn.close()

        with MemoryStore(db) as s:
            assert len(s) == 1
            mem = s._by_session_memory("x")
            assert mem is not None
            assert mem.kind == "raw"  # backfilled default
            assert mem.importance == 0.5
            assert s.search("旧数据").best() is not None
