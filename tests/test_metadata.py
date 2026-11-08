"""Tests for the metadata layer: tags, supersession, refs.

These exist because retrieval precision stops improving once documents are
indistinguishable from each other except by their words. Every test here is a
case where the text is a perfectly good match and the answer is still "no" --
wrong scope, out of date, or the wrong kind of thing. A benchmark measured this:
PrecisionMemBench went from 0.3766 to 0.6104 once these three were wired in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain4agent import MemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "m.db") as s:
        yield s


class TestTags:
    def test_tagged_memory_found_when_scope_matches(self, store: MemoryStore) -> None:
        store.add("Redis 用于会话存储", tags=["scope:code"])
        result = store.search("Redis", tags=["scope:code"])
        assert len(result.evidence) == 1

    def test_tagged_memory_hidden_when_scope_differs(self, store: MemoryStore) -> None:
        """Same words, wrong scope, and the words are all BM25 can see."""
        store.add("Redis 用于会话存储", tags=["scope:code"])
        result = store.search("Redis", tags=["scope:writing"])
        assert result.evidence == []

    def test_untagged_memory_is_not_scope_bound(self, store: MemoryStore) -> None:
        """A memory with no scope cannot be excluded by one."""
        store.add("Redis 用于会话存储")
        result = store.search("Redis", tags=["scope:writing"])
        assert len(result.evidence) == 1

    def test_any_overlap_is_enough(self, store: MemoryStore) -> None:
        store.add("Redis 用于会话存储", tags=["scope:code", "scope:infra"])
        assert store.search("Redis", tags=["scope:infra"]).evidence
        assert store.search("Redis", tags=["scope:writing"]).evidence == []

    def test_no_tags_argument_disables_the_filter(self, store: MemoryStore) -> None:
        store.add("Redis 用于会话存储", tags=["scope:code"])
        assert store.search("Redis").evidence

    def test_tag_with_space_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(ValueError, match="space"):
            store.add("x", tags=["scope:co de"])

    def test_tags_persist(self, tmp_path: Path) -> None:
        path = tmp_path / "p.db"
        with MemoryStore(path) as s:
            mid = s.add("Redis 用于会话存储", tags=["scope:code"])
        with MemoryStore(path) as s:
            assert s.memory(mid).tags == ("scope:code",)  # type: ignore[union-attr]
            assert s.search("Redis", tags=["scope:code"]).evidence


class TestKinds:
    def test_kind_filter(self, store: MemoryStore) -> None:
        store.add("Redis 会话存储", kind="fact")
        store.add("Redis 还是 Memcached？", kind="question")
        hits = store.search("Redis", kinds=["fact"])
        assert len(hits.evidence) == 1
        assert hits.evidence[0].memory.content.startswith("Redis 会话")

    def test_kind_unset_returns_all(self, store: MemoryStore) -> None:
        store.add("Redis 会话存储", kind="fact")
        store.add("Redis 还是 Memcached？", kind="question")
        assert len(store.search("Redis").evidence) == 2


class TestSupersession:
    def test_superseded_is_hidden_by_default(self, store: MemoryStore) -> None:
        old = store.add("主库是 MySQL")
        new = store.add("主库换成 PostgreSQL 了")
        assert store.supersede(old, new)
        hits = store.search("主库")
        assert [e.memory.content for e in hits.evidence] == ["主库换成 PostgreSQL 了"]

    def test_include_superseded_brings_it_back(self, store: MemoryStore) -> None:
        old = store.add("主库是 MySQL")
        new = store.add("主库换成 PostgreSQL 了")
        store.supersede(old, new)
        # Different sessions, so the store's session-level aggregation counts them
        # separately. The point is that the filter is the only thing hiding the old
        # one, not that the row is gone.
        assert len(store.search("主库", include_superseded=True).evidence) == 2

    def test_supersede_unknown_id_returns_false(self, store: MemoryStore) -> None:
        mid = store.add("x")
        assert store.supersede(mid, "nope") is False
        assert store.supersede("nope", mid) is False

    def test_supersede_self_rejected(self, store: MemoryStore) -> None:
        mid = store.add("x")
        with pytest.raises(ValueError, match="itself"):
            store.supersede(mid, mid)

    def test_superseded_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "s.db"
        with MemoryStore(path) as s:
            old = s.add("主库是 MySQL")
            new = s.add("主库换成 PostgreSQL")
            s.supersede(old, new)
        with MemoryStore(path) as s:
            assert s.search("主库").evidence[0].memory.content == "主库换成 PostgreSQL"


class TestRefs:
    def test_link_and_expand(self, store: MemoryStore) -> None:
        redis = store.add("Redis 用于会话存储")
        rel = store.add("auth 服务依赖 Redis")
        assert store.link(rel, [redis]) == 1
        hits = store.search("auth 服务依赖", top_k=1, expand_refs=True)
        assert len(hits.evidence) == 2

    def test_expansion_is_opt_in(self, store: MemoryStore) -> None:
        redis = store.add("Redis 用于会话存储")
        rel = store.add("auth 服务依赖 Redis")
        store.link(rel, [redis])
        assert len(store.search("auth 服务依赖", top_k=1).evidence) == 1

    def test_link_to_unknown_is_skipped(self, store: MemoryStore) -> None:
        mid = store.add("auth 服务依赖 Redis")
        assert store.link(mid, ["nope"]) == 0

    def test_link_is_idempotent(self, store: MemoryStore) -> None:
        a = store.add("a 相关内容")
        b = store.add("b 相关内容")
        assert store.link(a, [b]) == 1
        assert store.link(a, [b]) == 0

    def test_expansion_respects_kind_filter(self, store: MemoryStore) -> None:
        """A relation's participant of the wrong kind must not leak in."""
        question = store.add("Redis 还是 Memcached？", kind="question")
        rel = store.add("auth 服务依赖 Redis", kind="fact")
        store.link(rel, [question])
        hits = store.search("auth 服务依赖", top_k=1, kinds=["fact"], expand_refs=True)
        assert len(hits.evidence) == 1

    def test_expansion_respects_scope_filter(self, store: MemoryStore) -> None:
        other = store.add("Redis 用于写作素材", tags=["scope:writing"])
        rel = store.add("auth 服务依赖 Redis", tags=["scope:code"])
        store.link(rel, [other])
        hits = store.search("auth 服务依赖", top_k=1, tags=["scope:code"], expand_refs=True)
        assert [e.memory.content for e in hits.evidence] == ["auth 服务依赖 Redis"]

    def test_refs_persist(self, tmp_path: Path) -> None:
        path = tmp_path / "r.db"
        with MemoryStore(path) as s:
            a = s.add("a 相关内容")
            b = s.add("b 相关内容")
            s.link(a, [b])
        with MemoryStore(path) as s:
            hits = s.search("a 相关内容", top_k=1, expand_refs=True)
            assert len(hits.evidence) == 2


class TestIndexText:
    """Index a narrower signal than what gets stored.

    The motivating measurement: indexing a belief's full explanatory text scored
    0.6104 on PrecisionMemBench; indexing only its names scored 0.7922. The prose
    shares ordinary words with every other belief, and BM25 rewards that crowding.
    """

    def test_content_is_returned_verbatim(self, store: MemoryStore) -> None:
        mid = store.add("the full explanatory text", index_text="shortname")
        assert store.memory(mid).content == "the full explanatory text"  # type: ignore[union-attr]

    def test_matching_uses_index_text(self, store: MemoryStore) -> None:
        store.add("the full explanatory text", index_text="shortname")
        assert store.search("shortname").evidence

    def test_content_words_do_not_match(self, store: MemoryStore) -> None:
        """The whole point: words that appear only in the stored text are invisible."""
        store.add("the full explanatory text about backend code", index_text="shortname")
        assert store.search("backend").evidence == []

    def test_default_indexes_content(self, store: MemoryStore) -> None:
        store.add("the full explanatory text")
        assert store.search("explanatory").evidence

    def test_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "i.db"
        with MemoryStore(path) as s:
            s.add("the full explanatory text", index_text="shortname")
        with MemoryStore(path) as s:
            assert s.search("shortname").evidence
            assert s.search("explanatory").evidence == []

    def test_works_with_distillation(self, tmp_path: Path) -> None:
        """index_text must survive the remember() path, which also writes."""
        with MemoryStore(tmp_path / "d.db") as s:
            mid = s.add("text here", index_text="narrow")
            assert s.memory(mid) is not None


class TestSupersessionRedirect:
    """A hit on a replaced memory must resolve to its replacement, not vanish.

    Dropping the old one was the first implementation and it lost the answer:
    "main store" matched the MySQL memory, the PostgreSQL memory never says
    "main store", and the search returned nothing at all. A memory system that
    forgets a fact the moment it is updated is worse than one that never had it.
    """

    def test_old_word_finds_new_memory(self, store: MemoryStore) -> None:
        old = store.add("主库是 MySQL 8.0", session_id="jan")
        new = store.add("迁到 PostgreSQL，MySQL 已下线", session_id="mar")
        store.supersede(old, new)
        hits = store.search("主库")
        assert [e.memory.content for e in hits.evidence] == ["迁到 PostgreSQL，MySQL 已下线"]

    def test_chain_resolves_to_the_end(self, store: MemoryStore) -> None:
        a = store.add("v0 用 MySQL", session_id="a")
        b = store.add("v1 用 PostgreSQL", session_id="b")
        c = store.add("v2 用 SQLite", session_id="c")
        store.supersede(a, b)
        store.supersede(b, c)
        assert [e.memory.content for e in store.search("MySQL").evidence] == ["v2 用 SQLite"]

    def test_include_superseded_does_not_redirect(self, store: MemoryStore) -> None:
        old = store.add("主库是 MySQL", session_id="jan")
        new = store.add("迁到 PostgreSQL", session_id="mar")
        store.supersede(old, new)
        hits = store.search("主库", include_superseded=True)
        assert [e.memory.content for e in hits.evidence] == ["主库是 MySQL"]

    def test_two_olds_pointing_at_one_new_are_deduped(self, store: MemoryStore) -> None:
        a = store.add("缓存用 Redis", session_id="a")
        b = store.add("缓存也用 Redis", session_id="b")
        c = store.add("缓存换 Memcached 了", session_id="c")
        store.supersede(a, c)
        store.supersede(b, c)
        hits = store.search("缓存")
        assert len(hits.evidence) == 1

    def test_cycle_does_not_hang(self, store: MemoryStore) -> None:
        """A corrupt DB should return less, never loop."""
        a = store.add("甲", session_id="a")
        b = store.add("乙", session_id="b")
        store.supersede(a, b)
        # 手动制造一个环，绕过 supersede 的单向保护
        with store._db:
            store._db.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (a, b))
        store._dirty = True
        store.search("甲")  # 不抛异常、不挂死即可
