"""Tests for the store and evidence layer.

The behaviours pinned here are the ones that would silently corrupt results:
session-level retrieval, verbatim storage, and ``missing_terms`` telling the
truth about presence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from brain4agent import BM25, MemoryStore, describe, tokenize, word_tokens
from brain4agent.evidence import question_terms
from brain4agent.tokenize import both_tokens, jieba_tokens


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "brain.db"
        with MemoryStore(db) as s:
            yield s


class TestStoreBasics:
    def test_add_and_count(self, store: MemoryStore) -> None:
        sid = store.new_session()
        store.add("第一条", session_id=sid)
        store.add("第二条", session_id=sid, role="assistant")
        assert len(store) == 2
        assert store.sessions == 1

    def test_empty_content_rejected(self, store: MemoryStore) -> None:
        """An empty turn would enter the index and match nothing forever."""
        with pytest.raises(ValueError, match="empty"):
            store.add("   ")

    def test_exchange_creates_one_session(self, store: MemoryStore) -> None:
        sid = store.add_exchange(user="问题", assistant="回答")
        assert len(store.session(sid)) == 2
        assert store.sessions == 1

    def test_forget_session(self, store: MemoryStore) -> None:
        sid = store.add_exchange(user="a", assistant="b")
        other = store.add_exchange(user="c", assistant="d")
        assert store.forget_session(sid) == 2
        assert len(store) == 2
        assert store.sessions == 1
        assert store.session(other)

    def test_batch_defers_and_commits_once(self, store: MemoryStore) -> None:
        """A thousand individual commits is most of the cost of a thousand writes."""
        with store.batch():
            for i in range(50):
                store.add(f"批量第{i}条", session_id="bulk")
        assert len(store) == 50

    def test_batch_rolls_back_on_error(self, store: MemoryStore) -> None:
        """A half-written batch is worse than none."""
        store.add("批次前", session_id="keep")
        with pytest.raises(RuntimeError), store.batch():
            store.add("批次内", session_id="bulk")
            raise RuntimeError("boom")
        assert len(store) == 1
        assert store.session("bulk") == []  # rolled back
        assert len(store.session("keep")) == 1  # the earlier write survives

    def test_nested_batch_outer_failure_rolls_back_all(self, store: MemoryStore) -> None:
        """An inner batch exiting must not commit what the outer one still owns.

        With a boolean flag the inner exit cleared it and committed, so an outer
        failure afterwards left the data behind.
        """
        with pytest.raises(RuntimeError), store.batch():
            store.add("外层第一条", session_id="a")
            with store.batch():
                store.add("内层第一条", session_id="b")
            store.add("外层第二条", session_id="c")
            raise RuntimeError("boom")
        assert len(store) == 0

    def test_nested_batch_success_commits_once(self, store: MemoryStore) -> None:
        with store.batch():
            store.add("一", session_id="x")
            with store.batch():
                store.add("二", session_id="x")
            store.add("三", session_id="x")
        assert len(store) == 3

    def test_batch_result_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "b.db"
            with MemoryStore(db) as s, s.batch():
                s.add("跨进程", session_id="p")
            with MemoryStore(db) as s2:
                assert len(s2) == 1

    def test_persists_across_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "brain.db"
            with MemoryStore(db) as s:
                s.add("持久化的内容", session_id="s1")
            with MemoryStore(db) as s2:
                assert len(s2) == 1
                assert s2.search("持久化").evidence

    def test_prune_keeps_newest(self, store: MemoryStore) -> None:
        """Retention is by write order: recent work is what gets asked about."""
        for i in range(5):
            store.add(f"会话{i}的内容", session_id=f"s{i}")
        assert store.prune(keep_sessions=2) == 3
        assert store.stats()["sessions"] == 2
        assert store.session("s0") == []
        assert store.session("s4")

    def test_prune_noop_when_under_limit(self, store: MemoryStore) -> None:
        store.add("一条", session_id="a")
        assert store.prune(keep_sessions=10) == 0

    def test_prune_rejects_zero(self, store: MemoryStore) -> None:
        with pytest.raises(ValueError):
            store.prune(keep_sessions=0)

    def test_prune_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "p.db"
            with MemoryStore(db) as s:
                for i in range(4):
                    s.add(f"条目{i}", session_id=f"s{i}")
                s.prune(keep_sessions=1)
            with MemoryStore(db) as s2:
                assert s2.stats()["sessions"] == 1

    def test_content_length_is_capped(self, store: MemoryStore) -> None:
        """A conversation turn is not a document; fail loudly rather than slowly."""
        with pytest.raises(ValueError, match="too long"):
            store.add("中" * 100_001, session_id="big")

    def test_top_k_must_be_positive(self, store: MemoryStore) -> None:
        store.add("内容", session_id="s1")
        for k in (0, -1):
            with pytest.raises(ValueError, match="top_k"):
                store.search("内容", top_k=k)

    def test_second_instance_sees_first_instances_writes(self) -> None:
        """Two MemoryStore objects on one file must not serve stale indexes."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "multi.db"
            with MemoryStore(db) as a, MemoryStore(db) as b:
                a.add("甲写入的内容", session_id="x")
                # Ranked first, not "the only hit": both sessions share 写入的内容.
                assert b.search("甲写入的内容").best().session_id == "x"

                b.add("乙写入的内容", session_id="y")
                assert a.search("乙写入的内容").best().session_id == "y"

    def test_refresh_picks_up_sibling_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "multi2.db"
            with MemoryStore(db) as a, MemoryStore(db) as b:
                a.add("同一条", session_id="x")
                assert len(b) == 0  # __len__ stays cheap on purpose
                b.refresh()
                assert len(b) == 1

    def test_stats(self, store: MemoryStore) -> None:
        store.add("媒体投稿界面", session_id="s1")
        st = store.stats()
        assert st["turns"] == 1
        assert st["sessions"] == 1
        assert st["index_terms"] > 0


class TestChineseSearch:
    def test_finds_chinese_session(self, store: MemoryStore) -> None:
        sid = store.add_exchange(
            user="中文查询搜不到记忆",
            assistant="根因在 keywords.py 第 443 行，正则不含汉字",
        )
        r = store.search("中文查询为什么搜不到")
        assert not r.empty
        assert r.best().session_id == sid

    def test_no_match_is_empty(self, store: MemoryStore) -> None:
        store.add("媒体投稿界面", session_id="s1")
        assert store.search("完全不相关的主题").empty

    def test_session_level_not_turn_level(self, store: MemoryStore) -> None:
        """An answer split across two turns must be visible as one session."""
        sid = store.new_session()
        store.add("杭州测试机的情况", session_id=sid, role="user")
        store.add("上面装的是 PostgreSQL", session_id=sid, role="assistant")
        r = store.search("杭州测试机上装的是什么数据库")
        assert not r.empty
        assert "数据库" in r.best().missing_terms  # 数据库 itself absent
        # But 测试机 is present across the session, so it must not be missing.
        assert "测试机" not in r.best().missing_terms


class TestMissingTerms:
    def test_reports_absent_term(self, store: MemoryStore) -> None:
        store.add("杭州那台测试机的地址是 114.215.175.63", session_id="s1")
        r = store.search("杭州测试机上装的是什么数据库")
        assert "数据库" in r.best().missing_terms
        assert "数据库" in r.corpus_missing

    def test_no_missing_when_present(self, store: MemoryStore) -> None:
        store.add("杭州测试机的地址是 114.215.175.63", session_id="s1")
        r = store.search("杭州测试机的地址")
        assert "地址" not in r.best().missing_terms
        assert "地址" not in r.corpus_missing

    def test_corpus_missing_needs_total_absence(self, store: MemoryStore) -> None:
        """The distinction that fixed 16 false refusals: one session lacking a
        word is ordinary; the whole corpus lacking it is the signal."""
        store.add("用户问了一个关于部署的问题", session_id="s1")
        store.add("部署目录在 /opt/app", session_id="s2")

        r = store.search("部署目录在哪里")
        assert "部署" not in r.corpus_missing

        # A word no session has at all is the signal.
        r2 = store.search("部署用的是哪个数据库")
        assert "数据库" in r2.corpus_missing

    def test_session_missing_reported_per_session(self, store: MemoryStore) -> None:
        """Session-level absence still explains a ranking, even when the corpus
        knows the term from a different session."""
        store.add("用户问了问题", session_id="s1")
        store.add("部署在 /opt/app", session_id="s2")
        r = store.search("用户部署", top_k=2)
        assert not r.corpus_missing
        assert any("用户" in e.missing_terms for e in r.evidence)

    def test_corpus_missing_survives_empty_result(self, store: MemoryStore) -> None:
        """The empty path used to skip the computation entirely, dropping the one
        signal that explains why nothing matched."""
        store.add("杭州测试机的地址是 203.0.113.42", session_id="s1")
        r = store.search("zzzqqqxxx")
        assert r.empty
        assert r.corpus_missing == ["zzzqqqxxx"]
        assert r.sessions_searched == 1

    def test_coverage_in_range(self, store: MemoryStore) -> None:
        store.add("媒体投稿的排期文件在 apps/api/data", session_id="s1")
        r = store.search("媒体投稿排期")
        for e in r.evidence:
            assert 0.0 <= e.coverage <= 1.0

    def test_interrogatives_not_reported(self, store: MemoryStore) -> None:
        terms = question_terms("杭州测试机上装的是什么数据库？")
        assert "什么" not in terms
        assert any("数据库" in t or "数据" in t for t in terms)

    def test_verbs_and_adverbs_filtered(self, store: MemoryStore) -> None:
        """The 27-false-refusal lesson: 直接/应该/哪几个 are grammar, not content."""
        terms = question_terms("应该直接修复哪几个组件")
        assert "应该" not in terms
        assert "直接" not in terms
        assert "哪几个" not in terms


class TestTokenize:
    def test_chinese_bigrams(self) -> None:
        assert tokenize("媒体投稿") == ["媒体", "体投", "投稿"]

    def test_single_han_kept(self) -> None:
        assert tokenize("猫") == ["猫"]

    def test_pure_filler_dropped(self) -> None:
        assert tokenize("的了") == []

    def test_content_word_survives(self) -> None:
        """The lesson from neural-memory: 社会 must not be destroyed by 会."""
        for w in ("社会", "会议", "用户", "更新", "问题"):
            assert tokenize(w) == [w]

    def test_latin_lowercased(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]

    def test_mixed_script(self) -> None:
        toks = tokenize("v2rayN代理")
        assert "v2rayn" in toks
        assert "代理" in toks

    def test_word_tokens_returns_words(self) -> None:
        """Word-level output should not be bigram soup."""
        toks = word_tokens("杭州测试机上装的是什么数据库")
        assert any("数据库" in t for t in toks)


class TestBM25:
    def test_ranks_most_relevant_first(self) -> None:
        idx = BM25()
        idx.index([("a", "cat cat cat"), ("b", "cat dog"), ("c", "dog dog")])
        assert idx.search("cat")[0][0] == "a"

    def test_empty_query(self) -> None:
        idx = BM25()
        idx.index([("a", "cat")])
        assert idx.search("") == []

    def test_empty_index(self) -> None:
        assert BM25().search("cat") == []

    def test_top_k(self) -> None:
        idx = BM25()
        idx.index([(str(i), "cat") for i in range(10)])
        assert len(idx.search("cat", top_k=3)) == 3

    def test_separate_index_and_query_tokenisers(self) -> None:
        """The measured shape: wide on the writing side, precise on the asking side."""
        idx = BM25(index_fn=both_tokens, query_fn=jieba_tokens)
        idx.index([("a", "中文查询搜不到记忆"), ("b", "完全无关的天气情况")])
        hits = idx.search("中文查询")
        assert hits
        assert hits[0][0] == "a"

    def test_defaults_use_bigrams_both_sides(self) -> None:
        """Without explicit tokenisers, behaviour is the documented fallback."""
        idx = BM25()
        assert idx.index_fn is tokenize
        assert idx.query_fn is tokenize

    def test_ties_break_on_insertion_order(self) -> None:
        """Identical documents must not swap places run to run.

        The bug this guards: uuid ids made an entity-swapped hard negative win
        half the time, so the same query returned a different top answer.
        """
        idx = BM25()
        idx.index([("zzz", "相同 相同"), ("aaa", "相同 相同")])
        first = [i for i, _ in idx.search("相同")]
        assert first == ["zzz", "aaa"], "insertion order must win, not id order"


class TestDescribe:
    def test_empty(self, store: MemoryStore) -> None:
        r = store.search("zzzqqqxxx")
        assert "no evidence" in describe(r)

    def test_empty_still_reports_corpus_missing(self, store: MemoryStore) -> None:
        """An empty result's most useful line is which words the corpus lacks."""
        store.add("杭州测试机的地址", session_id="s1")
        text = describe(store.search("zzzqqqxxx"))
        assert "no evidence" in text
        assert "corpus has no" in text
        assert "zzzqqqxxx" in text

    def test_shows_corpus_missing(self, store: MemoryStore) -> None:
        store.add("杭州测试机的地址", session_id="s1")
        text = describe(store.search("杭州测试机上装的是什么数据库"))
        assert "corpus has no" in text
        assert "数据库" in text
