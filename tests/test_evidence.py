"""Tests for the evidence layer.

The point of this module is that it computes no verdict. What is tested is that
the checks it does report are correct and explainable -- a caller that trusts
``missing_terms`` must not be told a term is missing when it is present, or
present when the whole corpus lacks it.

These tests exercise the shipping implementation. Until D-035 they imported
``bench.zh.evidence``, a second copy that no benchmark script called.
"""

from __future__ import annotations

from pathlib import Path

from brain4agent import MemoryStore
from brain4agent.evidence import Evidence, SearchResult, describe, question_terms


def _seeded(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "e.db")
    store.add("杭州那台测试机的地址是 203.0.113.42，部署目录 /opt/demo-app", session_id="s1")
    store.add("媒体投稿界面增加了定时投放功能", session_id="s2")
    store.add("E 盘清理释放了 110.06GB", session_id="s3")
    return store


class TestQuestionTerms:
    def test_drops_interrogatives(self) -> None:
        assert "什么" not in question_terms("杭州测试机上装的是什么数据库？")

    def test_drops_filler(self) -> None:
        assert question_terms("的了是在") == []

    def test_keeps_specific_nouns(self) -> None:
        joined = "".join(question_terms("杭州测试机上装的是什么数据库？"))
        assert "数据库" in joined or "数据" in joined

    def test_longest_first(self) -> None:
        terms = question_terms("媒体投稿界面的定时投放")
        if len(terms) > 1:
            assert len(terms[0]) >= len(terms[-1])


class TestSearch:
    def test_returns_matching_session(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            r = store.search("杭州测试机的地址", top_k=3)
            assert not r.empty
            assert r.evidence[0].session_id == "s1"

    def test_reports_missing_terms_for_unanswerable_question(self, tmp_path: Path) -> None:
        """The signal a caller needs: what the question asks for that is absent."""
        with _seeded(tmp_path) as store:
            r = store.search("杭州测试机上装的是什么数据库？", top_k=1)
            assert not r.empty
            missing = " ".join(r.evidence[0].missing_terms)
            assert "数据库" in missing or "数据" in missing

    def test_no_missing_terms_for_answerable_question(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            r = store.search("杭州测试机的地址", top_k=1)
            assert not r.empty
            # "地址" is present in s1, so it must not be reported missing.
            assert "地址" not in r.evidence[0].missing_terms

    def test_coverage_is_bounded(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            for e in store.search("杭州测试机的地址", top_k=3).evidence:
                assert 0.0 <= e.coverage <= 1.0

    def test_term_frequencies_count_sessions(self, tmp_path: Path) -> None:
        """An agent needs to know when its query words are everywhere (D-039)."""
        with _seeded(tmp_path) as store:
            store.add("媒体平台凭据误报", session_id="s4")
            r = store.search("媒体 投稿", top_k=5)
            assert r.term_frequencies.get("媒体") == 2
            assert r.term_frequencies.get("投稿") == 1

    def test_term_frequencies_report_absent_terms(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            r = store.search("zzzqqq", top_k=1)
            assert r.term_frequencies == {"zzzqqq": 0}

    def test_related_terms_offer_words_the_query_lacks(self, tmp_path: Path) -> None:
        """The corpus suggests the third word a two-word query needs (D-040)."""
        with _seeded(tmp_path) as store:
            r = store.search("媒体 投稿", top_k=2)
            assert r.related_terms
            assert "媒体" not in r.related_terms
            assert "投稿" not in r.related_terms

    def test_empty_corpus_query_returns_no_evidence(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            r = store.search("zzzz", top_k=3)
            assert r.empty
            assert r.sessions_searched == 3
            assert "zzzz" in r.corpus_missing

    def test_top_k_respected(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            r = store.search("测试机 地址 媒体投稿 清理", top_k=2)
            assert len(r.evidence) <= 2

    def test_sessions_not_turns(self, tmp_path: Path) -> None:
        """An answer spread over a session must be visible to the caller."""
        with MemoryStore(tmp_path / "t.db") as store:
            sid = store.new_session()
            store.add("杭州测试机", session_id=sid)
            store.add("上面装的是 PostgreSQL", session_id=sid)
            r = store.search("杭州测试机上装的是什么数据库", top_k=1)
            assert not r.empty
            missing = " ".join(r.evidence[0].missing_terms)
            # The answer word appears in turn 2 of the same session, so the term
            # must not be reported missing just because turn 1 lacks it.
            assert "postgresql" not in missing.lower()

    def test_evidence_is_ranked_by_score(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            scores = [e.score for e in store.search("媒体投稿 定时投放", top_k=3).evidence]
            assert scores == sorted(scores, reverse=True)

    def test_corpus_missing_is_not_session_missing(self, tmp_path: Path) -> None:
        """D-018: absent from one session says nothing about the corpus.

        The assertion is substring-based because tokenisation depends on whether
        jieba is installed: with it the missing term is 数据库, without it the
        bigram path reports 数据 and 据库 separately. Both are the same signal.
        """
        with _seeded(tmp_path) as store:
            r = store.search("媒体投稿界面的部署目录 数据库", top_k=3)
            missing = " ".join(r.corpus_missing)
            assert "数据" in missing
            assert "部署" not in r.corpus_missing
            by_session = {e.session_id: e.missing_terms for e in r.evidence}
            assert "部署" in by_session["s2"]
            assert "媒体" in by_session["s1"]


class TestDescribe:
    def test_empty(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            assert "no evidence" in describe(store.search("zzzz", top_k=3))

    def test_mentions_missing(self, tmp_path: Path) -> None:
        with _seeded(tmp_path) as store:
            text = describe(store.search("杭州测试机上装的是什么数据库？", top_k=1))
            # 数据 is the substring both tokenisers produce: 数据库 with jieba,
            # 数据 + 据库 without it.
            assert "数据" in text


class TestAccessFallback:
    def test_annotate_credits_memory_id_when_no_match_id(self, tmp_path: Path) -> None:
        """A hand-built Evidence has no matched id; the counter falls back."""
        with _seeded(tmp_path) as store:
            mid = store.add("单独的一条记忆", session_id="s9")
            mem = store.memory(mid)
            assert mem is not None
            store._annotate(
                SearchResult(
                    query="q",
                    evidence=[Evidence(mem, 1.0, 1.0)],
                    corpus_terms=1,
                    sessions_searched=1,
                )
            )
            assert store.memory(mid).accesses == 1  # type: ignore[union-attr]

    def test_positional_arguments_keep_their_meaning(self) -> None:
        """`possible_duplicates` stays the fifth positional field."""

        class _Mem:
            id = "m1"
            session_id = "s1"
            content = "x"

        e = Evidence(_Mem(), 1.0, 1.0, [], ["dup-of-x"])
        assert e.possible_duplicates == ["dup-of-x"]
        assert e.matched_memory_id == ""
