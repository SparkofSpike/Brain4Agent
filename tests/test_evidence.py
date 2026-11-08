"""Tests for the evidence layer.

The point of this module is that it computes no verdict. What is tested is that
the checks it does report are correct and explainable -- a caller that trusts
``missing_terms`` must not be told a term is missing when it is present, or
present when the whole corpus lacks it.
"""

from __future__ import annotations

from bench.loader import MemoryItem
from bench.zh.evidence import (
    Evidence,
    SearchResult,
    _question_terms,
    describe,
    search,
    term_in_corpus,
)


def _mem(mid: str, sid: str, content: str, turn: int) -> MemoryItem:
    return MemoryItem(
        id=f"{sid}:{turn}", content=content, session_id=sid, timestamp="d", role="user"
    )


def _corpus() -> list[MemoryItem]:
    return [
        _mem("s1:0", "s1", "杭州那台测试机的地址是 203.0.113.42，部署目录 /opt/demo-app", 0),
        _mem("s2:0", "s2", "媒体投稿界面增加了定时投放功能", 0),
        _mem("s3:0", "s3", "E 盘清理释放了 110.06GB", 0),
    ]


class TestQuestionTerms:
    def test_drops_interrogatives(self) -> None:
        terms = _question_terms("杭州测试机上装的是什么数据库？")
        assert "什么" not in terms

    def test_drops_filler(self) -> None:
        terms = _question_terms("的了是在")
        assert terms == []

    def test_keeps_specific_nouns(self) -> None:
        terms = _question_terms("杭州测试机上装的是什么数据库？")
        joined = "".join(terms)
        assert "数据库" in joined or "数据" in joined

    def test_longest_first(self) -> None:
        terms = _question_terms("媒体投稿界面的定时投放")
        if len(terms) > 1:
            assert len(terms[0]) >= len(terms[-1])


class TestSearch:
    def test_returns_matching_session(self) -> None:
        r = search(_corpus(), "杭州测试机的地址", top_k=3)
        assert not r.empty
        assert r.evidence[0].memory.session_id == "s1"

    def test_reports_missing_terms_for_unanswerable_question(self) -> None:
        """The signal a caller needs: what the question asks for that is absent."""
        r = search(_corpus(), "杭州测试机上装的是什么数据库？", top_k=1)
        assert not r.empty
        missing = " ".join(r.evidence[0].missing_terms)
        assert "数据库" in missing or "数据" in missing

    def test_no_missing_terms_for_answerable_question(self) -> None:
        r = search(_corpus(), "杭州测试机的地址", top_k=1)
        assert not r.empty
        # "地址" is present in s1, so it must not be reported missing.
        assert "地址" not in r.evidence[0].missing_terms

    def test_coverage_is_bounded(self) -> None:
        r = search(_corpus(), "杭州测试机的地址", top_k=3)
        for e in r.evidence:
            assert 0.0 <= e.coverage <= 1.0

    def test_empty_corpus_query_returns_no_evidence(self) -> None:
        r = search(_corpus(), "zzzz", top_k=3)
        assert r.empty
        assert r.sessions_searched == 3

    def test_top_k_respected(self) -> None:
        r = search(_corpus(), "测试机 地址 媒体投稿 清理", top_k=2)
        assert len(r.evidence) <= 2

    def test_sessions_not_turns(self) -> None:
        """An answer spread over a session must be visible to the caller."""
        mems = [
            _mem("s1:0", "s1", "杭州测试机", 0),
            _mem("s1:1", "s1", "上面装的是 PostgreSQL", 1),
        ]
        r = search(mems, "杭州测试机上装的是什么数据库", top_k=1)
        assert not r.empty
        missing = " ".join(r.evidence[0].missing_terms)
        # The answer word appears in turn 2 of the same session, so the term
        # must not be reported missing just because turn 1 lacks it.
        assert "postgresql" not in missing.lower()

    def test_evidence_is_ranked_by_score(self) -> None:
        r = search(_corpus(), "媒体投稿 定时投放", top_k=3)
        scores = [e.score for e in r.evidence]
        assert scores == sorted(scores, reverse=True)


class TestTermInCorpus:
    def test_present(self) -> None:
        assert term_in_corpus(_corpus(), "部署")

    def test_absent(self) -> None:
        assert not term_in_corpus(_corpus(), "数据库")


class TestDescribe:
    def test_empty(self) -> None:
        r = SearchResult(query="q", evidence=[], corpus_terms=0, sessions_searched=0)
        assert "no evidence" in describe(r)

    def test_mentions_missing(self) -> None:
        e = Evidence(
            memory=_mem("s1:0", "s1", "x", 0),
            score=1.0,
            coverage=0.5,
            missing_terms=["数据库"],
        )
        r = SearchResult(query="q", evidence=[e], corpus_terms=10, sessions_searched=1)
        assert "数据库" in describe(r)
