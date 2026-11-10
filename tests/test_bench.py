"""Tests for the Stage 0 harness.

These pin behaviour that would silently corrupt numbers if it broke. A benchmark
that is subtly wrong is worse than no benchmark: it produces confident
conclusions from bad data.
"""

from __future__ import annotations

import pytest

from bench.loader import MemoryItem, Query, Task, _flatten_sessions
from bench.metrics import RetrievalScore, aggregate, score_one
from bench.pmb.run import entity_hits
from bench.pmb.run import main as pmb_main
from bench.retrieval import BM25, tokenize
from bench.runner import collapse_to_sessions


class TestTokenizer:
    def test_latin_lowercased(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]

    def test_latin_digits_kept(self) -> None:
        """Digits inside a word stay part of it; v2rayN is one token."""
        assert tokenize("v2rayN") == ["v2rayn"]
        assert "projectx_3" in tokenize("ProjectX_3.0")[0] or "geo" in tokenize("ProjectX_3.0")[0]

    def test_chinese_produces_bigrams(self) -> None:
        assert tokenize("媒体投稿") == ["媒体", "体投", "投稿"]

    def test_chinese_single_char_kept(self) -> None:
        """Dropping a one-character query would make it unsearchable."""
        assert tokenize("猫") == ["猫"]

    def test_chinese_never_returns_empty(self) -> None:
        """A run with real content must always be searchable."""
        for text in ("记忆", "神经记忆系统", "一个用户", "媒体投稿"):
            assert tokenize(text), text

    def test_pure_filler_returns_empty(self) -> None:
        """All-filler input correctly indexes nothing.

        The narrow fallback must not turn 的了 into a search term: it would then
        match every document containing those two characters in sequence.
        """
        assert tokenize("的了") == []
        assert tokenize("是在") == []

    def test_stop_char_bigrams_dropped(self) -> None:
        """的了 is filler; dropping it is safe because no content word is made of it."""
        assert "的了" not in tokenize("的了")

    def test_content_word_with_common_char_survives(self) -> None:
        """The lesson from neural-memory: 社会 must not be destroyed by 会."""
        assert tokenize("社会") == ["社会"]
        assert tokenize("会议") == ["会议"]
        assert tokenize("用户") == ["用户"]

    def test_mixed_script_split(self) -> None:
        toks = tokenize("v2rayN代理")
        assert "v2rayn" in toks
        assert "代理" in toks

    def test_empty_input(self) -> None:
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestBM25:
    def test_finds_exact_match(self) -> None:
        idx = BM25()
        idx.index([("a", "the cat sat"), ("b", "a dog barked")])
        hits = idx.search("cat")
        assert hits and hits[0][0] == "a"

    def test_ranks_by_relevance(self) -> None:
        idx = BM25()
        idx.index(
            [
                ("a", "cat cat cat"),
                ("b", "cat dog"),
                ("c", "dog dog dog"),
            ]
        )
        assert idx.search("cat")[0][0] == "a"

    def test_no_match_returns_empty(self) -> None:
        idx = BM25()
        idx.index([("a", "cat")])
        assert idx.search("zebra") == []

    def test_empty_index(self) -> None:
        assert BM25().search("anything") == []

    def test_chinese_query_matches(self) -> None:
        idx = BM25()
        idx.index([("a", "媒体投稿界面增加定时投放"), ("b", "服务器迁移完成")])
        assert idx.search("媒体投稿")[0][0] == "a"

    def test_top_k_respected(self) -> None:
        idx = BM25()
        idx.index([(str(i), "cat") for i in range(20)])
        assert len(idx.search("cat", top_k=5)) == 5


class TestScoreOne:
    def test_hit_at_rank_one(self) -> None:
        s = score_one(
            query_id="q",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"g1"}),
            ranked_session_ids=["g1", "x"],
        )
        assert s.hit and s.first_rank == 1 and s.reciprocal_rank == 1.0

    def test_miss(self) -> None:
        s = score_one(
            query_id="q",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"g1"}),
            ranked_session_ids=["x", "y"],
        )
        assert not s.hit and s.reciprocal_rank == 0.0

    def test_partial_recall(self) -> None:
        s = score_one(
            query_id="q",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"g1", "g2"}),
            ranked_session_ids=["g1", "x"],
        )
        assert s.recall == 0.5

    def test_duplicate_ids_not_double_counted(self) -> None:
        s = score_one(
            query_id="q",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"g1", "g2"}),
            ranked_session_ids=["g1", "g1", "g1"],
        )
        assert s.recall == 0.5

    def test_rank_is_first_occurrence(self) -> None:
        s = score_one(
            query_id="q",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"g2"}),
            ranked_session_ids=["x", "y", "g2"],
        )
        assert s.first_rank == 3


class TestAggregate:
    @staticmethod
    def _scores() -> list[RetrievalScore]:
        return [
            score_one(
                query_id="1",
                question_type="a",
                is_abstention=False,
                gold_session_ids=frozenset({"g"}),
                ranked_session_ids=["g"],
            ),
            score_one(
                query_id="2",
                question_type="a",
                is_abstention=False,
                gold_session_ids=frozenset({"g"}),
                ranked_session_ids=["x"],
            ),
        ]

    def test_hit_at_one(self) -> None:
        out = aggregate(self._scores())
        assert out["hit@1"] == 0.5

    def test_abstention_excluded_from_hit_rates(self) -> None:
        scores = self._scores() + [
            score_one(
                query_id="abs",
                question_type="a",
                is_abstention=True,
                gold_session_ids=frozenset({"g"}),
                ranked_session_ids=[],
            )
        ]
        out = aggregate(scores)
        # Still 2 answerable, so the rate must not move.
        assert out["queries"] == 2
        assert out["hit@1"] == 0.5

    def test_by_type_split(self) -> None:
        out = aggregate(self._scores())
        assert out["by_type"]["a"]["n"] == 2

    def test_empty_input(self) -> None:
        assert aggregate([]) == {"queries": 0}


class TestCollapseToSessions:
    def test_first_occurrence_wins(self) -> None:
        mems = [
            MemoryItem(id="s1:0", content="a", session_id="s1", timestamp="", role="user"),
            MemoryItem(id="s1:1", content="b", session_id="s1", timestamp="", role="user"),
            MemoryItem(id="s2:0", content="c", session_id="s2", timestamp="", role="user"),
        ]
        assert collapse_to_sessions(["s1:1", "s2:0", "s1:0"], mems) == ["s1", "s2"]

    def test_one_session_cannot_fill_every_slot(self) -> None:
        mems = [
            MemoryItem(id=f"s1:{i}", content="x", session_id="s1", timestamp="", role="u")
            for i in range(5)
        ]
        assert collapse_to_sessions([m.id for m in mems], mems) == ["s1"]

    def test_unknown_ids_ignored(self) -> None:
        mems = [MemoryItem(id="s1:0", content="a", session_id="s1", timestamp="", role="u")]
        assert collapse_to_sessions(["nope", "s1:0"], mems) == ["s1"]


class TestFlattenSessions:
    def test_flattens_with_stable_ids(self) -> None:
        raw = {
            "haystack_sessions": [[{"role": "user", "content": "hi"}]],
            "haystack_dates": ["2023-01-01"],
            "haystack_session_ids": ["s1"],
        }
        items = _flatten_sessions(raw)
        assert len(items) == 1
        assert items[0].id == "s1:0"
        assert items[0].turn_index == 0

    def test_empty_content_dropped(self) -> None:
        raw = {
            "haystack_sessions": [
                [{"role": "user", "content": "  "}, {"role": "user", "content": "ok"}]
            ],
            "haystack_dates": ["d"],
            "haystack_session_ids": ["s1"],
        }
        items = _flatten_sessions(raw)
        assert [i.content for i in items] == ["ok"]

    def test_task_requires_memories(self) -> None:
        q = Query(
            id="q",
            text="t",
            question_type="x",
            answer="a",
            question_date="",
            gold_session_ids=frozenset(),
        )
        with pytest.raises(ValueError, match="no memories"):
            Task(query=q, memories=[])


class TestPmbSelection:
    def test_entity_hits_requires_whole_word(self) -> None:
        """pipe inside pipeline is not a name match (D-038)."""
        names = {"b1": {"pipe"}}
        assert entity_hits("explicit error returns in pipe transforms", names) == 1
        assert entity_hits("kubernetes pipeline nodes", names) == 0

    def test_entity_hits_counts_each_visible_belief_once(self) -> None:
        names = {"b1": {"redis"}, "b2": {"redis"}}
        assert entity_hits("what's redis doing?", names) == 2


class TestPmbGate:
    """The README names PrecisionMemBench as part of the verification gate.

    It printed its pass rate and returned 0 either way, so a retrieval change
    that dropped 64/77 to 60/77 left CI green. These pin the exit code, the same
    way tests/test_e2e.py pins the Chinese gate.
    """

    def test_report_mode_passes(self) -> None:
        assert pmb_main([]) == 0

    def test_threshold_above_actual_fails(self) -> None:
        assert pmb_main(["--min-pass", "0.99"]) == 1

    def test_active_threshold_above_actual_fails(self) -> None:
        assert pmb_main(["--min-active", "0.99"]) == 1

    def test_recorded_thresholds_pass(self) -> None:
        assert pmb_main(["--min-pass", "0.90", "--min-active", "0.95"]) == 0
