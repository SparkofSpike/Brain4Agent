"""Tests for bench/metrics.py.

Every headline number in this project comes out of these functions. If hit@k or
MRR were computed wrong, the conclusions in docs/decisions.md would be wrong in
ways no other test could catch -- the retrieval tests check what came back, not
whether "0.9250" means what it claims to.

So this file pins the definitions, including the edge cases that only appear on
small or degenerate inputs.
"""

from __future__ import annotations

import pytest

from bench.metrics import (
    RetrievalScore,
    abstention_curve,
    aggregate,
    best_balanced,
    score_one,
)


class TestScoreOne:
    def test_direct_hit_at_rank_one(self) -> None:
        s = score_one(
            query_id="q1",
            question_type="single-session-user",
            is_abstention=False,
            gold_session_ids=frozenset({"a"}),
            ranked_session_ids=["a", "b", "c"],
        )
        assert s.hit is True
        assert s.first_rank == 1
        assert s.reciprocal_rank == 1.0
        assert s.recall == 1.0

    def test_hit_at_later_rank(self) -> None:
        s = score_one(
            query_id="q1",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"c"}),
            ranked_session_ids=["a", "b", "c"],
        )
        assert s.first_rank == 3
        assert s.reciprocal_rank == pytest.approx(1 / 3)

    def test_complete_miss(self) -> None:
        s = score_one(
            query_id="q1",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"z"}),
            ranked_session_ids=["a", "b"],
        )
        assert s.hit is False
        assert s.first_rank is None
        assert s.reciprocal_rank == 0.0
        assert s.recall == 0.0

    def test_multiple_gold_sessions_use_recall(self) -> None:
        """Two of three gold sessions found -> recall 2/3, first rank is the best."""
        s = score_one(
            query_id="q1",
            question_type="multi-session",
            is_abstention=False,
            gold_session_ids=frozenset({"a", "b", "c"}),
            ranked_session_ids=["a", "x", "b"],
        )
        assert s.hit_session_ids == frozenset({"a", "b"})
        assert s.recall == pytest.approx(2 / 3)
        assert s.first_rank == 1

    def test_stops_early_once_all_gold_found(self) -> None:
        """No point scanning the tail once every gold session has been seen."""
        s = score_one(
            query_id="q1",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"a"}),
            ranked_session_ids=["a", "b", "c", "d"],
        )
        assert s.hit_ranks == (1,)

    def test_duplicate_ids_waste_rank_positions(self) -> None:
        """Documented contract: the caller passes deduplicated sessions.

        A duplicate still consumes its enumerate position, so ranks are counted
        against the raw list. This is worth pinning because a caller that forgets
        to collapse would silently depress MRR without any error.
        """
        s = score_one(
            query_id="q1",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"b"}),
            ranked_session_ids=["a", "a", "b"],
        )
        assert s.first_rank == 3

    def test_abstention_has_no_gold(self) -> None:
        s = score_one(
            query_id="q_abs",
            question_type="t",
            is_abstention=True,
            gold_session_ids=frozenset(),
            ranked_session_ids=["a", "b"],
        )
        assert s.hit is False
        assert s.recall == 0.0
        assert s.first_rank is None

    def test_empty_ranking(self) -> None:
        s = score_one(
            query_id="q1",
            question_type="t",
            is_abstention=False,
            gold_session_ids=frozenset({"a"}),
            ranked_session_ids=[],
        )
        assert s.hit is False
        assert s.first_rank is None


class TestAggregate:
    @staticmethod
    def _hit(qid: str, rank: int, qtype: str = "t", gold: int = 1) -> RetrievalScore:
        return RetrievalScore(
            query_id=qid,
            question_type=qtype,
            is_abstention=False,
            gold_sessions=gold,
            hit_session_ids=frozenset({f"g{qid}"}),
            hit_ranks=(rank,),
        )

    @staticmethod
    def _miss(qid: str, qtype: str = "t") -> RetrievalScore:
        return RetrievalScore(
            query_id=qid,
            question_type=qtype,
            is_abstention=False,
            gold_sessions=1,
            hit_session_ids=frozenset(),
            hit_ranks=(),
        )

    @staticmethod
    def _abstention(qid: str) -> RetrievalScore:
        return RetrievalScore(
            query_id=qid,
            question_type="t",
            is_abstention=True,
            gold_sessions=0,
            hit_session_ids=frozenset(),
            hit_ranks=(),
        )

    def test_hit_at_k_counts(self) -> None:
        scores = [self._hit("1", 1), self._hit("2", 3), self._hit("3", 7), self._miss("4")]
        out = aggregate(scores)
        assert out["queries"] == 4
        assert out["hit@1"] == 0.25
        assert out["hit@5"] == 0.5
        assert out["hit@10"] == 0.75
        assert out["missed"] == 1

    def test_mrr_uses_reciprocal_rank(self) -> None:
        scores = [self._hit("1", 1), self._hit("2", 2), self._miss("3")]
        out = aggregate(scores)
        assert out["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3, abs=1e-4)

    def test_abstention_excluded_from_hit_at_k(self) -> None:
        """Mixing them would let a system that never answers look good."""
        scores = [self._hit("1", 1), self._abstention("a1"), self._abstention("a2")]
        out = aggregate(scores)
        assert out["queries"] == 1
        assert out["hit@1"] == 1.0

    def test_empty_input(self) -> None:
        assert aggregate([]) == {"queries": 0}

    def test_only_abstention_input(self) -> None:
        assert aggregate([self._abstention("a")]) == {"queries": 0}

    def test_by_type_split(self) -> None:
        scores = [
            self._hit("1", 1, qtype="single-session-user"),
            self._miss("2", qtype="single-session-user"),
            self._hit("3", 2, qtype="temporal-reasoning"),
        ]
        by_type = aggregate(scores)["by_type"]
        assert by_type["single-session-user"]["n"] == 2
        assert by_type["single-session-user"]["hit@10"] == 0.5
        assert by_type["temporal-reasoning"]["n"] == 1
        assert by_type["temporal-reasoning"]["hit@10"] == 1.0

    def test_recall_averages_over_gold_count(self) -> None:
        scores = [
            RetrievalScore("1", "t", False, 2, frozenset({"a"}), (1,)),  # 0.5
            RetrievalScore("2", "t", False, 2, frozenset({"a", "b"}), (1, 2)),  # 1.0
        ]
        assert aggregate(scores)["recall@10"] == 0.75


class TestAbstentionCurve:
    def test_empty_inputs(self) -> None:
        assert abstention_curve(answerable_top_scores=[], abstention_top_scores=[1.0]) == []
        assert abstention_curve(answerable_top_scores=[1.0], abstention_top_scores=[]) == []

    def test_normalises_by_own_maximum(self) -> None:
        """Raw scales differ between retrievers; the curve must be comparable."""
        curve = abstention_curve(
            answerable_top_scores=[10.0, 10.0],
            abstention_top_scores=[2.0, 2.0],
            thresholds=[0.0, 0.5, 1.0],
        )
        at_zero = next(r for r in curve if r["threshold"] == 0.0)
        # Nothing is <= 0 after normalisation, so no correct abstentions; every
        # answerable question is kept.
        assert at_zero["abstain_rate"] == 0.0
        assert at_zero["answer_rate"] == 1.0

        at_one = next(r for r in curve if r["threshold"] == 1.0)
        # Everything is <= 1, so every abstention is "correct" -- and no
        # answerable question survives, because abstaining on it is a miss.
        assert at_one["abstain_rate"] == 1.0
        assert at_one["answer_rate"] == 0.0

    def test_balanced_is_the_mean(self) -> None:
        curve = abstention_curve(
            answerable_top_scores=[1.0],
            abstention_top_scores=[1.0],
            thresholds=[0.5],
        )
        row = curve[0]
        assert row["balanced"] == pytest.approx((row["abstain_rate"] + row["answer_rate"]) / 2)

    def test_all_zero_scores_do_not_divide_by_zero(self) -> None:
        curve = abstention_curve(
            answerable_top_scores=[0.0],
            abstention_top_scores=[0.0],
            thresholds=[0.0, 1.0],
        )
        assert len(curve) == 2

    def test_best_balanced_picks_the_maximum(self) -> None:
        curve = [
            {"threshold": 0.0, "abstain_rate": 0.0, "answer_rate": 1.0, "balanced": 0.5},
            {"threshold": 0.5, "abstain_rate": 0.8, "answer_rate": 0.9, "balanced": 0.85},
            {"threshold": 1.0, "abstain_rate": 1.0, "answer_rate": 0.0, "balanced": 0.5},
        ]
        assert best_balanced(curve)["threshold"] == 0.5

    def test_best_balanced_empty(self) -> None:
        assert best_balanced([]) == {}
