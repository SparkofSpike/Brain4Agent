"""Scoring.

Three numbers matter, and they answer different questions:

- ``hit@k``     — did any gold session appear in the top k? Measures "did the
  right thing come back at all".
- ``recall@k``  — what fraction of gold sessions appeared? Matters for
  multi-session questions where several sessions carry the answer.
- ``mrr``       — how high did the first correct hit rank? A system that returns
  the answer at position 10 is not the same as one that returns it at 1, even
  though both score the same hit@10.

Abstention is scored separately and never averaged into the above: a system
that answers "not found" to everything would score perfectly on abstention and
zero on hit@k, and a single blended number would hide that.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Score for one query."""

    query_id: str
    question_type: str
    is_abstention: bool
    gold_sessions: int
    hit_session_ids: frozenset[str]
    hit_ranks: tuple[int, ...]  # 1-based rank of each gold session first seen

    @property
    def hit(self) -> bool:
        return bool(self.hit_session_ids)

    @property
    def recall(self) -> float:
        if not self.gold_sessions:
            return 0.0
        return len(self.hit_session_ids) / self.gold_sessions

    @property
    def first_rank(self) -> int | None:
        return min(self.hit_ranks) if self.hit_ranks else None

    @property
    def reciprocal_rank(self) -> float:
        r = self.first_rank
        return 1.0 / r if r else 0.0


def score_one(
    *,
    query_id: str,
    question_type: str,
    is_abstention: bool,
    gold_session_ids: frozenset[str],
    ranked_session_ids: list[str],
) -> RetrievalScore:
    """Score a ranked list of session ids against the gold set.

    The retriever is expected to return ranked *sessions* (deduplicated, best
    rank first). Ranking items and collapsing them afterwards would let one
    session occupy every slot in the top k.
    """
    seen: set[str] = set()
    hit_ids: set[str] = set()
    ranks: list[int] = []

    for rank, sid in enumerate(ranked_session_ids, start=1):
        if sid in seen:
            continue
        seen.add(sid)
        if sid in gold_session_ids:
            hit_ids.add(sid)
            ranks.append(rank)
        if len(hit_ids) == len(gold_session_ids):
            break

    return RetrievalScore(
        query_id=query_id,
        question_type=question_type,
        is_abstention=is_abstention,
        gold_sessions=len(gold_session_ids),
        hit_session_ids=frozenset(hit_ids),
        hit_ranks=tuple(ranks),
    )


def aggregate(scores: list[RetrievalScore]) -> dict[str, object]:
    """Roll per-query scores into the headline numbers, split by question type.

    Abstention questions are excluded from hit@k — there is nothing to hit —
    and reported separately by ``score_abstention``.
    """
    answerable = [s for s in scores if not s.is_abstention]
    if not answerable:
        return {"queries": 0}

    n = len(answerable)
    out: dict[str, object] = {
        "queries": n,
        "hit@1": round(sum(1 for s in answerable if s.first_rank == 1) / n, 4),
        "hit@5": round(sum(1 for s in answerable if (s.first_rank or 999) <= 5) / n, 4),
        "hit@10": round(sum(1 for s in answerable if (s.first_rank or 999) <= 10) / n, 4),
        "recall@10": round(sum(s.recall for s in answerable) / n, 4),
        "mrr": round(sum(s.reciprocal_rank for s in answerable) / n, 4),
        "missed": sum(1 for s in answerable if not s.hit),
    }

    by_type: dict[str, dict[str, float]] = {}
    for s in answerable:
        bucket = by_type.setdefault(s.question_type, {"n": 0, "hit@10": 0.0, "mrr": 0.0})
        bucket["n"] += 1
        bucket["hit@10"] += 1.0 if (s.first_rank or 999) <= 10 else 0.0
        bucket["mrr"] += s.reciprocal_rank

    out["by_type"] = {
        k: {
            "n": int(v["n"]),
            "hit@10": round(v["hit@10"] / v["n"], 4),
            "mrr": round(v["mrr"] / v["n"], 4),
        }
        for k, v in sorted(by_type.items())
    }
    return out


ABSTAIN_THRESHOLD_NOTE = """
An abstention question asks about something that was never mentioned, and every
one of them has a decoy: the user asks about a hamster but only ever mentioned
a cat, about Korea but only Japan. So a retriever that returns *something* is
wrong even if that something looks plausible.

To score this without a reader model, a retriever's top-1 score is compared
against a threshold: below it, the system is deemed to have abstained.

The threshold cannot be picked in isolation. It is a point on a tradeoff with
answerable questions, where abstaining is a miss. So the report includes the
whole curve rather than a single number picked by hand -- picking it to flatter
the system is exactly the kind of measurement this stage exists to prevent.
"""


def abstention_curve(
    *,
    answerable_top_scores: list[float],
    abstention_top_scores: list[float],
    thresholds: list[float] | None = None,
) -> list[dict[str, float]]:
    """Correct-abstention rate vs. answerable recall across thresholds.

    Both score lists are normalised to [0, 1] by their own maximum first, so the
    curve is comparable across retrievers whose raw scales differ.
    """
    if not answerable_top_scores or not abstention_top_scores:
        return []

    a_max = max(answerable_top_scores) or 1.0
    b_max = max(abstention_top_scores) or 1.0
    a = [s / a_max for s in answerable_top_scores]
    b = [s / b_max for s in abstention_top_scores]

    if thresholds is None:
        thresholds = [i / 20 for i in range(21)]

    rows: list[dict[str, float]] = []
    for t in thresholds:
        # `<=` because a system abstains when its best evidence is weak.
        correct_abstain = sum(1 for s in b if s <= t) / len(b)
        answerable_kept = sum(1 for s in a if s > t) / len(a)
        rows.append(
            {
                "threshold": round(t, 2),
                "abstain_rate": round(correct_abstain, 4),
                "answer_rate": round(answerable_kept, 4),
                "balanced": round((correct_abstain + answerable_kept) / 2, 4),
            }
        )
    return rows


def best_balanced(curve: list[dict[str, float]]) -> dict[str, float]:
    """The threshold maximising the mean of the two rates.

    Reported alongside the curve, never instead of it.
    """
    if not curve:
        return {}
    return max(curve, key=lambda r: r["balanced"])
