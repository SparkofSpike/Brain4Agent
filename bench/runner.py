"""Run a retriever over a benchmark and report numbers.

The runner owns the loop; the retriever only has to answer "given these
memories and this query, what do you rank highest". Keeping that boundary means
a new system can be scored by adding one adapter, not by editing the harness.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from bench.loader import MemoryItem, Task
from bench.metrics import (
    RetrievalScore,
    abstention_curve,
    aggregate,
    best_balanced,
    score_one,
)
from bench.retrieval import BM25

# A retriever takes (memories, query_text, top_k) and returns ranked (id, score).
Retriever = Callable[[list[MemoryItem], str, int], list[tuple[str, float]]]


@dataclass(slots=True)
class RunResult:
    name: str
    scores: list[RetrievalScore]
    seconds: float
    errors: list[str]
    top_scores: list[tuple[str, float]] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, object]:
        out = aggregate(self.scores)
        out["seconds"] = round(self.seconds, 2)
        out["errors"] = len(self.errors)
        return out

    def abstention_report(self) -> dict[str, object]:
        """Abstention capability, kept out of the hit@k summary.

        Mixing them would let a system that always abstains look good on one
        number while being useless on the other.
        """
        by_abs = {qid: s for qid, s in self.top_scores}
        is_abs = {s.query_id: s.is_abstention for s in self.scores}

        answerable = [s for q, s in by_abs.items() if not is_abs.get(q, False)]
        abstain = [s for q, s in by_abs.items() if is_abs.get(q, False)]

        curve = abstention_curve(answerable_top_scores=answerable, abstention_top_scores=abstain)
        return {
            "abstention_questions": len(abstain),
            "answerable_questions": len(answerable),
            "curve": curve,
            "best_balanced": best_balanced(curve),
        }


def bm25_retriever(top_k: int = 10) -> Retriever:
    """A retriever that rebuilds its index per task.

    Rebuilding is correct here: LongMemEval tasks have independent histories,
    and sharing an index would let one question's answer leak into another's
    ranking.
    """

    def _run(memories: list[MemoryItem], query: str, k: int) -> list[tuple[str, float]]:
        index = BM25()
        index.index([(m.id, m.content) for m in memories])
        return index.search(query, top_k=max(k, top_k))

    return _run


def collapse_to_sessions(ranked_item_ids: list[str], memories: list[MemoryItem]) -> list[str]:
    """Turn a ranked item list into a ranked session list.

    First occurrence wins. Without this, one long session could fill every slot
    and a system that found exactly one of five gold sessions would look better
    than it is.
    """
    by_id = {m.id: m.session_id for m in memories}
    ordered: list[str] = []
    seen: set[str] = set()
    for mid in ranked_item_ids:
        sid = by_id.get(mid)
        if sid is None or sid in seen:
            continue
        seen.add(sid)
        ordered.append(sid)
    return ordered


def run(
    tasks: list[Task],
    retriever: Retriever,
    *,
    name: str = "unnamed",
    top_k: int = 10,
    progress_every: int = 50,
) -> RunResult:
    started = time.perf_counter()
    scores: list[RetrievalScore] = []
    errors: list[str] = []
    top_scores: list[tuple[str, float]] = []

    for i, task in enumerate(tasks, start=1):
        try:
            ranked = retriever(task.memories, task.query.text, top_k)
            ranked_ids = [mid for mid, _ in ranked]
            ranked_sessions = collapse_to_sessions(ranked_ids, task.memories)[:top_k]
            top_scores.append((task.query.id, ranked[0][1] if ranked else 0.0))
            scores.append(
                score_one(
                    query_id=task.query.id,
                    question_type=task.query.question_type,
                    is_abstention=task.query.is_abstention,
                    gold_session_ids=task.query.gold_session_ids,
                    ranked_session_ids=ranked_sessions,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- one bad task must not sink the run
            errors.append(f"{task.query.id}: {type(exc).__name__}: {exc}")

        if progress_every and i % progress_every == 0:
            print(f"  [{name}] {i}/{len(tasks)}", flush=True)

    return RunResult(
        name=name,
        scores=scores,
        seconds=time.perf_counter() - started,
        errors=errors,
        top_scores=top_scores,
    )
