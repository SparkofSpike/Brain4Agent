"""Why LoCoMo hit@1 is 0.68 while hit@10 is 0.95: a failure-shape probe.

`run.py` reports one number. This probe reports the distribution behind it, so
a candidate change can be aimed at the failure that actually dominates instead
of the one that is easiest to imagine.

It answers five questions, each of which would imply a different fix:

1. **Where does the gold session land when it is not first?** Second place is a
   near-tie, tenth is a different problem.
2. **How much of the query does the gold session contain?** Low coverage is a
   vocabulary gap (the words are not in the memory). High coverage with a bad
   rank is a scoring/length problem, and no retriever change will fix it.
3. **Is the gold session longer or shorter than its distractors?** BM25's `b`
   penalises long documents. A session-level index of whole conversations makes
   every document long, and one that is 4x longer than its peers gets punished
   for containing more evidence.
4. **What does the winner have that the gold does not?** Term overlap with the
   query, and how many query terms the winner shares.
5. **Do the failures cluster by category?** multi-hop at 0.427 is a different
   failure from single-hop at 0.555.

    python -m bench.locomo.diagnose

Read-only: no writes to the store, no model calls.
"""

from __future__ import annotations

import statistics
import tempfile
from collections import Counter
from pathlib import Path

from bench.locomo.run import load, session_text
from bench.retrieval import isolated_bonus
from brain4agent.store import MemoryStore


def main() -> int:
    documents, queries = load()
    lens_by_session = {d["id"]: len(session_text(d)) for d in documents}
    print(f"corpus: {len(documents)} sessions")
    print(
        "session length chars: "
        f"min {min(lens_by_session.values())}  "
        f"median {statistics.median(lens_by_session.values()):.0f}  "
        f"max {max(lens_by_session.values())}"
    )
    print()

    ranks: Counter[int] = Counter()
    cov_at_rank1: list[float] = []
    cov_missed: list[float] = []
    len_ratio: list[float] = []
    winner_cov: list[float] = []
    missed_examples: list[dict] = []
    cat_rank: dict[str, list[int]] = {}

    with (
        tempfile.TemporaryDirectory() as td,
        # The histogram in D-042's Context section describes the state *before*
        # this change. With the shipped bonus live, every rank below reflects a
        # ranking the histogram was not taken from.
        isolated_bonus(proximity=0.0),
        MemoryStore(Path(td) / "locomo.db") as store,
    ):
        for doc in documents:
            sample = doc["id"].rsplit("_session_", 1)[0]
            store.add(session_text(doc), session_id=doc["id"], tags=[f"sample:{sample}"])

        for q in queries:
            gold = set(q["gold_ids"])
            if not gold:
                continue
            result = store.search(q["query"], top_k=10, tags=[f"sample:{q['user_id']}"])
            got = [e.session_id for e in result.evidence]
            first = next((i for i, sid in enumerate(got, 1) if sid in gold), None)
            cat = q["meta"]["category"]
            cat_rank.setdefault(cat, []).append(first or 99)
            if first is not None:
                ranks[first] += 1

            # Coverage of the gold session's own text by the query's terms.
            gold_sid = next(iter(gold))
            gcov = next((e for e in result.evidence if e.session_id in gold), None)
            if gcov is not None:
                (cov_at_rank1 if first == 1 else cov_missed).append(gcov.coverage)
            if first != 1 and got:
                top = result.evidence[0]
                winner_cov.append(top.coverage)
                if gcov is not None:
                    len_ratio.append(
                        lens_by_session.get(top.session_id, 0)
                        / max(1, lens_by_session.get(gold_sid, 1))
                    )
                if len(missed_examples) < 12:
                    missed_examples.append(
                        {
                            "cat": cat,
                            "query": q["query"],
                            "gold": sorted(gold),
                            "gold_rank": first,
                            "gold_cov": round(gcov.coverage, 2) if gcov else None,
                            "top": [
                                (e.session_id, round(e.score, 2), round(e.coverage, 2))
                                for e in result.evidence[:3]
                            ],
                        }
                    )

    total = sum(ranks.values())
    print(f"answerable: {total}")
    print("gold rank distribution:")
    for r in sorted(ranks):
        label = f"  rank {r}" if r < 10 else "  rank 10+"
        print(f"{label}: {ranks[r]:5d}  {ranks[r] / total:.3f}")
    print()

    def stat(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"{name}: (empty)")
            return
        xs = sorted(xs)
        q1 = xs[len(xs) // 4]
        med = statistics.median(xs)
        q3 = xs[(3 * len(xs)) // 4]
        print(
            f"{name}: n={len(xs)}  p25 {q1:.3f}  median {med:.3f}  "
            f"p75 {q3:.3f}  mean {statistics.fmean(xs):.3f}"
        )

    print("coverage of the gold session:")
    stat("  when ranked 1st", cov_at_rank1)
    stat("  when not 1st  ", cov_missed)
    print()
    print("on failures: winner's coverage, and winner/gold length ratio")
    stat("  winner coverage   ", winner_cov)
    stat("  length ratio (w/g)", len_ratio)
    print()
    print("mean gold rank by category (99 = not in top 10):")
    for cat, rs in sorted(cat_rank.items()):
        print(f"  {cat:<14} mean {statistics.fmean(rs):5.2f}  n={len(rs)}")
    print()

    print("12 failures, first three retrieved:")
    for ex in missed_examples:
        print(f"  [{ex['cat']}] {ex['query']}")
        print(f"      gold {ex['gold']} rank={ex['gold_rank']} cov={ex['gold_cov']}")
        print(f"      got  {ex['top']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
