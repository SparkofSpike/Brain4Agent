"""Candidate re-ranks for LoCoMo's near-misses, measured one at a time.

`diagnose.py` established the shape: of 417 misses, 188 put the gold session
second, usually within a few percent of the winner. `attribution.py` is a
debugging tool for reading one case; this is the measurement that decides.

The rule this file follows is D-015's: a candidate is scored through the same
BM25 and the same tokenisers the store ships. Only the re-rank applied to a
fixed candidate window varies, so a difference in the number is a difference in
the signal, not in the retrieval.

Signals tried here, none of which D-041 covered (it rejected BM25L, coverage
multipliers and distinct-term counts):

- **window proximity** -- the fraction of query terms that share a 40-token
  window. Coverage asks whether the words are in the session; this asks whether
  they are in the same passage. A conversation that mentions Caroline every
  turn and a conversation *about* Caroline's conference both have full
  coverage; only one keeps the nouns together.
- **minimum span** -- `1 / (1 + span)` over the shortest token window holding
  every matched query term. The degenerate version of the same idea: no tuning
  constant, and it punishes a session that spreads its mentions out.
- **early position** -- how early the first query term appears, as a fraction of
  the document. A session whose matching passage is in the opening turns is more
  likely to be the session the question came from.

Each is applied as `score * (1 + alpha * signal)`, so alpha = 0 is exactly the
published configuration and the alpha sweep shows whether a gain is a plateau or
a spike. The signal is also compared gold-versus-winner directly: a signal that
does not separate them cannot fix them, and a sweep alone is just fishing.

    python -m bench.locomo.rerank_probe
"""

from __future__ import annotations

import statistics
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from bench.locomo.run import load, session_text
from bench.retrieval import isolated_bonus
from brain4agent.store import MemoryStore
from brain4agent.tokenize import both_tokens, query_tokens, stem_term, window_proximity

TOP_K = 10


# `window_proximity` is imported rather than reimplemented: a probe carrying its
# own copy measures a different function than the one that ships (D-015). The
# rejected candidates below stay here so their rejection has a script behind it.
def doc_tokens(text: str) -> list[str]:
    """Exactly what BM25 keeps for a document: wide tokens, then the stem."""
    return [stem_term(t) for t in both_tokens(text)]


def query_terms(query: str) -> list[str]:
    """Exactly what BM25 matches with: narrow query tokens, then the stem."""
    return [stem_term(t) for t in query_tokens(query)]


def min_span(q_terms: list[str], toks: list[str]) -> float:
    """1 / (1 + shortest window width containing every matched query term)."""
    want = set(dict.fromkeys(q_terms))
    if not want:
        return 0.0
    positions = [(i, t) for i, t in enumerate(toks) if t in want]
    need = len({t for _, t in positions})
    if need < 2:
        return 0.0
    counts: Counter[str] = Counter()
    left = 0
    have = 0
    best = len(toks) + 1
    for right in range(len(positions)):
        t = positions[right][1]
        counts[t] += 1
        if counts[t] == 1:
            have += 1
        while have == need:
            width = positions[right][0] - positions[left][0] + 1
            if width < best:
                best = width
            old = positions[left][1]
            counts[old] -= 1
            if not counts[old]:
                del counts[old]
                have -= 1
            left += 1
    return 0.0 if best > len(toks) else 1.0 / (1.0 + best)


def early_position(q_terms: list[str], toks: list[str]) -> float:
    """1 - (first matching token's index / document length)."""
    if not toks:
        return 0.0
    want = set(q_terms)
    for i, t in enumerate(toks):
        if t in want:
            return 1.0 - i / len(toks)
    return 0.0


SIGNALS: dict[str, Callable[[list[str], list[str]], float]] = {
    "window40": lambda q, t: window_proximity(q, t, 40),
    "window100": lambda q, t: window_proximity(q, t, 100),
    "min-span": min_span,
    "early-position": early_position,
}


def score(queries: list[dict], retrieved: dict[str, list[str]]) -> dict[str, float]:
    hit1 = hit5 = hit10 = mrr = 0.0
    n = 0
    for q in queries:
        gold = set(q["gold_ids"])
        if not gold:
            continue
        n += 1
        first = next(
            (i for i, sid in enumerate(retrieved.get(q["id"], []), 1) if sid in gold), None
        )
        hit1 += first == 1
        hit5 += bool(first and first <= 5)
        hit10 += bool(first and first <= 10)
        mrr += (1.0 / first) if first else 0.0
    n = n or 1
    return {"hit@1": hit1 / n, "hit@5": hit5 / n, "hit@10": hit10 / n, "mrr": mrr / n}


def report(label: str, row: dict[str, float]) -> None:
    print(
        f"{label:34s} hit@1 {row['hit@1']:.4f}  hit@5 {row['hit@5']:.4f}  "
        f"hit@10 {row['hit@10']:.4f}  MRR {row['mrr']:.4f}"
    )


def main() -> int:
    documents, queries = load()
    texts = {d["id"]: session_text(d) for d in documents}
    toks_by_session = {sid: doc_tokens(t) for sid, t in texts.items()}
    print(f"corpus: {len(documents)} sessions  queries: {len(queries)}")
    print()

    baseline: dict[str, list[str]] = {}
    deep: dict[str, list[tuple[str, float, tuple[float, ...]]]] = {}

    # Isolate each signal against the pre-D-042 baseline. Without this the store
    # hands back already-boosted scores and every row here is a delta on itself.
    with (
        tempfile.TemporaryDirectory() as td,
        isolated_bonus(proximity=0.0),
        MemoryStore(Path(td) / "locomo.db") as store,
    ):
        for doc in documents:
            sample = doc["id"].rsplit("_session_", 1)[0]
            store.add(texts[doc["id"]], session_id=doc["id"], tags=[f"sample:{sample}"])

        for q in queries:
            if not q["gold_ids"]:
                continue
            qt = query_terms(q["query"])
            result = store.search(q["query"], top_k=TOP_K, tags=[f"sample:{q['user_id']}"])
            baseline[q["id"]] = [e.session_id for e in result.evidence]
            rows = []
            for e in result.evidence:
                toks = toks_by_session[e.session_id]
                sig = tuple(fn(qt, toks) for fn in SIGNALS.values())
                rows.append((e.session_id, e.score, sig))
            deep[q["id"]] = rows

    report("baseline (published)", score(queries, baseline))
    print()

    for si, name in enumerate(SIGNALS):
        for alpha in (0.2, 0.5, 1.0, 2.0):
            reranked: dict[str, list[str]] = {}
            for qid, rows in deep.items():
                ordered = sorted(rows, key=lambda r: -(r[1] * (1.0 + alpha * r[2][si])))
                reranked[qid] = [r[0] for r in ordered]
            report(f"  {name} alpha={alpha}", score(queries, reranked))
        print()

    print("gold versus winner on the misses, per signal (gold higher means the signal helps):")
    for si, name in enumerate(SIGNALS):
        diffs: list[float] = []
        higher = 0
        for q in queries:
            gold = set(q["gold_ids"])
            rows = deep.get(q["id"], [])
            if not gold or not rows:
                continue
            if rows[0][0] in gold:
                continue
            g = next((r for r in rows if r[0] in gold), None)
            if g is None:
                continue
            diffs.append(g[2][si] - rows[0][2][si])
            higher += g[2][si] > rows[0][2][si]
        if not diffs:
            continue
        print(
            f"  {name:16s} n={len(diffs):4d}  median {statistics.median(diffs):+.3f}  "
            f"gold higher in {higher} ({higher / len(diffs):.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
