"""Window size and strength for the proximity re-rank.

`rerank_probe.py` found the signal: a query whose terms share one 40-token
window outranks a session that mentions them once each, and LoCoMo hit@1 moved
0.6826 -> 0.7074. This probe sweeps the two constants that define the signal so
the shipped value is a point on a plateau rather than a peak.

Grid: window in {20, 30, 40, 60, 80} tokens, alpha in {0.25, 0.5, 1, 1.5, 2, 3},
applied as `score * (1 + alpha * proximity)`. Reproduces `bench.locomo.run`'s
baseline as the alpha=0 row, so a run that disagrees with the published number
is a bug in the probe.

**The grid re-ranks a top-10 window per query, so every cell understates what
the store does with the same alpha.** The store re-ranks all 272 candidates
before folding. For window 40 / alpha 1.5 the grid reads 0.7080 and the store
reads 0.7100 -- 0.0020 of signal the window throws away. That is fine for
choosing a constant (the plateau is what matters, and the shape is unaffected)
and wrong for pricing a configuration: use
`bench/locomo/floor_attribution_probe.py`, which goes through the store, when
the number has to be the shipped one.

A second caveat, from getting this wrong once: do not subtract two numbers
measured differently. `0.7080 - 0.7054` looks like the cost of the five-term
floor, but one side came from this window and the other from the store. Measured
on one path the floor costs 0.0046.

    python -m bench.locomo.proximity_sweep
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bench.locomo.rerank_probe import doc_tokens, query_terms, report, score, window_proximity
from bench.locomo.run import load, session_text
from bench.retrieval import isolated_bonus
from brain4agent.store import MemoryStore

TOP_K = 10
WINDOWS = (20, 30, 40, 60, 80)
ALPHAS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def main() -> int:
    documents, queries = load()
    texts = {d["id"]: session_text(d) for d in documents}
    toks_by_session = {sid: doc_tokens(t) for sid, t in texts.items()}
    print(f"corpus: {len(documents)} sessions  queries: {len(queries)}")
    print()

    baseline: dict[str, list[str]] = {}
    # qid -> list of (sid, score, {window: proximity})
    deep: dict[str, list[tuple[str, float, dict[int, float]]]] = {}

    with (
        tempfile.TemporaryDirectory() as td,
        # The store ships this re-rank now; isolate it or the alpha=0 row stops
        # rewriting the published baseline it claims to reproduce.
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
                prox = {w: window_proximity(qt, toks, w) for w in WINDOWS}
                rows.append((e.session_id, e.score, prox))
            deep[q["id"]] = rows

    base = score(queries, baseline)
    report("baseline (alpha=0)", base)
    print()

    print(f"{'window':>7s}  " + "  ".join(f"a={a:<4g}" for a in ALPHAS))
    best = (0.0, 0, 0.0)
    for w in WINDOWS:
        cells = []
        for a in ALPHAS:
            reranked: dict[str, list[str]] = {}
            for qid, rows in deep.items():
                ordered = sorted(rows, key=lambda r: -(r[1] * (1.0 + a * r[2][w])))
                reranked[qid] = [r[0] for r in ordered]
            r = score(queries, reranked)
            cells.append(r["hit@1"])
            if r["hit@1"] > best[0]:
                best = (r["hit@1"], w, a)
        print(f"{w:>7d}  " + "  ".join(f"{c:.4f} " for c in cells))

    print()
    print(f"best hit@1 {best[0]:.4f} at window={best[1]} alpha={best[2]:g}")
    w, a = best[1], best[2]
    reranked = {}
    for qid, rows in deep.items():
        ordered = sorted(rows, key=lambda r: -(r[1] * (1.0 + a * r[2][w])))
        reranked[qid] = [r[0] for r in ordered]
    report(f"full row (w={w}, a={a:g})", score(queries, reranked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
