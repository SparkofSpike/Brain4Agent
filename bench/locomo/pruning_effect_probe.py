"""Why the proximity re-rank has no pruning bounds.

D-042 shipped with two, and this probe is the measurement that removed them. It
counts `window_proximity` calls per query in the two shapes the store uses:

- `top_k=5` with no filter: the store asks for `top_k * 10` candidates, so a
  bound would have room to cut the tail.
- `top_k=len(index)` as the tag filter uses: the whole corpus is ranked before
  the filter sees it (D-025), so a bound would have to cut almost everything to
  help.

Both shapes score every candidate: 271.6 calls per query out of 272. Neither
bound fires at alpha 1.5, so both were removed instead of kept as dead code.
Re-run this before reintroducing one -- a bound has to show a lower call count.

    python -m bench.locomo.pruning_effect_probe
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from bench.locomo.run import load, session_text
from brain4agent.store import MemoryStore

# `import brain4agent.tokenize as tk` resolves to the *function* of that name:
# the package re-exports it and the attribute shadows the submodule. The module
# object has to come from sys.modules.
tk = sys.modules["brain4agent.tokenize"]

N = 60


def count_calls(store: MemoryStore, queries: list[dict], *, tags: bool) -> tuple[float, float]:
    original = tk.window_proximity
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    tk.window_proximity = counting
    try:
        total = 0
        for q in queries:
            calls["n"] = 0
            if tags:
                store.search(q["query"], top_k=10, tags=[f"sample:{q['user_id']}"])
            else:
                store.search(q["query"], top_k=5)
            total += calls["n"]
    finally:
        tk.window_proximity = original
    return total / len(queries), len(store._index)  # noqa: SLF001


def main() -> int:
    documents, queries = load()
    # Long English questions only: the five-term floor would skip the rest and
    # the count would be zero for a different reason.
    from brain4agent.tokenize import _phrasable, query_tokens, stem_term

    long_qs = [
        q
        for q in queries
        if len(
            [
                t
                for t in dict.fromkeys(stem_term(x) for x in query_tokens(q["query"]))
                if _phrasable(t)
            ]
        )
        >= 5
    ][:N]
    print(
        f"corpus: {len(documents)} sessions   queries sampled: {len(long_qs)}"
        " (all >= 5 ASCII terms)"
    )

    store = MemoryStore(Path(tempfile.mkdtemp()) / "pe.db")
    for doc in documents:
        sample = doc["id"].rsplit("_session_", 1)[0]
        store.add(session_text(doc), session_id=doc["id"], tags=[f"sample:{sample}"])
    store.refresh()

    plain_calls, docs = count_calls(store, long_qs, tags=False)
    print(f"unfiltered  top_k=5 : {plain_calls:6.1f} proximity calls/query  of {docs} candidates")
    tagged_calls, _ = count_calls(store, long_qs, tags=True)
    print(
        f"tag-filtered top_k=10: {tagged_calls:6.1f} proximity calls/query"
        f" of {docs} candidates (~{docs // 10} of them in the query's sample)"
    )
    print()
    print("Both shapes score every candidate. The tag filter runs on the store side")
    print("after BM25 has already ranked the whole corpus (D-025), so the in-sample")
    print("count is what comes back, not what gets scored -- 90% of this work is on")
    print("sessions the caller asked to exclude. Pushing the filter into the ranker")
    print("would fix it and is out of scope here.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
