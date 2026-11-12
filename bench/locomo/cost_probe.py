"""Cost of the proximity path on a LoCoMo-sized corpus.

D-041 published a rebuild/query table for the phrase index, and adding a second
position-dependent mechanism makes that table stale. This prints the new one,
plus the token count the sequence cache holds, because that is the number that
scales with a real store.

    python -m bench.locomo.cost_probe
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import brain4agent.store as sm
from bench.locomo.run import load, session_text
from brain4agent.store import MemoryStore

QUERY_SAMPLE = 200


def measure(bonus: float, phrase: float) -> tuple[float, float, float, int]:
    documents, queries = load()
    sm.PROXIMITY_BONUS = bonus
    sm.PHRASE_BONUS = phrase
    store = MemoryStore(Path(tempfile.mkdtemp()) / "cost.db")
    t0 = time.perf_counter()
    for doc in documents:
        sample = doc["id"].rsplit("_session_", 1)[0]
        store.add(session_text(doc), session_id=doc["id"], tags=[f"sample:{sample}"])
    t1 = time.perf_counter()
    store.refresh()
    t2 = time.perf_counter()
    subset = queries[:QUERY_SAMPLE]
    t3 = time.perf_counter()
    for q in subset:
        store.search(q["query"], top_k=10, tags=[f"sample:{q['user_id']}"])
    t4 = time.perf_counter()
    seq_tokens = sum(len(s) for s in store._index._seqs)  # noqa: SLF001 - probe
    store.close()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t4 - t3) / len(subset) * 1000, seq_tokens


def main() -> int:
    print(f"{'configuration':<24}{'insert':>9}{'rebuild':>9}{'query':>8}{'seq tokens':>12}")
    for label, bonus, phrase in (
        ("phrase only (D-041)", 0.0, 0.3),
        ("phrase + proximity", 1.5, 0.3),
    ):
        ins, reb, que, seq = measure(bonus, phrase)
        print(f"{label:<24}{ins:>8.0f}m{reb:>8.0f}m{que:>7.2f}m{seq:>12,}")
    print()
    print("LoCoMo fixture: 272 sessions. Insert is one commit per turn; rebuild is")
    print("what a process pays once, query is the per-lookup cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
