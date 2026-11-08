"""Prune by weight against prune by age, on a real memory corpus.

The question this answers: when you have to drop memories, does weighting by
retrieval history keep more of what was actually useful than dropping the oldest?

Until now that question had no data. The Chinese benchmark is 44 hand-made
sessions; PrecisionMemBench is 35 synthetic beliefs. Neither contains the one
thing needed to judge a forgetting policy -- **a record of which memories were
ever actually recalled**.

The nmem brain on this machine has exactly that. 2925 fibers, of which 661 carry
a non-zero `frequency` and 2264 were never recalled at all. That split is real
usage, accumulated over months, and it is the ground truth a pruning experiment
needs.

    python -m bench.nmem.prune_probe [--db PATH] [--limit N]

The database is copied to a temp file and never opened for writing. No memory
content leaves the machine and none is committed -- only the numbers.

## Design

1. Load fibers with usable text (summary, falling back to essence).
2. Drop mojibake rows. A share of this corpus is UTF-8 decoded as GBK and reads
   as `涓�` noise; it is unsearchable, so including it would test nothing.
3. Import into a fresh store. Each fiber becomes one memory.
4. Replay the usage history: a fiber recalled `frequency` times is reinforced
   that many times, nothing else is.
5. Prune to the same size two ways -- oldest-first, and lowest-weight-first.
6. Score both survivors on *recall of the lived set*: what fraction of the 661
   genuinely-used memories are still there.

A third baseline is included because it is the honest floor: keep a random
subset of the same size.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

from brain4agent.store import MemoryStore

DEFAULT_DB = os.path.expanduser(r"~\.neuralmemory\brains\default.db")

# Mojibake from UTF-8 read as GBK. These blocks appear when the decoder was wrong,
# and their presence marks a row whose text cannot match a real query.
_MOJIBAKE = re.compile(r"[\u3010-\u3011]?\s*(?:涓|锛|鐨|鏄|鍦|涓|瀹|绋|瀛|鏁|鎴|浣|闇|瑕)")


def is_mojibake(text: str) -> bool:
    return bool(_MOJIBAKE.search(text))


def load_fibers(db_path: str) -> list[dict]:
    tmp = Path(tempfile.gettempdir()) / "nmem_probe.db"
    shutil.copy2(db_path, tmp)
    try:
        conn = sqlite3.connect(tmp)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, summary, essence, frequency, created_at FROM fibers"
        ).fetchall()
        conn.close()
    finally:
        tmp.unlink(missing_ok=True)

    out: list[dict] = []
    for r in rows:
        text = (r["summary"] or "").strip() or (r["essence"] or "").strip()
        if not text or is_mojibake(text):
            continue
        out.append(
            {
                "id": r["id"],
                "text": text,
                "frequency": int(r["frequency"] or 0),
                "created_at": r["created_at"] or "",
            }
        )
    return out


def replay(store: MemoryStore, fibers: list[dict]) -> dict[str, str]:
    """Import and reinforce according to real usage. Returns fiber id -> memory id."""
    mapping: dict[str, str] = {}
    with store.batch():
        for f in fibers:
            mapping[f["id"]] = store.add(f["text"], session_id=f"fiber:{f['id']}", tags=["nmem"])
    for f in fibers:
        for _ in range(min(f["frequency"], 30)):  # cap: 30 reinforcements is weight 0.5+0.9
            store.reinforce(mapping[f["id"]])
    # One decay pass before pruning. Without it every memory still sits at the
    # neutral 0.5 and "prune by weight" deletes nothing, because the floor is
    # exclusive. In real use this is the scheduled pass the caller runs; here it
    # runs once so the two policies have something to compare.
    store.decay(factor=0.9)
    return mapping


def recall_rate(store: MemoryStore, fibers: list[dict], mapping: dict[str, str]) -> float:
    """Fraction of fibers a self-query still finds at rank 1.

    The query is the fiber's own text truncated to its first clause. That is not a
    realistic question, but it is a consistent one: it asks whether a memory is
    still reachable by what it says, which is the property pruning must preserve.
    """
    if not fibers:
        return 0.0
    hits = 0
    for f in fibers:
        mid = mapping[f["id"]]
        query = f["text"].split("\n")[0][:60]
        result = store.search(query, top_k=1, tags=["nmem"])
        if result.evidence and result.evidence[0].memory.id == mid:
            hits += 1
    return hits / len(fibers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit", type=int, default=None, help="cap fibers for a quick run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"no nmem database at {args.db}", file=sys.stderr)
        return 1

    fibers = load_fibers(args.db)
    if args.limit:
        fibers = fibers[: args.limit]
    lived = [f for f in fibers if f["frequency"] > 0]

    print(f"fibers with usable text: {len(fibers)}")
    print(f"  ever recalled: {len(lived)} ({len(lived) / len(fibers):.1%})")
    print(f"  never recalled: {len(fibers) - len(lived)}")
    print()

    target = len(lived)  # prune down to the number of memories that actually mattered
    if target < 10 or target >= len(fibers):
        print("corpus too small or too skewed for this experiment", file=sys.stderr)
        return 1
    print(f"pruning target: {target} of {len(fibers)} memories kept")
    print()

    with tempfile.TemporaryDirectory() as td:
        # Each strategy gets its own copy. They used to share one file and the
        # earlier strategy's deletions shrank the corpus the later ones pruned,
        # which made the last two look catastrophic for the wrong reason.
        base = Path(td) / "base.db"
        with MemoryStore(base) as store:
            mapping = replay(store, fibers)
            baseline_recall = recall_rate(store, fibers, mapping)

        def fresh(name: str) -> MemoryStore:
            copy = Path(td) / f"{name}.db"
            shutil.copy2(base, copy)
            return MemoryStore(copy)

        # ---- strategy A: keep by weight (i.e. by usage) ----
        with fresh("weight") as store:
            # 0.5 * 0.9 = 0.45 for a never-reinforced memory, 0.5+ for one that
            # was: the floor sits between the two populations.
            store.prune_by_weight(floor=0.46)
            kept_ids = {m.id for m in store._by_id.values()}
            kept_fibers = [f for f in fibers if mapping[f["id"]] in kept_ids]
            kept_lived = [f for f in kept_fibers if f["frequency"] > 0]
            weight_recall = recall_rate(store, kept_fibers, mapping)
            weight_lived = len(kept_lived) / len(lived)
            weight_size = len(kept_fibers)

        # ---- strategy B: keep the newest N ----
        with fresh("age") as store:
            by_age = sorted(fibers, key=lambda f: f["created_at"], reverse=True)
            doomed = {mapping[f["id"]] for f in by_age[target:]}
            with store._db:
                store._db.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in doomed])
            store._reload()
            kept_fibers = [f for f in fibers if mapping[f["id"]] not in doomed]
            kept_lived = [f for f in kept_fibers if f["frequency"] > 0]
            age_recall = recall_rate(store, kept_fibers, mapping)
            age_lived = len(kept_lived) / len(lived)
            age_size = len(kept_fibers)

        # ---- baseline C: keep a random subset of the same size ----
        with fresh("random") as store:
            rng = random.Random(args.seed)
            sample = rng.sample(fibers, target)
            keep = {mapping[f["id"]] for f in sample}
            doomed = {mapping[f["id"]] for f in fibers} - keep
            with store._db:
                store._db.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in doomed])
            store._reload()
            kept_fibers = [f for f in fibers if mapping[f["id"]] in keep]
            kept_lived = [f for f in kept_fibers if f["frequency"] > 0]
            rnd_recall = recall_rate(store, kept_fibers, mapping)
            rnd_lived = len(kept_lived) / len(lived)
            rnd_size = len(kept_fibers)

    print(f"{'strategy':<22} {'kept':>6} {'lived kept':>12} {'self-recall':>12}")
    print(f"{'-' * 22} {'-' * 6} {'-' * 12} {'-' * 12}")
    print(f"{'no pruning':<22} {len(fibers):>6} {'100.0%':>12} {baseline_recall:>12.1%}")
    print(
        f"{'by weight (usage)':<22} {weight_size:>6} {weight_lived:>12.1%} {weight_recall:>12.1%}"
    )
    print(f"{'by age (newest first)':<22} {age_size:>6} {age_lived:>12.1%} {age_recall:>12.1%}")
    print(f"{'random':<22} {rnd_size:>6} {rnd_lived:>12.1%} {rnd_recall:>12.1%}")
    print()
    print("lived kept  = share of the ever-recalled memories that survived")
    print("self-recall = share of survivors a self-query still finds at rank 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
