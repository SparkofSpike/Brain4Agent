"""A/B the proximity path on the real store: quality and cost, same queries.

`bench/real/probe.py` measures leave-one-out keyword retrieval on a copy of the
live store. This runs it twice under the same seed -- once with the bonus off,
once with the shipped value -- and times both, because the LoCoMo fixture is all
long conversations and the real store is not: only 9.8% of its 2910 memories
clear the 120-token floor.

Run against a **copy**. `MemoryStore.search` increments access counters, so even
a read writes.

    python -m bench.real.ab_probe --db <copy of a store>
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import brain4agent.store as sm
from bench.real.probe import build_queries, snapshot
from brain4agent.store import MemoryStore


def run(db: Path, bonus: float, limit: int) -> tuple[dict[str, object], float, float]:
    items = build_queries(db, limit, 20261110, 40, 500)
    sm.PROXIMITY_BONUS = bonus
    store = MemoryStore(db)
    # Warm up: the first search builds the in-memory index, and charging that to
    # the first query divides one rebuild across the average.
    store.search("warmup", top_k=1)
    results = []
    t0 = time.perf_counter()
    for item in items:
        response = store.search(str(item["q"]), top_k=10)
        ranked = [e.session_id for e in response.evidence]
        pos = next((i + 1 for i, s in enumerate(ranked) if s == item["sid"]), None)
        results.append({"q": item["q"], "n": item["n"], "pos": pos})
    elapsed = time.perf_counter() - t0
    store.close()

    def bucket(rows: list[dict[str, object]]) -> dict[str, float | int]:
        total = len(rows) or 1
        return {
            "n": len(rows),
            "hit@1": sum(1 for r in rows if r["pos"] == 1) / total,
            "hit@5": sum(1 for r in rows if r["pos"] and r["pos"] <= 5) / total,
            "mrr": sum(1 / r["pos"] for r in rows if r["pos"]) / total,
        }

    by_len: dict[int, list[dict[str, object]]] = {}
    for row in results:
        by_len.setdefault(int(row["n"]), []).append(row)
    summary = {
        **bucket(results),
        "by_len": {str(k): bucket(v) for k, v in sorted(by_len.items())},
    }
    return summary, elapsed, elapsed / len(items) * 1000


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"no such store: {args.db}")
        return 2

    # Work on a copy: search writes access counters.
    tmp = Path(tempfile.mkdtemp()) / "ab.db"
    snapshot(args.db, tmp)

    print(
        f"{'bonus':>6} {'hit@1':>7} {'hit@5':>7} {'MRR':>7} {'ms/query':>9}  2-word  3-word  4-word"
    )
    for bonus in (0.0, 1.5):
        summary, _, per_query = run(tmp, bonus, args.limit)
        bl = summary["by_len"]
        cells = []
        for k in ("2", "3", "4"):
            cells.append(f"{bl[k]['hit@1']:.3f}" if k in bl else "  -  ")
        print(
            f"{bonus:>6.1f} {summary['hit@1']:>7.4f} {summary['hit@5']:>7.4f} "
            f"{summary['mrr']:>7.4f} {per_query:>8.2f}m  " + "   ".join(cells)
        )
    tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
