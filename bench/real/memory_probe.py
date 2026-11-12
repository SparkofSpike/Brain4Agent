"""What the sequence cache costs when the proximity path is on.

`_seqs` holds one list of normalised tokens per document. The strings are not
copies -- `Counter(seq)` already kept those objects, so the extra cost is the
list slots and nothing else -- but "and nothing else" should be a measurement,
not a claim.

Reports the token total, the list overhead, and the resident delta measured by
`tracemalloc` across the two configurations.

    python -m bench.real.memory_probe --db <copy of a store>
"""

from __future__ import annotations

import argparse
import tempfile
import tracemalloc
from pathlib import Path

import brain4agent.store as sm
from bench.real.probe import snapshot
from brain4agent.store import MemoryStore


def build(db: Path, bonus: float) -> tuple[int, int, float]:
    sm.PROXIMITY_BONUS = bonus
    store = MemoryStore(db)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    store.refresh()  # builds the index under the current constants
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    index = store._index  # noqa: SLF001
    tokens = sum(len(s) for s in index._seqs)
    seq_lists = len(index._seqs)
    store.close()
    return tokens, seq_lists, (peak - base) / 1e6


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    # Required, not defaulted: this repository is public and a local store path
    # does not belong in it (AGENTS.md: no hard-coded local paths). The store is
    # copied anyway, because measuring a build writes nothing but reading a live
    # file under someone else's writes is a way to measure noise.
    ap.add_argument("--db", type=Path, required=True, help="a copy of a real store")
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"no store at {args.db}")
        return 2

    tmp = Path(tempfile.mkdtemp()) / "mem.db"
    snapshot(args.db, tmp)

    print(f"{'bonus':>6} {'seq docs':>9} {'seq tokens':>11} {'index peak MB':>14}")
    for bonus in (0.0, 1.5):
        tokens, docs, peak = build(tmp, bonus)
        print(f"{bonus:>6.1f} {docs:>9,} {tokens:>11,} {peak:>14.2f}")
    tmp.unlink(missing_ok=True)
    print()
    print("The delta between the rows is the sequence cache: a list of references")
    print("per document, sharing the string objects the term Counters already hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
