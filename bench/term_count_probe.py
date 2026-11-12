"""How many distinct query terms does each benchmark actually have?

`ab_probe` showed the bonus costs the real store 0.027 hit@1 across 2-, 3- and
4-word queries, every bucket. `bound_debug` showed a LoCoMo query with five. If
that is the boundary, the rule is "proximity needs a question, not a keyword
list" -- and it can be stated as a threshold instead of tuned per corpus.

This prints the distribution for LoCoMo, PrecisionMemBench and the real store so
the threshold is set from data on both sides of it.

    python -m bench.term_count_probe
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

from brain4agent.tokenize import _phrasable, query_tokens, stem_term


def count(query: str) -> int:
    terms = [stem_term(t) for t in query_tokens(query)]
    return len([t for t in dict.fromkeys(terms) if _phrasable(t)])


def report(label: str, counts: list[int]) -> None:
    if not counts:
        print(f"{label}: empty")
        return
    counts = sorted(counts)
    print(
        f"{label}: n={len(counts)}  median {statistics.median(counts):.0f}  "
        f"min {counts[0]}  max {counts[-1]}"
    )
    for cut in range(2, 9):
        share = sum(1 for c in counts if c >= cut) / len(counts)
        print(f"    >= {cut} terms: {share:5.1%}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    # No default: this repository is public and a local store path does not belong
    # in it (AGENTS.md: no hard-coded local paths). Pass a copy -- search writes.
    ap.add_argument("--db", type=Path, default=None, help="a copy of a real store")
    args = ap.parse_args(argv)

    locomo = Path(__file__).with_name("locomo") / "data" / "queries.json"
    if locomo.exists():
        qs = json.loads(locomo.read_text("utf-8"))
        report("LoCoMo questions", [count(q["query"]) for q in qs])
        print()

    pmb = Path(__file__).with_name("pmb") / "data" / "retrieval.cases.json"
    if pmb.exists():
        cases = json.loads(pmb.read_text("utf-8"))
        report("PrecisionMemBench queries", [count(c["query"]) for c in cases])
        print()

    # The real store's queries are generated the same way `bench.real.probe`
    # builds them: two to four keywords taken from a memory.
    if args.db is not None and args.db.exists():
        from bench.real.probe import build_queries

        items = build_queries(args.db, 300, 20261110, 40, 500)
        report("real-store keyword queries", [count(str(i["q"])) for i in items])
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        print(f"    (store holds {n} memories)")
    elif args.db is not None:
        print(f"no such store: {args.db}")
    else:
        print("real store skipped; pass --db <copy of a store> to include it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
