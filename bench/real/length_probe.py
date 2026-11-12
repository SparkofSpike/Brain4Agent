"""Does the proximity path cost anything on a real store?

LoCoMo is 272 whole conversations, each ~500-1100 index tokens, so every session
clears the "three windows" floor and the query pays for all of them. A working
agent's store looks the other way: thousands of short turns, most of them far
below 120 tokens. If that is true here, the benchmark's ~21 ms per query (D-042;
an earlier draft said 49 ms from a run that overlapped other work) is a property
of a fixture made entirely of long documents, not of how the store is used.

Read-only across two databases: the shipped real store, and the Chinese fixture.

    python -m bench.real.length_probe
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
from pathlib import Path

FLOOR = 120


def report(label: str, lengths: list[int]) -> None:
    if not lengths:
        print(f"{label}: empty")
        return
    lengths = sorted(lengths)
    over = sum(1 for n in lengths if n >= FLOOR)
    print(
        f"{label}: n={len(lengths)}  median {statistics.median(lengths):.0f}  "
        f"p90 {lengths[int(len(lengths) * 0.9)]}  max {lengths[-1]}  "
        f">= {FLOOR} tokens: {over} ({over / len(lengths):.1%})"
    )


def main(argv: list[str] | None = None) -> int:
    from brain4agent.tokenize import both_tokens

    ap = argparse.ArgumentParser()
    # No default: this repository is public and a local store path does not belong
    # in it (AGENTS.md: no hard-coded local paths).
    ap.add_argument("--db", type=Path, default=None, help="a copy of a real store")
    args = ap.parse_args(argv)

    if args.db is not None and args.db.exists():
        db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        rows = db.execute("SELECT content FROM memories").fetchall()
        db.close()
        lengths = [len(both_tokens(r[0])) for r in rows]
        report("real store", lengths)
    elif args.db is not None:
        print(f"no such store: {args.db}")
    else:
        print("real store skipped; pass --db <copy of a store> to include it")

    zh = Path(__file__).parent.parent / "zh" / "zh_eval.json"
    if zh.exists():
        import json

        data = json.loads(zh.read_text("utf-8"))
        task = data["tasks"][0]
        lengths = [
            len(both_tokens("\n".join(t["content"] for t in turns)))
            for turns in task["haystack_sessions"]
        ]
        report("Chinese fixture sessions", lengths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
