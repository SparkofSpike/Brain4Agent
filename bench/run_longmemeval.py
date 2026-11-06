"""Score a baseline on LongMemEval.

Usage:
    python -m bench.run_longmemeval                    # oracle, quick sanity
    python -m bench.run_longmemeval --dataset s        # realistic haystack
    python -m bench.run_longmemeval --limit 50         # smoke test
"""

from __future__ import annotations

import argparse
import json

from bench.loader import S_CLEANED, load_longmemeval, summarize
from bench.runner import bm25_retriever, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["oracle", "s"], default="oracle")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    name = S_CLEANED if args.dataset == "s" else "longmemeval_oracle.json"
    print(f"loading {name} ...")
    tasks = load_longmemeval(name, limit=args.limit)

    shape = summarize(tasks)
    print("shape:", json.dumps(shape, ensure_ascii=False))
    if shape["tasks_missing_gold"]:
        print(f"WARNING: {shape['tasks_missing_gold']} tasks have no gold session in haystack")

    print(f"\nscoring BM25 over {len(tasks)} tasks (top_k={args.top_k})")
    result = run(tasks, bm25_retriever(), name="bm25", top_k=args.top_k)
    summary = result.summary

    print("\n=== result ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result.errors:
        print(f"\n{len(result.errors)} errors, first 3:")
        for e in result.errors[:3]:
            print("  ", e)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"\nwritten to {args.json_out}")


if __name__ == "__main__":
    main()
