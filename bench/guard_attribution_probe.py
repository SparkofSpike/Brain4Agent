"""The cost of each proximity guard, measured by switching it off.

D-042 justifies three guards with three regressions. Two of them are constants, so
their cost can be re-measured; the third is a code path, recorded in D-042 with
the number it produced. This probe covers the two:

    length floor off   -> PrecisionMemBench pass count should fall
    term floor off     -> LoCoMo hit@1 should rise (the guard has a cost)
                          real store hit@1 should fall (this is why it exists)

Every row goes through `MemoryStore`, so what is measured is what ships.

    python -m bench.guard_attribution_probe [--db <copy of a store>]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from bench.retrieval import isolated_bonus
from brain4agent.store import MemoryStore

# The child has to patch the constant *before* the harness builds its store, and
# `isolated_bonus` in this process cannot reach a child. Passing the value as
# argv keeps the child free of string formatting.
PMB_CHILD = (
    "import sys\n"
    "import brain4agent.store as sm\n"
    "sm.PROXIMITY_MIN_DOC_TOKENS = int(sys.argv[1])\n"
    "from bench.pmb.run import main\n"
    "main([])\n"
)


def pmb_pass_with(min_doc_tokens: int) -> str:
    out = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", PMB_CHILD, str(min_doc_tokens)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ),
    )
    for line in out.stdout.splitlines():
        if line.strip().startswith("pass"):
            # The line is `pass   : 74/77 = 0.9610`; the ratio is the number that
            # belongs in a column.
            return line.split(":", 1)[1].strip().split(" ")[0]
    tail = out.stderr.strip().splitlines()
    return f"? ({tail[-1][:40] if tail else 'no output'})"


def locomo_hit1() -> float:
    from bench.locomo.run import load, session_text

    documents, queries = load()
    store = MemoryStore(Path(tempfile.mkdtemp()) / "lc.db")
    for doc in documents:
        sample = doc["id"].rsplit("_session_", 1)[0]
        store.add(session_text(doc), session_id=doc["id"], tags=[f"sample:{sample}"])
    h1 = n = 0
    for q in queries:
        gold = set(q["gold_ids"])
        if not gold:
            continue
        n += 1
        got = [
            e.session_id
            for e in store.search(q["query"], top_k=10, tags=[f"sample:{q['user_id']}"]).evidence
        ]
        first = next((i for i, sid in enumerate(got, 1) if sid in gold), None)
        h1 += first == 1
    store.close()
    return h1 / (n or 1)


def real_hit1(db: Path) -> float:
    from bench.real.probe import build_queries, snapshot

    tmp = Path(tempfile.mkdtemp()) / "real.db"
    snapshot(db, tmp)
    store = MemoryStore(tmp)
    store.search("warmup", top_k=1)  # pay the rebuild outside the loop
    h1 = n = 0
    for item in build_queries(tmp, 300, 20261110, 40, 500):
        n += 1
        got = [e.session_id for e in store.search(str(item["q"]), top_k=10).evidence]
        h1 += bool(got) and got[0] == item["sid"]
    store.close()
    tmp.unlink(missing_ok=True)
    return h1 / (n or 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=None, help="a copy of a real store")
    args = ap.parse_args(argv)

    print(f"{'guard':<24}{'benchmark':<18}{'shipped':>12}{'guard off':>12}")
    print("-" * 66)

    # 0 keeps the built-in three-window rule; 1 token lets every document qualify.
    print(f"{'length floor':<24}{'PMB pass':<18}{pmb_pass_with(0):>12}{pmb_pass_with(1):>12}")

    # `proximity=None` keeps the shipped bonus on: this row isolates the term
    # floor, and switching the whole signal off would compare two baselines.
    with isolated_bonus(proximity=None):
        lc_on = locomo_hit1()
    with isolated_bonus(proximity=None, min_terms=0):
        lc_off = locomo_hit1()
    print(f"{'term floor':<24}{'LoCoMo hit@1':<18}{lc_on:>12.4f}{lc_off:>12.4f}")

    if args.db and args.db.exists():
        with isolated_bonus(proximity=None):
            real_on = real_hit1(args.db)
        with isolated_bonus(proximity=None, min_terms=0):
            real_off = real_hit1(args.db)
        print(f"{'term floor':<24}{'real store hit@1':<18}{real_on:>12.4f}{real_off:>12.4f}")
    else:
        print(f"{'term floor':<24}{'real store hit@1':<18}{'(skipped: --db)':>24}")

    print()
    print("The ASCII filter on the query is a code path, not a constant, so it has")
    print("no switch here. Its cost is in D-042: Chinese hit@5 1.0000 -> 0.9750")
    print("without it, on a question where every candidate scored the same")
    print("proximity -- the factor separated nothing while perturbing the fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
