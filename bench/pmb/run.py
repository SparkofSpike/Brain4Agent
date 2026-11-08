"""Run PrecisionMemBench against this store and report the pass rate.

    python -m bench.pmb.run

The published baseline to beat is 0.4675 (open-knowledge-format), the highest of
the eight systems currently recorded in AMB's external_results.json. zep is
0.1169, supermemory 0.2727, gbrain 0.4416.

Two numbers matter and they are not the same:

- ``pass`` — every case, including the ones whose assertions only touch tiers
  the harness derives locally. A store that returns nothing still scores here.
- ``active`` — only cases whose relevantBeliefs tier is non-empty, i.e. the
  cases that actually require query-dependent retrieval. This is the column
  upstream calls "Active passes", and it is the honest one.

Both are reported because publishing only the first would be the kind of number
this project keeps complaining about.
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

from bench.pmb.harness import belief_to_text, load, score_case
from brain4agent.store import MemoryStore


def main() -> int:
    beliefs, cases = load()
    print(f"corpus: {len(beliefs)} beliefs · cases: {len(cases)}")

    with tempfile.TemporaryDirectory() as td, MemoryStore(Path(td) / "pmb.db") as store:
        with store.batch():
            for b in beliefs:
                # One session per belief; the session id is the belief id, so a
                # retrieved session maps back with no resolver heuristics.
                store.add(belief_to_text(b), session_id=b["_id"])

        passed = active_passed = 0
        active_total = 0
        failures: list[tuple[str, list[str]]] = []
        by_cat: Counter[tuple[str, str]] = Counter()

        for case in cases:
            k = (case.get("budget") or {}).get("maxBeliefs", 20)
            # Upstream skips search entirely when the belief budget is zero, so a
            # budget-0 case is asserting that nothing is returned.
            if k <= 0 or not case["query"].strip():
                retrieved: list[str] = []
            else:
                result = store.search(case["query"], top_k=k)
                retrieved = [e.session_id for e in result.evidence]

            ok, why, is_active = score_case(case, beliefs, retrieved)
            if is_active:
                active_total += 1
                if ok:
                    active_passed += 1
            if ok:
                passed += 1
            else:
                failures.append((case["caseId"], why))
            by_cat[(case["category"], "pass" if ok else "fail")] += 1

    n = len(cases)
    print()
    print(f"pass   : {passed}/{n} = {passed / n:.4f}")
    print(f"active : {active_passed}/{active_total} = {active_passed / active_total:.4f}")
    print()
    print("published comparison (AMB external_results, all systems, any model):")
    for name, score in [
        ("open-knowledge-format", 0.4675),
        ("gbrain", 0.4416),
        ("supermemory", 0.2727),
        ("zep", 0.1169),
        ("agentmemory", 0.0909),
    ]:
        print(f"  {name:<24} {score:.4f}")
    print()

    print("by category:")
    cats = sorted({c for c, _ in by_cat})
    for cat in cats:
        p = by_cat[(cat, "pass")]
        f = by_cat[(cat, "fail")]
        print(f"  {cat:<44} {p}/{p + f}")

    if failures:
        print()
        print(f"failures ({len(failures)}):")
        for cid, why in failures[:12]:
            print(f"  {cid}")
            for w in why[:3]:
                print(f"      {w}")
        if len(failures) > 12:
            print(f"  ... and {len(failures) - 12} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
