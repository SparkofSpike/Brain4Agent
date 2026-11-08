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

Every label used here comes from the fixture: ``scope``, ``type``,
``superseded_by`` and ``participants`` are fields the benchmark's own seed corpus
carries. The harness supplies them at write time and filters on them at read
time. Nothing is inferred from the query, and no model is involved.
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

from bench.pmb.harness import belief_to_text, load, score_case
from brain4agent.store import MemoryStore

# Non-question beliefs. Open questions are a separate tier in the benchmark and
# must not compete for the relevantBeliefs budget.
ANSWER_KINDS = ("preference", "entity", "decision", "relation")


def scope_tags(belief: dict) -> tuple[str, ...]:
    """Tags for one belief -- every scope it claims, universal included.

    ``user:universal`` is a scope, not an exemption. The benchmark's own
    ``_pinned_facts`` reads it the same way: a belief is in scope only when one of
    its scopes appears in the case's scope list. Treating universal as "visible
    everywhere" inflated recall and wrecked precision, because the universal
    beliefs carry generic words (recommendation, pushback) that match almost any
    question about working style.
    """
    return tuple(f"scope:{s}" for s in (belief.get("scope") or []))


def main() -> int:
    beliefs, cases = load()
    print(f"corpus: {len(beliefs)} beliefs · cases: {len(cases)}")

    with tempfile.TemporaryDirectory() as td, MemoryStore(Path(td) / "pmb.db") as store:
        belief_to_memory: dict[str, str] = {}
        with store.batch():
            for b in beliefs:
                belief_to_memory[b["_id"]] = store.add(
                    belief_to_text(b),
                    session_id=b["_id"],
                    tags=scope_tags(b),
                    kind=b.get("type") or "belief",
                    superseded_by=b.get("superseded_by"),
                )
        # Relations point at their participants. Linked after the batch so a
        # participant written later still resolves.
        for b in beliefs:
            parts = b.get("participants") or []
            targets = [belief_to_memory[p] for p in parts if p in belief_to_memory]
            if targets:
                store.link(belief_to_memory[b["_id"]], targets)

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
                result = store.search(
                    case["query"],
                    top_k=1,
                    tags=[f"scope:{s}" for s in (case.get("scope") or [])],
                    kinds=ANSWER_KINDS,
                    expand_refs=True,
                )
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
        for cid, why in failures[:10]:
            print(f"  {cid}")
            for w in why[:3]:
                print(f"      {w}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
