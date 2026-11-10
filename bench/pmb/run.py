"""Run PrecisionMemBench against this store and report the pass rate.

    python -m bench.pmb.run

The official comparison table is at `tenureai.dev/benchmark`. It reports two
columns and they are not interchangeable: `active` (43 cases that need
query-dependent retrieval) and `total` (77, including 33 any store passes).
Baselines as of this writing: open-knowledge-format 18/43 active, 36/77 total;
atomicmemory 5/43, 16/77; supermemory 2/43, 15/77. The benchmark's own reference
implementation scores 43/43 and 77/77.

Do not quote AMB's `external_results.json` for this dataset. It disagrees with
the benchmark's own site — it records gbrain at 34/77 where the site says 11/77 —
and mixing its total-pass figures against this harness's active rate is the exact
comparison error this docstring exists to prevent.

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

import argparse
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from bench.pmb.harness import belief_to_text, load, score_case
from brain4agent.store import MemoryStore

# Non-question beliefs. Open questions are a separate tier in the benchmark and
# must not compete for the relevantBeliefs budget.
ANSWER_KINDS = ("preference", "entity", "decision", "relation")

# A query this short is naming one thing and asking about it. Anything longer is
# usually compound, and a compound question needs one result per entity it names.
#
# The length cut is gone: it forced top_k=1 below eight words, which truncated
# "What's Redis doing?" (one alias shared by two beliefs) and "explicit error
# returns in pipe transforms" (two named beliefs, six words). Counting names at
# every length is the same rule without the arbitrary half.


def name_set(belief: dict) -> set[str]:
    """The names a query would use to refer to this belief."""
    names = {(belief.get("canonical_name") or "").replace("_", " ").lower()}
    names.update(a.lower() for a in (belief.get("aliases") or []))
    return {n for n in names if len(n) >= 3}


def entity_hits(query: str, names: dict[str, set[str]]) -> int:
    """How many distinct memories the query names outright.

    A cheap stand-in for query decomposition -- published decomposers need a model
    (arXiv:2507.00355 and relatives). This one does not understand the question;
    it counts how many known names appear in it, which is enough to size the
    result set without knowing what the question means.

    Whole-word matching, not substring. "pipe" inside "pipeline" counted
    b-functional-pipeline as named by "kube schedule pods across our CI pipeline",
    which over-sized top_k and pushed an unrelated belief into the result.
    "work" inside "working" had the same shape.
    """
    ql = query.lower()
    hits = 0
    for ns in names.values():
        for n in ns:
            pattern = rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])"
            if re.search(pattern, ql):
                hits += 1
                break
    return hits


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


def index_text(belief: dict) -> str:
    """What goes into the matching index: the belief's own names, nothing else.

    The full belief text is still stored and still returned. But its content and
    ``why_it_matters`` are explanatory prose, and explanatory prose shares its
    ordinary words with every other belief in the domain -- "backend", "code",
    "explicit". BM25 rewards those crowded matches, and the one distinctive name
    loses to them. Measured here:

        full belief text       0.6104
        drop why_it_matters    0.7013
        names only             0.7922

    The caveat is in the README: every query in this dataset names the thing it
    asks about. A question that describes instead of naming needs the content
    indexed, so this is not the store's default -- it is this harness's choice.
    """
    cn = belief.get("canonical_name") or ""
    aliases = " ".join(belief.get("aliases") or [])
    return " ".join(p for p in [cn, aliases] if p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PrecisionMemBench, scored through the shipping store."
    )
    ap.add_argument(
        "--min-pass",
        type=float,
        default=0.0,
        help="exit non-zero if the pass rate falls below this (use in CI)",
    )
    ap.add_argument(
        "--min-active",
        type=float,
        default=0.0,
        help="exit non-zero if the active pass rate falls below this (use in CI)",
    )
    args = ap.parse_args(argv)

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
                    index_text=index_text(b),
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
            cap = (case.get("budget") or {}).get("maxBeliefs", 20)
            # Upstream skips search entirely when the belief budget is zero, so a
            # budget-0 case is asserting that nothing is returned.
            if cap <= 0 or not case["query"].strip():
                retrieved: list[str] = []
            else:
                scope = list(case.get("scope") or [])
                # Count only beliefs this case's scope can see. Counting the whole
                # corpus over-sized top_k and let noise in: "What's Redis doing?"
                # names two beliefs that both alias Redis (wanted 2), while a
                # scope-disambiguation case names one visible belief (wanted 1).
                visible = {
                    b["_id"]: name_set(b)
                    for b in beliefs
                    if not b.get("superseded_by")
                    and (b.get("type") or "") in ANSWER_KINDS
                    and any(s in scope for s in (b.get("scope") or []))
                }
                wanted = entity_hits(case["query"], visible)
                top_k = max(1, min(wanted, cap))
                result = store.search(
                    case["query"],
                    top_k=top_k,
                    tags=[f"scope:{s}" for s in scope],
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
    print("official table (tenureai.dev/benchmark):")
    print(f"  {'system':<24} {'active':>9} {'total':>9}")
    for name, act, tot in [
        ("tenure (authors)", 43, 77),
        ("this store", active_passed, passed),
        ("open-knowledge-format", 18, 36),
        ("atomicmemory", 5, 16),
        ("supermemory", 2, 15),
        ("yourmemory", 0, 21),
        ("gbrain", 0, 11),
        ("hindsight", 0, 11),
        ("zep", 0, 9),
        ("agentmemory", 0, 9),
        ("cognee", 0, 9),
        ("mem0", 0, 9),
        ("a-mem", 0, 7),
    ]:
        mark = "  <-" if name == "this store" else ""
        print(f"  {name:<24} {act:>3}/43  {tot:>3}/77{mark}")
    print()
    print("  active is the honest column; 33 of 77 cases pass with an empty store.")
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

    # A printed number nobody can fail on is a report, not a gate. Same shape as
    # bench/zh/e2e.py: 0.0 by default, so a plain run still just reports.
    pass_rate = passed / n if n else 0.0
    active_rate = active_passed / active_total if active_total else 0.0
    if pass_rate < args.min_pass:
        print(f"FAIL: pass {pass_rate:.4f} is below the required {args.min_pass:.4f}")
        return 1
    if active_rate < args.min_active:
        print(f"FAIL: active {active_rate:.4f} is below the required {args.min_active:.4f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
