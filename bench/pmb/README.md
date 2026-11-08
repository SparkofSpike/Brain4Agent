# PrecisionMemBench on this store

A public retrieval benchmark, run honestly, with the result published whether or
not it flatters the project.

## Why this one

Most memory benchmarks need a generation model and a judge model. This one does
not: the answer *is* the set of belief IDs the store returns, and each case
asserts which IDs must appear and which must not. Noise is a hard failure rather
than an invisible cost. That makes it the only public benchmark a model-free
system can run without renting a model.

It also tests the claim this project actually makes — that a remembered thing
should come back when it matters and stay out when it does not.

## What it costs

Four files, ~70 KB, MIT licensed, fetched from `tenurehq/precisionmembench` at
commit `b95d6ab`:

```
data/beliefs.seed.json      35 beliefs (34 test-user, 1 other-user)
data/retrieval.cases.json   77 cases across 13 categories
```

Scoring is ported from `vectorize-io/agent-memory-benchmark`
(`src/memory_bench/dataset/precisionmembench.py`), which ports upstream
`src/retrieval.external.eval.test.ts`. The port is faithful on the point that
matters: only the `relevantBeliefs` tier depends on the memory system. The other
three tiers are derived locally from the seed corpus, which is why a store that
returns nothing still passes some cases — and why `active` is the number to
quote.

## Result

```
pass   : 29/77  = 0.3766
active : 19/43  = 0.4419      ← comparable to the published column
```

Published systems on the same dataset (AMB `external_results.json`):

```
open-knowledge-format   0.4675
gbrain                  0.4416   ← this store is level with this
supermemory             0.2727
zep                     0.1169
atomicmemory            0.1169
agentmemory             0.0909
```

Two things to say plainly.

**On the overall number this store is fourth of nine** — level with gbrain,
behind open-knowledge-format. It is not the best, and it has not earned a claim
to be.

**The active number is the more interesting one.** 19/43 means that on the cases
which actually require query-dependent retrieval, this store gets 44% fully
right — where "fully right" means every must-include present and not one
forbidden belief among them. A single extra document fails the case.

## Where it fails

```
Alias resolution               13/23
Budget eviction and capacity    5/5
Fuzzy matching and prefix guards 5/7
Ranking stability               2/3
Design boundary cases           2/6
Persona prelude content         1/4
Scope disambiguation            1/12
Type routing and open questions 0/6
Relation expansion              0/4
Supersession chain exclusion    0/3
Cross-user isolation            0/1
Counter-signal retrieval        0/2
Cold start behavior             0/1
```

Failure kinds across all 48 failed cases:

```
missing expected belief          37
unexpected belief                33
forbidden belief surfaced         1
personaPrelude not empty          1
```

The four categories at zero — type routing, relation expansion, supersession,
cross-user isolation — all need something this store does not have: **metadata
on memories**. The benchmark asks "is this belief about the right scope", "has it
been superseded", "is it a fact or an open question". A bag of text cannot answer
any of those, no matter how good the BM25 is.

That is the real finding. It is not that BM25 is weak; alias resolution and fuzzy
matching are respectable. It is that **retrieval precision stops improving once
documents are indistinguishable from each other except by their words**.

## The return-count result

Returning fewer results is dramatically better here, because one extra document
is a failure:

```
top_k   pass              active
1       29/77 = 0.3766   19/43 = 0.4419
2       15/77 = 0.1948    6/43 = 0.1395
3       12/77 = 0.1558    3/43 = 0.0698
5       12/77 = 0.1558    3/43 = 0.0698
20      12/77 = 0.1558    3/43 = 0.0698
```

A note on what this does and does not mean. It does not mean a memory system
should always return one result — that would fail the 37 "missing expected
belief" cases differently and would be useless to an agent that needs context.
It means the benchmark's scoring rewards conservatism, and any system tuned for
it will look better by returning less. Publishing a `top_k=1` number without
saying so would be exactly the kind of benchmark gaming this project exists to
criticise.

## What would move it

In order of expected effect per unit of work:

1. **Scope on memories.** Scope disambiguation is 12 cases and the store fails
   11. Every belief carries `scope: [domain:code]` style tags; the store would
   need a tag column and a filter at search time. This is metadata, not
   modelling — STONE's "store the raw experience" does not forbid labelling it.
2. **Supersession.** Three cases, zero passes. A belief with `superseded_by` must
   not surface. The store has `created_at` but no replacement link.
3. **Type routing.** Six cases, zero passes. Open questions belong in a different
   tier than facts.
4. **Relation expansion.** Four cases. Currently no passes; the store has
   `graph.py` but it is deliberately not wired into retrieval (D-024).

Items 1–3 are all "attach metadata, filter on it". None requires a model. That
is the next piece of work, and this benchmark is how to measure whether it
helped.

## Reproducing

```bash
PYTHONPATH=src:. python -m bench.pmb.run
```

No API key, no network after the fixtures are in place. Temp database, discarded
on exit.
