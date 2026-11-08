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

## Result

```
pass   : 47/77  = 0.6104
active : 27/43  = 0.6279      ← comparable to the published column
```

Published systems on the same dataset (AMB `external_results.json`):

```
this store             0.6104
open-knowledge-format  0.4675
gbrain                 0.4416
supermemory            0.2727
zep                    0.1169
atomicmemory           0.1169
agentmemory            0.0909
```

Two numbers, two meanings. `pass` counts every case, including ones whose
assertions only touch tiers the harness derives from the seed corpus — a store
that returns nothing still passes those. `active` counts only cases needing
query-dependent retrieval, which is the column upstream calls "Active passes".

## How it got there

Every improvement is the same idea in a different place: **the store needs to
know things the text does not say**. Nothing in this table uses a model.

```
configuration                          pass     active
raw BM25, top_k=1                      0.3766   0.4419
+ drop superseded memories             0.3896   0.4419
+ scope tags                           0.5065   0.5116
+ kind whitelist (no open questions)   0.5584   0.5349
+ relation expansion                   0.6104   0.6279
```

The last row is `bench/pmb/run.py` as it stands. The four fields it relies on —
`scope`, `type`, `superseded_by`, `participants` — all come from the fixture's
own seed corpus. The harness supplies them at write time and filters on them at
read time. No guessing about the query.

Per-category now:

```
Alias resolution                14/23
Budget eviction and capacity     5/5
Design boundary cases            3/6
Fuzzy matching and prefix guards 5/7
Ranking stability                2/3
Relation expansion               4/4
Scope disambiguation             6/12
Supersession chain exclusion     2/3
Type routing and open questions  3/6
Persona prelude content          2/4
Counter-signal retrieval         1/2
Cross-user isolation             0/1
Cold start behavior              0/1
```

## What the failure modes are

Out of 30 failures, 33 assertions fail as "missing expected belief" and 17 as
"unexpected belief" — one case can fail both ways.

The dominant remaining pattern is **questions that legitimately need more than
one memory**. 32 cases want exactly one belief, which `top_k=1` serves; 8 want
two, 2 want six, 1 wants seven. Those are the alias-compound and cap-stress
cases: "how should ReactJS components handle exceptions from async calls" needs
both the React entity and the error-handling belief.

Raising `top_k` does not fix this. Measured:

```
top_k=1   0.6104
top_k=2   0.4156
top_k=3   0.3247
top_k=20  0.3247
```

One extra document fails a `shouldOnlyInclude` assertion, and there are far more
of those than there are multi-belief questions. A score-ratio heuristic (keep
hits within 0.5 or 0.7 of the top score) was tried and scored worse on both
axes; results are in the commit history if anyone wants to re-derive it.

So the honest position is: this configuration is tuned for precision, it trades
recall on compound questions, and it wins that trade decisively on this dataset.

## Return-count disclosure

`top_k=1` is a deliberate choice, not an accident. Quoting 0.6104 without saying
the setting would be the kind of benchmark gaming this project exists to
criticise, so: **every number on this page uses `top_k=1` with relation
expansion**. The store's default is `top_k=5`.

## Reproducing

```bash
PYTHONPATH=src:. python -m bench.pmb.run
```

No API key, no network after the fixtures are in place. Temp database, discarded
on exit. Data is upstream `tenurehq/precisionmembench` at commit `b95d6ab`, MIT.

## Caveats worth stating

- **Scope comes from the caller.** The benchmark ships it; a real deployment has
  to know which scope a memory belongs to when it writes it. That is a design
  cost, not something this store infers.
- **Relation expansion is opt-in** and off by default, because it trades
  precision for completeness and only the caller knows which they need.
- **`user:universal` is a scope, not an exemption.** Treating it as
  "visible everywhere" scored 0.5325 against 0.6104 — the universal beliefs carry
  generic words that match almost any question about working style. The
  benchmark's own `_pinned_facts` reads it the same way this store does.
- **This is one benchmark.** It measures retrieval precision. It says nothing
  about conversational recall (LoCoMo, LongMemEval), where a different set of
  tradeoffs applies and this store has not been measured at all.
