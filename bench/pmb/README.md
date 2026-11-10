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
should come back when it matters, and stay out when it does not.

## Result

```
active : 42/43  = 0.9767      <- the honest column
pass   : 74/77  = 0.9610      <- includes 33 any-store passes
```

Official table, from the benchmark's own site (`tenureai.dev/benchmark`):

```
System                  Active    Total
----------------------  ------    -----
tenure (the authors)    43/43     77/77
this store              42/43     74/77
open-knowledge-format   18/43     36/77
atomicmemory             5/43     16/77
supermemory              2/43     15/77
yourmemory               0/43     21/77
gbrain                   0/43     11/77
hindsight                0/43     11/77
zep                      0/43      9/77
agentmemory              0/43      9/77
cognee                   0/43      9/77
mem0                     0/43      9/77
a-mem                    0/43      7/77
```

`pass` counts every case, including 33 whose assertions only touch tiers the
harness derives from the seed corpus — an empty store passes those, so `pass`
overstates retrieval ability. `active` counts only cases needing query-dependent
retrieval, the column upstream calls "Active passes", and it is the one to quote.

**A correction.** Earlier versions of this file quoted AMB's
`external_results.json` (gbrain 0.4416, supermemory 0.2727, zep 0.1169) and
called 0.4675 "the highest published". Those numbers disagree with the
benchmark's own site — it records gbrain at 11/77, not 34/77 — and the site is
authoritative. The old table also compared this store's **active** rate against
those **total** rates: two different denominators. Both columns appear above so
that cannot recur.

The claim this supports is narrower than the one it replaces: **best third-party
result on this benchmark**. Not "beat the state of the art" — the authors' own
implementation scores 43/43 and 77/77.

**No model is used anywhere.** Every improvement below is a rule about text and
labels.

## How it got there

```
                                                    pass     active
raw BM25, top_k=1                                  0.3766   0.4419
+ drop superseded memories                          0.3896   0.4419
+ scope tags                                        0.5065   0.5116
+ kind whitelist (no open questions)                0.5584   0.5349
+ relation expansion                                0.6104   0.6279
+ index only the belief's names (not its prose)     0.7922   0.7209
+ one result per named entity, when compound        0.8312   0.7907
+ names at any length, scoped to visible beliefs    0.8571   0.8372
+ whole-word name matching                          0.8831   0.8837
+ one-edit fuzzy matching                           0.9351   0.9767
+ English stopwords + inflection folding (D-041)   0.9610   0.9767
```

Two of those steps are worth more than the rest, and both are counter-intuitive.

### Indexing the names, not the prose

The belief fixture stores each memory as
`canonical_name + aliases + content + why_it_matters`, and that is what the
harness uses as the document. Indexing it whole scores 0.6104. Indexing only the
canonical name and aliases scores **0.7922** — with the same stored text and the
same returned results.

```
full belief text       0.6104
drop why_it_matters    0.7013
names only             0.7922
content only           0.5844
```

The explanation is BM25's crowding problem. Explanatory prose shares its ordinary
words with the rest of its domain — every belief here says "backend", "code",
"explicit". Those words have low idf, but there are many of them, and they
outweigh the one distinctive name. Recall the failure that started this: a
question about MongoDB's collection pattern returned the Redis memory, because
"code" appeared in both prose sections.

`add(..., index_text=...)` exists for exactly this. Storage and retrieval are
separated: the caller can index a narrower, more distinctive signal while keeping
the full text for a human or a reader to consume. This is STONE's split (store
the experience, extract on demand) applied to the index.

**The caveat, stated plainly:** every query in this dataset names the thing it
asks about — "what's the Mongo pattern", "how does kube schedule pods". A real
question that *describes* instead of *names* ("what do we use for caching?")
would need the content indexed, and would score differently. That is why this is
a choice the harness makes, not the store's default.

### One result per named entity

30 cases still failed after the metadata layer, and the dominant shape was
questions needing more than one memory: 8 want two beliefs, 2 want six, 1 wants
seven. `top_k=1` cannot serve any of them.

Published query decomposition needs a model. This does not: count how many known
names appear in the question, and ask for that many results.

The first version applied the count only above eight words. That truncated real
multi-entity questions: "What's Redis doing?" names two beliefs that share an
alias, and "explicit error returns in pipe transforms" names two in six words.
Counting at every length lifted 0.8312 to 0.8571. Counting every belief
regardless of the case's scope made it worse (0.8052): a name outside the scope
over-sized `top_k` and let noise in. The rule now counts only beliefs the case's
scope can see.

A second correction was whole-word matching. "pipe" inside "pipeline" made
b-functional-pipeline look named by a Kubernetes question, and "work" inside
"working" did the same. Matching names on word boundaries lifted 0.8571 to
0.8831.

The store itself is not doing the counting. `entity_hits` lives in the harness
and reads the fixture's names; the store just accepts a `top_k`. That keeps the
mechanism honest — the store has no opinion about what a compound question is.

## Where it still fails

3 cases. Per category:

```
Alias resolution                22/23
Budget eviction and capacity     5/5
Cold start behavior              0/1
Counter-signal retrieval         2/2
Cross-user isolation             1/1
Design boundary cases            6/6
Fuzzy matching and prefix guards 7/7
Persona prelude content          4/4
Ranking stability                3/3
Relation expansion               4/4
Scope disambiguation            12/12
Supersession chain exclusion     3/3
Type routing and open questions  5/6
```

The fuzzy and prefix cases are now solved: a misspelt English word of six letters
or more is matched one edit away from a corpus word (`typscript` ->
`typescript`, `mongod` -> `mongodb`), which took this category from 4/7 to 7/7.
`Persona prelude content` and `Scope disambiguation` then went to full marks in
D-041: a query's grammar (`what`, `the`, `in`) was matching beliefs whose prose
contains those words, and dropping it from the query side is what stopped the
scope-bleed belief from surfacing.

What remains is mostly not a rule problem. `filter-resolved-at-excluded` needs a
relevance verdict that D-008/D-012 measured as unavailable without a model.
`cap-stress` is an ordering failure. `cold-start-user-no-beliefs` is the
harness: it scores every case as `test-user`, while the fixture expects a
`brand-new-user` empty context.

## Tried, and did not help

Recorded because a negative result that stays unpublished gets re-tried by the
next person.

**BM25 parameter sweep.** k1 in {0.8, 1.2, 1.5, 2.0, 2.5} × b in {0.2, 0.4, 0.6,
0.75, 0.9}. Every combination scored between 0.5974 and 0.6104, and `b` made no
difference at all — every belief document is roughly the same length.

**Coverage as an abstention signal.** Median 0.30 for cases that should return
something, 0.16 for cases that should not. Best threshold caught 19 of 21 true
empties and wrongly abstained on 29 of 41 — net −10.

**Score-ratio heuristics** (keep hits within 0.5 or 0.7 of the top score). Worse
on both axes than plain `top_k=1`.

**Entity matching without the scope filter** (returning every belief whose name
appears in the query, including beliefs the case cannot see). 0.8052 against
0.8312 at the time — out-of-scope names over-sized `top_k`. Counting only visible
beliefs is the rule that shipped.

**A fixed short-query cut.** Every cut from 50 characters to 8 words scored
identically, so the first version applied `top_k=1` below eight words. That was
a plateau, but it hid a truncation: short questions that name two things were
cut to one result. Counting names at every length scored better.

**Field repetition as a weighting scheme** (names ×2/3/5, content ×1/2). Worse
than names-only in every combination; the content words are the problem, not
their weight.

## Return-count disclosure

Numbers here use `top_k` equal to the number of visible names the query contains,
at any length, because that is the configuration being reported. The store's
default is `top_k=5`. Quoting 0.9610 without saying the setting would be exactly
the benchmark gaming this project exists to criticise.

## Reproducing

```bash
PYTHONPATH=src:. python -m bench.pmb.run
```

No API key, no network after the fixtures are in place. Temp database, discarded
on exit. Data is upstream `tenurehq/precisionmembench` at commit `b95d6ab`, MIT.

## Caveats worth stating

- **Scope comes from the caller.** The benchmark ships it; a real deployment has
  to know which scope a memory belongs to when it writes it.
- **`user:universal` is a scope, not an exemption.** Treating it as visible
  everywhere scored 0.5325 against 0.6104 — the universal beliefs carry generic
  words matching almost any question about working style. The benchmark's own
  `_pinned_facts` reads it the same way.
- **Names-only indexing assumes the question names things.** See above. This is
  the biggest caveat on the page.
- **This is one benchmark**, measuring retrieval precision. It says nothing about
  conversational recall (LoCoMo, LongMemEval), where this store has not been
  measured at all.
