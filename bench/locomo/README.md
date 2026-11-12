# LoCoMo retrieval

The second public benchmark, run on the same terms as the first: fixed data, no
model in the loop, and the number published whatever it says.

## Result

```
answerable: 1531 of 1540 questions

hit@1   0.7054
hit@5   0.9086
hit@10  0.9549
mrr     0.7921
```

By category (hit@1):

```
open-domain   0.7919
temporal      0.6906
single-hop    0.5445
multi-hop     0.4494
```

**These are retrieval numbers, not answer accuracy.** Published LoCoMo scores
(0.67–0.92) measure whether a generated answer matches the gold answer under an
LLM judge. A retrieval number can only bound that: with `hit@5 = 0.9086`, the
best possible answer accuracy **for a reader that sees exactly five sessions and
cannot reason across them** is 90.9%.

That condition matters and this file originally omitted it. Allow ten sessions
and the bound is this store's own `hit@10 = 0.9549`; allow a reader to reason
between sessions — which AMB's own prompt instructs it to do — and there is no
single-retrieval bound at all. It is a statement about one configuration, not
about the field, and it is not evidence that any other system's retrieval is
worse.

Anyone comparing 0.7054 against Mem0's 0.669 or Letta's 0.740 is comparing two
different measurements. The README says so here so that nobody has to discover it
later.

## Why it is reproducible

```bash
PYTHONPATH=src:. python -m bench.locomo.run
```

- **Data is shipped, not fetched.** `data/` holds AMB's export of LoCoMo
  `locomo10`: 272 sessions, 5882 turns, 1540 questions. Two files, 1.9 MB. No
  download step, nothing to drift.
- **No model is called.** No sampling variance, no API version to pin, no key to
  have. Same input, same number, on any machine.
- **The metric is the same one `bench/zh/e2e.py` uses**, so those two numbers are
  the only pair on this page that can be read side by side.
- **The temp database is discarded**, so a run leaves nothing behind.

## The one design decision that mattered

**Retrieval unit must match the answer's unit.** LoCoMo labels each answer with a
session id. Indexing turns and joining them at score time — the way the Chinese
benchmark does — loses badly:

```
turn-level      hit@1 0.5447   hit@5 0.8210   hit@10 0.9099   mrr 0.6595
session-level   hit@1 0.6270   hit@5 0.8850   hit@10 0.9471   mrr 0.7369
```

(Both rows predate D-041; the current session-level column is the Result block
above.)

+8.2 points of hit@1, from changing which text goes into the index and nothing
else. An answer spread across four short turns produces four mediocre matches;
the same text as one document produces one strong one.

Keeping the speaker labels (`Caroline: ...`) measured slightly better than
dropping them (0.6270 against 0.6218) — a question that names a person gains from
the name sitting next to what they said.

## What did not matter

**BM25 parameters.** k1 in {1.2, 1.5, 2.0} × b in {0.2, 0.4, 0.6, 0.75, 0.9}
spans 0.6166 to 0.6316 before D-041. The default (1.5 / 0.75) is within noise of
the best. Reported because a flat grid is a result: this corpus is not sensitive
to length normalisation, unlike PrecisionMemBench where `b` had literally no
effect because every document was the same size.

**Coverage re-ranking and BM25L.** Measured again on this corpus in D-041:
coverage re-weighting is flat or worse (0.6688-0.6702 against 0.6702), BM25L's
length compensation adds 0.0013 hit@1 at best. Both stay out of the product.

D-042 then showed what the difference is, and it is not a matter of weight. Full
coverage asks *whether* the words are in the session, which BM25 already knows.
Proximity asks *where* — the fraction of distinct query terms sharing one 40-token
window — and that scored 0.6950-0.7080 across a 5x6 grid, every cell above the
baseline. The distinction is not "coverage again, tuned better": it is a fact the
index discards, recovered rather than reweighted.

## Query-side folding

The largest single gain on this benchmark is not in the scorer (D-041). English
questions spend much of their length on grammar, and answers state inflections
the question did not use:

```
hit@1 0.6310 -> 0.6702   drop English stopwords, fold inflections on both sides
hit@1 0.6702 -> 0.6826   reward adjacent query words (phrase bonus 0.3)
hit@1 0.6826 -> 0.7054   reward terms that share a passage (proximity 1.5, D-042)
```

The first row costs nothing extra to query; the second builds a pair index, which
roughly doubles rebuild time; the third keeps the token sequences, which costs
query time on long documents only. `python -m bench.locomo.tuning_probe`
reproduces the first two, one variant at a time, including the sweep that picked
0.3; `python -m bench.locomo.proximity_sweep` does the same for the third.

## Sample isolation

LoCoMo contains ten independent conversation samples. A question from `conv-26`
must not be able to retrieve `conv-30`'s sessions, and the benchmark treats
cross-sample retrieval as a wrong answer regardless of content.

That isolation is a tag filter, not a separate database:
`tags=["sample:conv-26"]` at write, `tags=["sample:conv-26"]` at read. This is
the D-025 scope mechanism doing a real job rather than serving a leaderboard.

## What this does not say

- **It says nothing about PrecisionMemBench-style precision.** There, a single
  extra result fails a case; here, top-5 is fine. Same store, opposite tuning.
- **It says nothing about answer quality.** No generator ran.
- **0.7054 is not a state-of-the-art claim.** It is a baseline with no model, on
  a benchmark where most published systems spend tokens on every step.

## Data provenance

`documents.json`, `queries.json` come from `vectorize-io/agent-memory-benchmark`
`data/locomo/locomo10/`, itself derived from the LoCoMo release. Redistributed
here under the same terms so the run is self-contained. If the upstream snapshot
is ever corrected, this copy has to be replaced and the numbers re-measured —
that is the cost of shipping data instead of downloading it, and the trade is
worth it for a benchmark whose whole purpose is being checkable.
## On answer accuracy: probed, not measured

Five questions were run through the read stage (`b4a_ask`) to see whether an
answer-accuracy number was worth producing. It is not, yet — for two reasons
worth recording rather than hiding.

**Temporal questions need dates the read stage does not pass.** Asked "when did
Caroline go to the LGBTQ support group", the reader answered "yesterday" — which
is what the transcript says, and is exactly right *relative to the conversation
date*. The gold is "7 May 2023". LoCoMo ships `query_timestamp` in every
question's metadata precisely because resolving "yesterday" requires knowing when
the question was asked. A reader that ignores that metadata fails every temporal
question for reasons having nothing to do with retrieval.

**Three of five calls returned nothing.** Not a wrong answer — no answer. That is
a transport or rate-limit failure surfacing as `None`, which means an accuracy
number from this path would measure API reliability as much as memory quality.

Both are fixable. Neither is fixed, so no number is published. `hit@5 = 0.9086`
remains the honest bound: retrieval finds the answer's session 90.9% of the time,
and what a reader does with it is unmeasured.

Running this properly would need session dates in the evidence block,
`query_timestamp` in the prompt, retry with backoff around the model call, and a
judge whose prompt is published. That is a separate piece of work. Doing it badly
would produce a number that looks comparable to Mem0's 0.669 and is not.
