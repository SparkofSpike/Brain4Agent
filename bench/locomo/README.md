# LoCoMo retrieval

The second public benchmark, run on the same terms as the first: fixed data, no
model in the loop, and the number published whatever it says.

## Result

```
answerable: 1531 of 1540 questions

hit@1   0.6270
hit@5   0.8850
hit@10  0.9471
mrr     0.7369
```

By category (hit@1):

```
open-domain   0.7111
temporal      0.6156
single-hop    0.4591
multi-hop     0.4045
```

**These are retrieval numbers, not answer accuracy.** Published LoCoMo scores
(0.67–0.92) measure whether a generated answer matches the gold answer under an
LLM judge. A retrieval number can only bound that: with `hit@5 = 0.8850`, the
best possible answer accuracy at five retrieved sessions is 88.5%, and a real
system scoring well below that means the loss is in reading, not in finding.

Anyone comparing 0.6270 against Mem0's 0.669 or Letta's 0.740 is comparing two
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

+8.2 points of hit@1, from changing which text goes into the index and nothing
else. An answer spread across four short turns produces four mediocre matches;
the same text as one document produces one strong one.

Keeping the speaker labels (`Caroline: ...`) measured slightly better than
dropping them (0.6270 against 0.6218) — a question that names a person gains from
the name sitting next to what they said.

## What did not matter

**BM25 parameters.** k1 in {1.2, 1.5, 2.0} × b in {0.2, 0.4, 0.6, 0.75, 0.9}
spans 0.6166 to 0.6316. The default (1.5 / 0.75) is within noise of the best.
Reported because a flat grid is a result: this corpus is not sensitive to length
normalisation, unlike PrecisionMemBench where `b` had literally no effect because
every document was the same size.

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
- **0.6270 is not a state-of-the-art claim.** It is a baseline with no model, on
  a benchmark where most published systems spend tokens on every step.

## Data provenance

`documents.json`, `queries.json` come from `vectorize-io/agent-memory-benchmark`
`data/locomo/locomo10/`, itself derived from the LoCoMo release. Redistributed
here under the same terms so the run is self-contained. If the upstream snapshot
is ever corrected, this copy has to be replaced and the numbers re-measured —
that is the cost of shipping data instead of downloading it, and the trade is
worth it for a benchmark whose whole purpose is being checkable.
