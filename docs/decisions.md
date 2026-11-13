# Design Decisions

Each entry records the context, the options considered, the measurement that
decided it, and what it cost. Rejected hypotheses stay in the log: "tried and it
failed" is different from "never tried", and the difference saves the next
person a week.

---

## D-001 Evaluation before implementation

**Date:** 2026-11-06
**Status:** Done

**Context**

The project had a motivating failure — a memory system returning "no relevant
memories" for a Chinese query while the index held eighteen matching entries.
There was no way to tell a fix from a coincidence, because nothing was measured.

**Options**

1. Write the memory system first, designed from the papers, and evaluate later.
2. Build the benchmark first, establish a baseline, and let the numbers decide
   what to write.

**Evidence**

"No relevant memories" is a plausible output. Without a quantitative acceptance
criterion, "fixed" is a feeling.

**Decision**

Option 2. Stage 0 builds evaluation only and does not touch the implementation.
Stage 1 writes core code, and only where Stage 0 showed a problem.

**Consequences**

Two days spent on measurement before any feature. In exchange, every later
claim in this document has a number behind it.

---

## D-002 LongMemEval rather than a homegrown benchmark

**Date:** 2026-11-06
**Status:** Done

**Context**

A benchmark is needed; building one from scratch is a research project of its
own and would measure the benchmark as much as the system.

**Evidence**

- LongMemEval (arXiv:2410.10813, ICLR 2025): 500 questions, 6 categories,
  official cleaned release.
- Downloadable (`xiaowu0162/longmemeval-cleaned`, MIT).
- The original release has sessions whose noise interferes with answer
  correctness; the cleaned version fixes it.

**Decision**

Use the cleaned release. Do not use the original.

**Consequences**

It is English. A Chinese benchmark has to be built separately — recorded as
D-006.

---

## D-003 Identify abstention by the `_abs` suffix, not by `question_type`

**Date:** 2026-11-06
**Status:** Done (after a wrong first reading)

**Context**

The dataset contains unanswerable questions and they need to be identified.

**The mistake**

No `abstention` value appears among the six `question_type` categories, which
was read as "this benchmark has no abstention questions". That conclusion was
wrong.

**Evidence**

Questions whose `question_id` ends in `_abs` number **30**, spread across four
`question_type` values:

```
multi-session         12
single-session-user    6
temporal-reasoning     6
knowledge-update       6
```

**Decision**

`is_abstention = question_id.endswith("_abs")`.

**Consequences**

`question_type` classifies *how a question is asked*; abstention is a property
of *whether an answer exists*. The two are orthogonal. Filtering one dimension
by the other silently drops an entire class of samples — the same category of
failure the project exists to avoid.

---

## D-004 Zero dependencies until one lifts a specific number

**Date:** 2026-11-06
**Status:** In force

**Context**

Should the first commit pull in jieba, numpy, or a vector store?

**Reasoning**

Stage 0 exists to measure. Every library added before the first number is a
variable that can mask a real result. A baseline is only useful if nobody has to
trust it; BM25 is from 1994 and easy to verify.

**Decision**

`dependencies = []`. BM25 written by hand, Chinese handled with character
bigrams.

**Exit condition**

A benchmark number showing "adding X lifts Y". When that happens, add it and
record the decision here. jieba met this bar later (D-019).

---

## D-005 Character bigrams for Chinese, not dictionary segmentation

**Date:** 2026-11-06
**Status:** Partially validated, later superseded by D-019

**Context**

Chinese has no whitespace. Any tokeniser that splits on non-word characters
collapses a Han query to a single token, which matches nothing.

**Options**

1. jieba — dictionary plus HMM, ~20 MB, pure Python.
2. Character bigrams — no dictionary, O(n).
3. A vector model — heavy.

**Argument for bigrams**

- Never returns zero tokens for Han text, which is the failure mode to avoid.
- No dictionary means no unknown-word problem: proper nouns (`覆巢演出`,
  `角色管理页`) are never mis-segmented.
- Costs index size (n characters → n−1 bigrams) and some precision.

**Noise reduction that shipped**

A bigram made entirely of function characters is dropped. Runs are **not**
split character by character: `社会`, `会议`, `用户` survive; `的了`, `是在` do
not. This specific trade came from an earlier failure in which splitting on
function characters destroyed 36 of 49 common words.

**Status**

Unvalidated at the time, because no Chinese benchmark existed. D-006.

---

## D-006 A Chinese benchmark is debt that had to be taken

**Date:** 2026-11-06
**Status:** Resolved by D-009

**Context**

None of the eight surveyed papers handle CJK; LongMemEval and LoCoMo are
English throughout.

**Consequences**

- The Chinese tokenisation strategy (D-005) could not be validated.
- The "Chinese-first" positioning had no evidence behind it.

This was the largest open problem in the project at the time.

---

## D-007 Retrieval is not the bottleneck

**Date:** 2026-11-06
**Status:** Measured

**Context**

The original premise for building a new system was that retrieval is the hard
part.

**Evidence**

BM25, zero dependencies, 470 answerable questions, haystack averaging 493 turns:

```
hit@1     0.866
hit@5     0.9511
hit@10    0.966
recall@10 0.9087
mrr       0.899
complete misses  16 / 470
```

By question type:

| question_type | n | hit@10 | mrr |
|---|---|---|---|
| knowledge-update | 72 | 1.000 | 0.9769 |
| single-session-assistant | 56 | 1.000 | 1.0000 |
| single-session-user | 64 | 1.000 | 0.9701 |
| multi-session | 121 | 0.975 | 0.8973 |
| temporal-reasoning | 127 | 0.945 | 0.8786 |
| single-session-preference | 30 | 0.800 | 0.4656 |

**Decision**

Retrieval is not a reason to rebuild. A thirty-year-old algorithm with no
dependencies reaches 96.6% hit@10 on realistic haystacks.

**Consequences**

One of the main justifications for the project was removed. The weakest category
is `single-session-preference` (mrr 0.47), which is not a retrieval problem: it
asks the system to infer a preference, not to locate a session containing a
keyword.

---

## D-008 Scores are not an abstention signal

**Date:** 2026-11-06
**Status:** Measured, hypothesis rejected

**Context**

If the top-1 BM25 score separated answerable from unanswerable questions, a
single threshold would implement abstention and Stage 1 would be cheap.

**Evidence**

Top-1 raw scores over 500 questions:

```
abstention n=30   min= 9.61  median=19.67  max= 43.38
answerable n=470  min= 9.35  median=25.23  max=135.49
```

The distributions overlap almost completely. Abstention questions reach 43,
higher than more than half the answerable ones.

The reason is structural: every abstention question carries a decoy — it asks
about a hamster when only a cat was mentioned, or about Korea when only Japan
came up. Decoy and question share vocabulary, so BM25 scores them highly.

Threshold sweep (full curve in `bench/results/abstention-probe.json`):

| threshold | refuse | keep | balanced |
|---|---|---|---|
| 0 | 0.000 | 1.000 | 0.500 |
| 18 | 0.467 | 0.736 | 0.601 |
| **24** | **0.833** | **0.543** | **0.688** |
| 44 | 1.000 | 0.157 | 0.579 |

Best balanced score is 0.688 against a random baseline of 0.5, and reaching it
discards **46% of normal queries**.

**Decision**

Score thresholds cannot implement abstention. This is not a tuning problem; the
signal does not exist.

**Consequences**

Abstention requires modelling whether a memory **answers** the question, not
whether it **contains** related words. That is comprehension, not ranking. Stage
1's first task was fixed here, with a quantitative target: raise refusal without
dropping keep below 0.95.

---

## D-009 Chinese benchmark: 59 questions over a shared corpus

**Date:** 2026-11-06
**Status:** Built

**Context**

Eight papers, all English benchmarks, no evidence for a Chinese-first claim.

**Approach**

`bench/zh/build.py` writes 30 sessions (127 turns) and 59 questions (40
answerable, 19 abstention) by hand. Content is drawn from real agent working
memory, not machine-translated — translation preserves English sentence
structure and erases exactly the property under test.

**One shared corpus, not one haystack per question.** The real situation is one
agent with one project library; every question searches the same memories. This
is harder and more honest.

**Abstention questions carry decoys.** Each asks about something never
mentioned, while the corpus mentions something similar. A question about a
completely unrelated topic is free to answer and flatters any threshold.

**Manual review found real errors.** `zh048_abs` asked "how many seconds is the
global budget" while the corpus explicitly stated 380 seconds — that question
was answerable, and was removed. `zh041_abs` was a mixed question and was
narrowed to its unanswerable half. All 19 abstention questions were checked
against top-2 retrieval to confirm the answer is genuinely absent.

**Added difficulty.** 14 sessions have entity-swapped hard negatives (`s01v`
and so on), taking the corpus from 30 to 44 sessions. Same topic, different
facts — a real memory library has three projects with a media-submission page.

**Limits**

44 sessions is small; hit@10 saturates at 1.0. **hit@1 is the discriminating
metric.**

---

## D-010 Bigrams are adequate; jieba is more precise

**Date:** 2026-11-06
**Status:** D-005 partially confirmed

**Evidence**

59 Chinese questions, same BM25:

| metric | bigram | jieba |
|---|---|---|
| hit@1 | 0.825 | **0.875** |
| hit@5 | 0.950 | 0.950 |
| hit@10 | **1.000** | 0.975 |
| MRR | 0.8911 | **0.9098** |
| complete misses | **0** | 1 |

**Decision at the time**

Keep bigrams as the default: equal or better recall, zero dependencies, no
mis-segmentation of proper nouns. jieba stays as a measuring instrument in
`bench/zh/compare_tokenizers.py`.

**Hypothesis rejected**

Dictionary segmentation was expected to improve the abstention signal. It does
not — 0.696 against 0.695. The abstention bottleneck is not segmentation.

**Superseded**

D-019 later showed the real answer is not "pick one" but "use different
tokenisers on each side".

---

## D-011 Return evidence, not verdicts

**Date:** 2026-11-06
**Status:** Architecture fixed

**Context**

D-008 ruled out score thresholds. Can a local model decide instead?

**It can, and it should not be the library's job.**

A local model asked "does this evidence contain what the question needs?", over
the 59 Chinese questions:

```
Stage 0 (always answer)    keep 1.000   refuse 0.000
with model judgement       keep 0.950   refuse 0.895   <- target met
```

The cost: `qwen3.5:9b` holds 15.2 GB resident. The machine has 23.6 GB, and a run
locked it up entirely. `4b` needs 9 GB and reaches keep 0.92. `2b` needs 1 GB and
reaches keep 0.75.

**Why the memory system should not do this**

The caller is an agent. It already has a model. Starting a second one to ask
"is this memory relevant" is duplicated spend, not a feature, and it turns a
tens-of-megabytes library into a fifteen-gigabyte service — the wrong shape for
a memory store.

**Architecture**

```
memory system                        caller (agent)
─────────────                        ──────────────
retrieve → rank                      receives evidence + missing terms
each with: score / coverage / gaps   judges with the model it already has
no model, no inference, tens of MB   or shows the gaps to a human
```

`missing_terms` is the key field — not a score, specific words:

```
answerable:   what is the address and deploy directory?   cov=1.00  no gaps
unanswerable: which database is installed on it?          cov=0.44  missing ['数据库']
```

A caller reading `missing: 数据库` can decide without any extra inference.

**Ceiling of non-model approaches, all measured**

| signal | refusal at keep ≥ 0.95 |
|---|---|
| BM25 top-1 score | ~0 |
| IDF-weighted coverage | ~0 |
| unseen-word ratio (bigram) | ~0.05 |
| unseen-word ratio (jieba) | 0.105 |
| word-level coverage | ~0.10 |
| local LLM judgement | 0.895 |

An order of magnitude apart. Abstention is the caller's decision, made from
evidence. The memory system's responsibility is to make the evidence clean.

**What this gives up**

Out-of-the-box abstention. The cost lands on the caller — which is the party
that should carry it.

**Tokenizer consequence**

`missing_terms` needs real words, not bigrams: `数据库` split into `数据` and
`据库` loses the signal, because `数据` appears elsewhere. Word extraction uses
jieba (a static dictionary, not a model, no inference cost); retrieval keeps
bigrams for their recall. Without jieba, extraction falls back to bigrams and
the report is noisier but still correct.

---

## D-012 QPP and classifiers: tried, insufficient

**Date:** 2026-11-07
**Status:** Measured, both routes closed

**Context**

D-011 handed abstention to the caller. Before doing so, how far can a zero-model
approach actually get? Information retrieval has a subfield for this — Query
Performance Prediction — and its classical methods are purely statistical, which
is exactly the right shape.

**Evidence**

59 Chinese questions, refusal rate at keep ≥ 0.95:

| method | refusal | source |
|---|---|---|
| NQC | 0.158 | Shtok et al. 2012 |
| UQC | 0.158 | same |
| WIG | 0.000 | Zhou & Croft 2007 |
| Clarity | 0.105 | Cronen-Townsend et al. 2002 |
| top1 − top10 margin | 0.105 | this project |
| 13 features + logistic regression + LOOCV | **0.211** | this project |
| local LLM | 0.895 | D-011 |

**The literature agrees.** arXiv:2504.01101 (Chifu et al., 2025) tested QPP
across four collections and four retrievers: predictors correlate poorly with
actual performance, are collection-sensitive, and "selective query processing
offers only marginal gains".

**Root cause**

QPP predicts **ranking quality**, not **whether an answer exists**. A query can
retrieve the most relevant session perfectly while that session contains no
answer — which is precisely how abstention questions are constructed. Predicting
one thing and needing another.

**Why the classifier does not close the gap either**

Thirteen features reach 0.211, twice any single feature, still a quarter of a
model's refusal rate. The strongest weight is `t_howmany` ("how many / how many
times", +0.859): questions asking for a specific number are more often
unanswerable. But deciding whether 数据库 and 部署目录 name the same concept
requires semantic understanding that counting cannot supply.

**Decision**

Zero-model abstention ceiling is 0.211. Any such capability ships labelled
best-effort, not as a guarantee. Real adjudication belongs to the caller.

---

## D-013 The gap is Chinese, no model, and a single file

**Date:** 2026-11-07
**Status:** Surveyed

**Context**

Before building in a crowded space, find out what is actually missing. Full
analysis in `docs/landscape.md`.

**Findings**

- **TencentDB-Agent-Memory**: 30,584 stars, TypeScript, MIT. A team-level memory
  hub with four asset types. Installation begins by filling in two sets of LLM
  parameters — a hard dependency on a model.
- **causal-memory**: 104 stars, Rust, Apache-2.0, 368 tests, still active. Causal
  edges, spreading activation, consolidation, its own benchmark. Uses local ONNX
  embeddings — not an LLM, still a model. Self-reports that factual recall is not
  its strength (LongMemEval-S 76.4% against mem0's 94.4%). No Chinese support, no
  abstention.
- mem0 / Zep / Letta / Cognee: all require embeddings or an LLM.

**Decision**

1. Do not rebuild the causal model — causal-memory has 368 tests; using it beats
   duplicating it.
2. Narrow the positioning to **Chinese + no model + single SQLite file**. No
   project occupies all three.
3. Borrow from causal-memory's write path (raw turns into an audit table, distilled
   facts into a graph) without copying its architecture.

**Honest caveat**

This is a technical gap, not a validated demand. How many users would trade
accuracy to avoid a model is unknown.

---

## D-014 A hosted API works, but not on the hot path

**Date:** 2026-11-07
**Status:** Measured

**Context**

Local models either lock up the machine (9B / 15 GB) or miss the target (2B /
keep 0.75). Does a hosted API avoid that trade?

**Evidence**

`deepseek-flash`, 59 Chinese questions, one retry:

```
keep_rate    0.925      (target 0.95 -- short by one question)
refuse_rate  1.000
errors       0 / 109 calls
latency      mean 0.807s  median ~0.6s  max 1.611s
tokens       213.8 / call
```

Comparison:

| approach | keep | refuse | latency | resident |
|---|---|---|---|---|
| BM25 only | 1.000 | 0.000 | 0.22 ms | 0 |
| local 2B | 0.750 | 1.000 | 0.77 s | ~1 GB |
| local 9B | 0.950 | 0.895 | 1.5–2 s | 15.2 GB |
| hosted `deepseek-flash` | 0.925 | 1.000 | 0.81 s | 0 |

Zero resident memory for near-9B accuracy. That is its only advantage, and it is
a real one.

**What went wrong in the integration**

1. `deepseek-chat` returned HTTP 503 "Service is too busy" throughout.
   `deepseek-flash` worked.
2. `deepseek-flash` is a reasoning model and emits `reasoning_content` first.
   With `max_tokens=8` the reasoning consumed the budget and `content` came back
   as an empty string — indistinguishable from a refusal unless you know. 256 is
   the working value.
3. Without retries, 5–6% of calls failed (500, 503, literal `null` body). One
   retry took 109 calls to zero failures. **Retry is required, not an
   optimisation.**
4. One run without a timeout produced a 363-second call. Timeout is now 30 s, and
   a timeout counts as "no judgement", never as "no answer".

**Why it is still not the answer for the hot path**

0.81 s × 30 memory lookups per task is 24 seconds of added latency, spent on a
judgement the calling agent could make while it is already reasoning. Duplicated
inference, serialised.

Cost is not the obstacle: 214 tokens per call, roughly ¥0.0001. Latency is.

**Decision**

Available as an optional enhancement, off by default, enabled only when a key is
configured. Requires retry, short timeout, and caching together.

---

## D-015 Benchmarks import the shipping code

**Date:** 2026-11-08
**Status:** Done

**Context**

Stage 0 had its own `tokenize` and `BM25` in `bench/retrieval.py`. Once those
moved into `src/brain4agent/`, two copies existed.

**The risk is not theoretical**

Two copies drift. The benchmark keeps scoring a version nobody ships, and the
README describes an implementation that no longer runs. An earlier project
failure came from exactly this shape: write and query paths tokenising
differently while tests covered only one path.

**Decision**

`bench/retrieval.py` re-exports from `brain4agent.tokenize` and holds no
implementation. A benchmark that scores code nobody ships is a story about the
past.

**Consequences**

`bench/` now depends on `src/`. `pyproject.toml` sets `pythonpath = ["src"]` so
the scripts run from a checkout.

**Superseded by D-035.** The exception above did not last: by the time it was
re-examined, `bench/zh/evidence.py` had no benchmark caller left -- only the
evidence tests imported it -- so it was deleted and those tests now run against
`brain4agent.evidence` and `MemoryStore.search`.

---

## D-016 The hosted judge has three outcomes, not two

**Date:** 2026-11-08
**Status:** Done

**Context**

D-014 established that a hosted judge works but is slow. Whether and how to
integrate it.

**Decision**

Integrate with three constraints:

1. **Off by default.** `Judge.from_env()` reads `BRAIN4AGENT_JUDGE_KEY` and
   returns `None` when unset. The core path `store.search()` never touches the
   network.
2. **Three states.** `search_judged()` returns `found`, `not_found`, or
   `uncertain`.
3. **Retry and timeout are built in**, not configurable away. Thirty seconds and
   one retry, because D-014 measured both as necessary.

**Why the third state exists**

With two states, a judge that times out has to be reported as `not_found`. During
an outage the system would then tell a user "I never recorded this" — while
actually not knowing.

That is the failure this project exists to fix, in a better suit: not an empty
list, but a confident wrong conclusion.

**Implementation note**

`max_checks` bounds how many sessions are judged, because each judgement is an
API call. Without it, cost scales linearly with retrieval depth.

**Known gap**

keep 0.925, below the local 9B's 0.950. The missing question has not been
diagnosed — it could be a model error or a data problem (`zh012`'s answer shares
43% of its characters with the question). This does not affect the "optional
enhancement" positioning but does affect whether it should default on.

---

## D-017 MCP implemented against the protocol, not the SDK

**Date:** 2026-11-08
**Status:** Done

**Context**

To be usable by an agent, the library needs an MCP interface. The official `mcp`
package brings a large transitive dependency tree.

**Trade-off**

The stdio transport needs four methods: `initialize`, `tools/list`,
`tools/call`, `ping`, plus the rule that notifications get no reply. The wire
format is newline-delimited JSON-RPC 2.0.

The official SDK provides schema validation, type hints, multiple transports, and
OAuth. All useful; none required to let an agent read one memory.

**Decision**

Implement it directly. `mcp` does not enter `dependencies`.

**Costs, stated plainly**

1. Protocol drift is now ours to track. Pinned to `2024-11-05`.
2. No schema validation; tool arguments are checked by hand.
3. stdio only — no SSE, no HTTP.

If an HTTP transport is ever needed, adopting the SDK at that point is right:
that part is not four dictionaries.

**One invariant**

stdout carries protocol messages and nothing else. A single stray `print`
corrupts the stream and the client sees a parse error instead of a tool result.
All diagnostics go to stderr, and one end-to-end test runs the real process and
feeds every output line through `json.loads`.

---

## D-018 Corpus absence is not session absence

**Date:** 2026-11-08
**Status:** Done

**Context**

D-011 proposed "a word from the question that the corpus lacks" as a zero-model
signal. Measured at top-1, keep was **0.325** — 27 false refusals out of 40
answerable questions. Unusable.

**Two separate errors**

**First: no part-of-speech filter.** `直接`, `应该`, `不是`, `哪几个`, `最终`,
`需要`, `修复` are the grammar of asking, not the content of an answer. Their
absence from a corpus is ordinary. Restricting extraction to nominals
(n / nr / ns / nt / nz / eng / vn / an) plus long digit strings — an IP or a
version number is exactly what should be reported as missing — took false
refusals from 27 to 16.

Incidentally, jieba tags `一共` as `j` (abbreviation), which it is not. The `j`
tag was removed from the allow-list: an unreliable label is worse than no label.

**Second, and larger: the semantics were wrong.** The measurement asked whether
a word was absent from *that session*. `用户` and `记忆` are absent from most
individual sessions and present in the corpus. Ten of the sixteen false refusals
came from this.

The right question is whether the word exists **anywhere in the corpus**:

- session lacks a word → this retrieval is not relevant. A ranking matter.
- corpus lacks a word → the system does not know the thing. That is a fact.

Both are now exposed, under separate names:

- `SearchResult.corpus_missing` — the whole corpus lacks it (strong, factual)
- `Evidence.missing_terms` — this session lacks it (explains the ranking)

**Measured after the fix**

```
                     keep    refuse
always answer        1.000   0.000
missing_terms        0.875   0.632     <- zero model
local 9B             0.950   0.895
hosted flash         0.925   1.000
```

**Relationship to D-012**

Every signal in D-012 reached at most 0.211 refusal at keep 0.95.
`corpus_missing` reaches 0.632 at keep 0.875 — 7.5 points lower keep for roughly
three times the refusal rate.

It is not "better", it is a different kind of thing: a stated fact about the
corpus rather than a confidence score. A caller receives something it can show,
reason about, and act on, not `0.31`.

**Still not a binary decision**

keep 0.875 means 12.5% of answerable questions would be refused automatically.
That is not good enough to decide for the caller, so the signal remains evidence.

**The remaining five false refusals**

`组件`, `核心`, `水平`, `态度`, `调整` — all genuine nouns, genuinely absent from
the corpus. Not fixable by filtering, and not being fixed: these words could be
the key entity in another corpus, and tuning a stop-list against 59 questions
would only make it look good on 59 questions.

**Verification**

`python -m bench.zh.missing_terms_signal`. `bench/zh/e2e.py` runs the shipping
code over the same corpus and matches the D-010 retrieval baseline exactly, so
none of this moved the ranking.

---

## D-019 Index wide, query narrow

**Date:** 2026-11-08
**Status:** Measured and shipped

**Context**

The project's founding claim — indexing should cover broadly, querying should
stay precise — had never been tested. It was inferred from LongMemEval's
three-stage framing and repeated in the README.

**Evidence**

`bench/zh/indexing_probe.py`, 3 index strategies × 3 query strategies over the
same Chinese corpus:

```
index    query      hit@1   hit@5  hit@10     MRR
bigram   bigram    0.8250  0.9500  1.0000  0.8911   <- original
bigram   jieba     0.9000  0.9750  0.9750  0.9229
bigram   content   0.9000  0.9750  0.9750  0.9229
jieba    bigram    0.8750  0.9500  0.9750  0.9073
jieba    jieba     0.8750  0.9500  0.9750  0.9114
jieba    content   0.9000  0.9500  1.0000  0.9233
both     bigram    0.8250  0.9750  1.0000  0.8883
both     jieba     0.9250  0.9750  1.0000  0.9404   <- adopted
both     content   0.9000  0.9500  0.9750  0.9231
```

**Findings**

1. **Every asymmetric pairing beats the symmetric `bigram/bigram` cell.**
2. Best is index `both` (bigrams plus dictionary words), query `jieba` words.
3. Against the original: hit@1 +0.10 (0.825 → 0.925), MRR +0.049, hit@10 held
   at 1.0.

**Why**

- Wide index: bigrams give unknown words (names, coinages, code identifiers) a
  matching path, dictionary words give the precise term. With both, missing one
  still leaves the other.
- Precise query: words rather than bigrams, so `州测` and `试机` — grams that
  cross a word boundary and mean nothing — never enter scoring.

**A variant measured and rejected**

Filtering single-character function words (`和`, `的`, `先`, `做`) from the
query looked obviously cleaner:

```
both/content    0.9000  0.9500  0.9750  0.9231   <- worse
```

Losing 0.025 hit@1 and 0.017 MRR. Single-character words carry more signal than
noise, at least on this corpus. Not changing something because it reads better.

**Costs**

- Index size roughly doubles (bigrams plus words).
- Querying depends on jieba.

jieba therefore remains an optional extra. Without it, `_tokenizers()` falls back
to `bigram/bigram` and the numbers drop to 0.825 while everything keeps working.
The fallback is three lines because it is simply the default path.

**Product verification**

`bench/zh/e2e.py` runs the shipping `MemoryStore` and returns identical numbers:
hit@1 0.9250, hit@5 0.9750, hit@10 1.0000, MRR 0.9404. No gap between the probe
and the product, because the benchmark imports the product (D-015).

**One known regression**

The single temporal-reasoning question (zh032, "which came first — the first fix
or the limit adjustment?") fell from rank 1 to rank 4. BM25 is a bag of words and
has never understood ordering; the earlier rank-1 hit was luck. hit@10 remains
1.0 and the net gain elsewhere is larger.

## D-020 Bag-of-words retrieval degrades on dense same-topic corpora

**Date:** 2026-11-08
**Status:** Measured, no fix adopted

**Context**

Every benchmark number in this project comes from 44 sessions. A working agent
accumulates thousands. If hit@1 decays with corpus size, the benchmarks describe
a toy.

**Experiment**

`bench/zh/scale_probe.py` adds noise sessions at 100 / 300 / 1000 depth. Two
kinds, because they ask different questions:

- **same-answer** — same topic, different fact. The variant says "the root cause
  is tokenizer.py line 512"; the original says "keywords.py line 443". Both
  answer the question and they disagree. This is the common case in a real
  library: three deployments, each with a correct address.
- **unanswered** — same vocabulary, no answer. The variant says "looked at some
  file, some line". A true negative.

Noise is written first, so originals cannot win ties for free.

**Results**

| corpus | hit@1 | hit@5 | MRR |
|---|---|---|---|
| 44 (clean) | 0.9250 | 0.9750 | 0.9404 |
| +300 same-answer | 0.0000 | 0.0000 | 0.0844 |
| +1000 same-answer | 0.0000 | 0.0000 | 0.0000 |
| +300 unanswered | 0.0000 | 0.0000 | 0.0844 |

With the original sessions written first instead, hit@1 returns to 0.9250 at
every noise level. The score is identical; only insertion order decides.

**Why**

BM25 matches literal terms. Asked *"which file and line is the root cause in"*:

- the session saying `root cause is keywords.py line 443` contains 根因 but
  neither 文件 nor 行
- the session saying `looked at some file, some line` contains 根因, 文件, and 行

The vague non-answer matches the question better than the precise answer does.
This is not an implementation defect; it is what a bag of words does.

**Three candidate fixes, all measured, all rejected**

| fix | noisy hit@1 | clean hit@1 | verdict |
|---|---|---|---|
| BM25 `b` ∈ {0, 0.3, 0.75, 1.0} | 0.0000 | — | no effect |
| recency tie-break (newest first) | 0.0000 | 0.6750 | damages the normal case |
| concreteness bonus (IP/path/file/line) at α=0.5 | 0.0250 | 0.9000 | recovers 1 of 40, costs 0.025 |

The `b` sweep rules out length normalisation: `b=0` does not help a shorter
non-answer, so document length is not the mechanism.

The concreteness bonus was the most promising idea — real answers carry paths,
file names and line numbers; talk about a question does not. It recovers exactly
one question and costs more than it gains. Rejected.

**What this means**

- **0.9250 hit@1 is real but optimistic.** It describes a 44-session corpus with
  no dense near-duplicates. It is not a claim about a year of accumulated memory.
- **real deployments will see lower hit@1**, in proportion to how many similar
  memories share a topic.
- **readers should use top-5 and `corpus_missing`**, not top-1. hit@5 survived
  the same-answer noise at 100 depth and collapsed at 300, which is the honest
  shape of the limit.
- **it confirms D-011.** The memory system cannot judge relevance. A heuristic
  standing in for that judgement does not work either, and this is now measured
  rather than assumed.

**Not fixed**

No product change. Every candidate fix made the normal case worse. Stating the
limit is more useful than shipping a heuristic that hides it.

**What would actually help**

Semantic matching — which means a model, which the core path does not have. The
available options are therefore: accept the limit, or let the calling agent supply
context (the current conversation) so the query carries enough signal to
disambiguate. The second is cheap and belongs in the caller, not here.

## D-021 Oracle entities recover 15%: the failure is not missing literals

**Date:** 2026-11-08
**Status:** Measured, closes the line of enquiry

**Context**

D-020 showed hit@1 collapsing under same-topic noise and diagnosed it as literal
matching: a vague non-answer repeats the question's words, while a precise answer
contains file names and line numbers the question never mentions.

If that diagnosis is right, adding the missing detail to the query should help.
This measures how much, and whether the ceiling is worth chasing.

**Experiment**

`bench/zh/context_probe.py`. The query is extended with concrete entities taken
straight from the gold session — file names, line references, IPs, paths, commit
hashes — at three depths. This is an **oracle**: a user could not type it without
already knowing the answer. It is an upper bound, not a feature.

300 noise sessions, written first.

| query | hit@1 | hit@5 | MRR |
|---|---|---|---|
| plain | 0.0000 | 0.0000 | 0.0844 |
| + 1 entity | 0.0500 | 0.0500 | 0.1344 |
| + 3 entities | 0.1500 | 0.1500 | 0.2344 |
| + all entities | 0.1500 | 0.1500 | 0.2344 |

**Finding 1: over half the questions have nothing to extract**

23 of 40 answers contain no concrete token at all.

- "Did the downstream components need changes?" — answered "no".
- "What did the first fix get wrong?" — answered with a paragraph about
  character-splitting destroying common words.

There is no file name, no address, no number. The oracle has no lever for 57.5%
of the set, so 0.425 is the arithmetic ceiling before anything is measured.

**Finding 2: even with entities, the oracle recovers 6 of 17**

Knowing the exact file name and line number, BM25 still puts the right session
first less than half the time.

**A bug that had to be fixed first**

The first run reported 0.10. The cause was not retrieval: `substitute_same_answer`
replaced four patterns and left commit hashes untouched, so seven variants shared
`4a29ae6` with the original and tied with it honestly. The gold session sat at
rank 8 behind them. Fixing the substitution moved hit@1 to 0.15 — a real
improvement, and still nowhere near recovery.

Worth recording because the same trap applies to any future noise experiment: a
variant that shares one un-substituted token makes the result look like a
retrieval failure when it is a fixture failure.

**Conclusion**

The D-020 diagnosis was correct and incomplete. Literal matching explains the
failure; supplying the missing literals does not fix it. What remains is the part
of the question with no literal form — *which* of several similar facts the user
means.

That is semantic resolution, and the core path has no model. The honest position:

- BM25 is adequate when each topic appears once or twice in the corpus.
- It degrades in proportion to how many near-duplicate memories share a topic.
- No query-side trick recovers it, because the query is not what is missing.

**Why this is D-011 for the third time**

The project has now arrived at "push the judgement to the caller" from three
independent directions:

- D-008: scores cannot express abstention.
- D-012: QPP predicts ranking quality, not answer presence.
- D-021: retrieval cannot express which instance the user means.

Each was a different question, each measured, each landing on the same boundary.
The architecture is not a preference; it is where three separate measurements
converge.

**What actually helps**

Not retrieval. Either the calling agent supplies enough context that the question
names the specific instance — "the keywords.py fix" rather than "the fix" — or
the system returns top-k and the caller disambiguates. Both put the judgement
where a model already exists.

## D-022 Forgetting is storage management, not a retrieval fix

**Date:** 2026-11-08
**Status:** Implemented, measured, limits stated

**Context**

The README claimed forgetting was part of the design — "a memory store that only
grows degrades" — and no code implemented it. That is the kind of gap that turns
documentation into marketing.

**Decision**

`prune(keep_sessions=N)` deletes the oldest sessions by write order. Exposed as
`brain4agent prune --keep N`. Retention is by recency, on the assumption that a
user's recent work is what they ask about.

**Measurement: does forgetting fix D-020?**

1,044 sessions — 1,000 same-topic noise written first, the 44 originals last —
then pruned progressively:

| keep | removed | hit@1 | hit@5 |
|---|---|---|---|
| 1000 | 44 | 0.000 | 0.000 |
| 500 | 500 | 0.000 | 0.000 |
| 200 | 300 | 0.000 | 0.650 |
| 100 | 100 | 0.000 | 0.950 |
| 50 | 50 | 0.775 | 0.950 |

**Recovery requires deleting 95% of the store.** At keep=50 only the 44 originals
plus six noise sessions remain. That is not forgetting; that is emptying.

Even then hit@1 is 0.775 rather than the original 0.925, because six near-duplicate
sessions still win two questions.

**Conclusion**

Forgetting controls size and cost. It does not make a corpus of near-duplicates
retrievable — the duplicates are the problem, and a recency policy keeps exactly
the recent ones, which in a real library is exactly the noise.

**What it is good for**

- Bounding disk and memory for a long-running agent.
- Dropping sessions a user has moved past.
- Not a substitute for semantic matching (D-021).

**The obvious next step, not taken**

There is no access counter and no importance score; retention is pure write
order. If retention ever matters in practice, counting retrievals and keeping
what is used would be the first improvement. It is not built because there is no
evidence yet that it would help, and this project's rule is that a heuristic
needs a number.

## D-023 Distillation improves recall and damages top-1 precision

**Date:** 2026-11-08
**Status:** Measured; distillation is optional, not the default

**Context**

D-020 showed that storing every turn creates near-duplicate noise. The obvious
fix is to store distilled facts instead — that is what neural-memory does with
fibers, and what `remember()` now does with a hosted model.

Does it actually retrieve better?

**Three stores, one extraction pass**

Same 40 Chinese questions, same corpus. Extraction ran once; the fact-only and
fact+source stores were built from the same 103 memories so the model was not
called twice.

| store | records | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| raw turns | 127 | **0.9250** | 0.9750 | 1.0000 | **0.9404** |
| facts only | 103 | 0.7000 | 0.9750 | 1.0000 | 0.8238 |
| facts + source indexed | 103 | 0.7750 | **1.0000** | 1.0000 | 0.8771 |

`remember()` alone, extracting single turns, scored 0.4750 — see below.

**Finding 1: extracting answers without their questions halves the score**

127 turns produced 63 empty results, and almost every one was a *question*.
Questions look like "根因在哪个文件" — nothing durable to a model reading them
in isolation — but they contain exactly the words a future search will use.
Dropping them dropped hit@1 from 0.9250 to 0.4750.

`remember_exchange()` fixes this by distilling the pair: the question supplies
the vocabulary, the answer supplies the content. 0.4750 → 0.7250.

**Finding 2: distilled facts compete with each other**

Even paired, distilled facts score lower on top-1 while matching raw turns on
hit@10. Indexing the original text alongside the distilled sentence recovers
some of it (0.7000 → 0.7750) but not all.

The mechanism is the same one from D-020. Every fact is a clean, topic-focused
sentence, so 103 facts share far more vocabulary than 127 turns did. Common
terms gain frequency and lose IDF, and the first result becomes a coin flip
between several memories that are all "about" the question.

**What distillation is actually good for**

- **Recall.** hit@10 is 1.0 and hit@5 is 1.0 with source indexing: the answer is
  always in the first five. What suffers is only the ordering.
- **Reading.** "extraction/keywords.py 第 443 行的正则不含汉字范围" is a better
  thing to hand an agent than a four-turn transcript.
- **Discarding.** 63 exchanges distilled into 103 memories instead of 127 turns,
  and chit-chat is dropped rather than stored.

**What it is not good for**

Top-1 precision, which is what a caller reading one result consumes.

**Decision**

- `add()` stays the default for benchmarking: raw turns, hit@1 0.9250.
- `remember()` / `remember_exchange()` are opt-in, and `index_source=True` is
  the better configuration when using them.
- The MCP tool defaults to `distil=true` because a real agent's memory is not a
  benchmark corpus — chit-chat filtering matters more there than 0.15 of hit@1
  on a 40-question set.

**What this says about "better than neural-memory"**

Distillation alone does not get there. nmem retrieves through a graph
(spreading activation, causal traversal), which is how it can afford to store
clean facts: the graph supplies the discrimination that word matching loses.
Storing facts and searching them with BM25 gets the worst of both — the noise
reduction of distillation and the ranking power of a bag of words.

The next piece is the graph. That is now the measured gap, not a guess.

## D-024 Two attempts to fix distilled ranking, both rejected

**Date:** 2026-11-08
**Status:** Measured, both discarded

**Context**

D-023 showed distilled memories recall as well as raw turns (hit@10 1.000) but
rank worse (hit@1 0.775 against 0.925). The failure is inside the candidate set,
not outside it: the right fact is in the top ten and not first.

Two mechanisms were tried, both zero-model.

**Attempt 1: entity co-occurrence graph**

`graph.py` indexes memory ↔ entity, and retrieval expands one hop from the top
BM25 hits along shared entities with 0.35 decay.

| alpha | hit@1 | MRR |
|---|---|---|
| 0.0 | 0.8000 | 0.8794 |
| 0.2 – 1.0 | 0.8000 | 0.8794 |

**Identical at every weight.** The reason is structural, not a tuning miss: the
graph can only surface memories *outside* the BM25 candidate list, and the ones
that need help are already inside it. A boost applied to a memory that is
already ranked tenth can move it, but every strong hit also propagates a boost,
so the relative order barely changes.

`EntityGraph` was kept — it is used for nothing yet, and it may matter for
cross-session questions where recall rather than ordering is the problem — but
it is not wired into search.

**Attempt 2: re-ranking signals**

Four signals swept over the BM25 order:

| signal | best hit@1 | best alpha |
|---|---|---|
| none (baseline) | **0.8000** | — |
| coverage | 0.7750 | 0.5 / 2.0 |
| entity overlap | 0.7750 | 0.5 / 2.0 |
| specificity (paths, numbers) | 0.8000 | 0.5 |
| brevity | 0.7500 | 0.5 |

**Every signal is neutral or harmful.** The most informative one, `coverage`,
is already an IDF-weighted version of the thing being measured, and re-weighting
by it double-counts the same evidence BM25 already used.

**Conclusion**

BM25's order is the ceiling for the signals available without a model. The
0.775–0.800 range is not an under-tuned result; it is what distilled facts score
when matched by words.

**What follows**

- Distillation is a **write-time** feature: it filters chit-chat, produces
  readable memories, and keeps recall at 1.0. It is not a retrieval upgrade, and
  it costs top-1 precision.
- `index_source=True` stays the recommended configuration when distilling: it is
  the best of the distilled variants (0.775 against 0.700).
- Raw turns remain the default for anything measured against the benchmark,
  because that is what scores 0.9250.

**The honest reading of "better than neural-memory"**

On Chinese retrieval, faster and simpler deployment, this project is ahead — and
the comparison in `bench/results/nmem-comparison` shows by how much (0.9250
against 0.1750, though nmem's number is dominated by a CJK bug this project
already has a fix for upstream).

On memory *structure* — causal edges, consolidation, decay, graph traversal —
nmem is ahead, and adding a graph here did not close the gap, because a graph
answers a question this evaluator does not ask: it finds memories that share
almost no words with the query. What it cannot do is pick between several
memories that share all of them.
---

## D-025 The metadata layer: what the text does not say

**Context.** PrecisionMemBench scored 0.3766. Every failure fell into one of two
shapes: a memory that should have come back did not, or one that should have
stayed out did. The second kind dominated, and none of them was a matching
problem -- the words matched fine. The memory was about the wrong scope, or had
been replaced, or was an open question rather than a fact.

**Decision.** Three fields on a memory, all supplied by the caller at write time:

- `tags` -- labels retrieval can filter on. Scope is the motivating case.
- `superseded_by` -- set by `supersede(old, new)`; excluded from search by default.
- `refs` -- other memories this one points at; expanded at search time only when
  the caller asks.

None of them is inferred, and none of them needs a model.

**Measured.** Same 77 cases, one layer at a time:

```
raw BM25, top_k=1                    0.3766   active 0.4419
+ drop superseded                    0.3896           0.4419
+ scope tags                         0.5065           0.5116
+ kind whitelist                     0.5584           0.5349
+ relation expansion                 0.6104           0.6279
```

Third-party results on this dataset, per the benchmark's own site
(`tenureai.dev/benchmark`): open-knowledge-format 18/43 active and 36/77 total is
the best of them. The benchmark's authors score 43/43 and 77/77 with their own
implementation. Worth repeating: this is a retrieval
precision benchmark, not conversational recall, and none of these numbers say
anything about LoCoMo or LongMemEval.

**Two things I had wrong.**

`user:universal` is a scope, not an exemption. I first treated it as "visible to
every query", on the reasoning that universal means universal. It scored 0.5325
instead of 0.6104: those beliefs carry generic words (recommendation, pushback)
that match nearly any question about working style, and letting them in wrecked
precision. The benchmark's own `_pinned_facts` treats universal as a scope like
any other -- a belief is in scope only when one of its scopes appears in the
case's scope list. Now this store does the same.

`distil` defaults to False now, a reversal of the previous default and of an
earlier decision in this same file. The old reasoning was that storing every turn
floods memory with near-duplicates. That reasoning is not wrong, but it optimised
the wrong quantity: D-023 measured write-time distillation as a hit@1 loss from
0.9250 to 0.7750, and STONE (arXiv:2602.16192) names the cause -- the extractor
cannot know what the memory will later be asked. Extraction now happens at read
time, in `b4a_ask`.

**Why top_k=1.** The benchmark fails a case for one extra document, so returning
less wins: 0.6104 at k=1, 0.4156 at k=2, 0.3247 at k=3. This is disclosed in the
bench README rather than presented as an unqualified score. The store's own
default remains top_k=5; k=1 is a choice made for this benchmark, and an agent
reading context wants more than one memory.

**What did not work.** BM25 parameter sweeps (flat across the whole grid),
coverage-based abstention (net -10), score-ratio heuristics (worse on both axes),
missing-term ratio (net +4 at the cost of five false abstentions). All four try
to read intent off the query's words. The signals that worked were labels
supplied at write time. Recorded in `bench/pmb/README.md` so they are not
re-tried blind.

**The limit this exposed.** 30 cases still fail. The dominant remaining shape is
questions that legitimately need more than one memory: 8 cases want two beliefs,
2 want six, 1 wants seven, and `top_k=1` serves none of them. Serving them
requires raising k, which fails the far more numerous `shouldOnlyInclude` cases.
That trade cannot be made from the query side with the signals available -- it
needs a model that knows the question is compound, which is exactly what this
project has been avoiding.
---

## D-026 Index the signal, not the prose

**Context.** After D-025, PrecisionMemBench sat at 0.6104. Sixteen cases still
failed, and reading them showed a pattern that had nothing to do with matching:
questions about MongoDB returned the Redis memory because "code" appeared in
both. Questions about the protagonist returned the wrong belief because "chapter"
appeared in both.

**Decision.** Separate what is stored from what is indexed. `add(index_text=...)`
supplies a narrower signal for the index; `content` remains what gets returned.
Storage and retrieval answer different questions and no longer have to share one
string.

Measured on the same 77 cases, with the same stored text:

```
full belief text       0.6104
drop why_it_matters    0.7013
names only             0.7922
content only           0.5844
```

The mechanism is BM25's crowding problem. Explanatory prose shares ordinary words
with the whole domain -- every belief says "backend", "code", "explicit". Each has
low idf; there are many of them; together they outweigh the one distinctive name.
This is the same failure STONE describes at a larger scale: an extraction step
that keeps the *explanation* and loses the *identifier*.

Worth being explicit about what this is not. It is not "throw away the content" --
the content is still stored and still returned, and `b4a_ask` reads it. It is the
index that narrows, because an index's job is discrimination, not comprehension.

**The caveat that must travel with this number.** Every query in this dataset
names what it asks about. A question that describes instead of naming ("what do we
use for caching?") needs the content indexed and would score differently. That is
why `index_text` defaults to None and the store indexes what it stores; the
narrow indexing is the harness's choice, made for a benchmark whose queries name
things.

**Second decision: size the result set from the question.** Published query
decomposition needs a model (arXiv:2507.00355 and relatives). This does not need
one:

```
query words < 8   ->  top_k = 1
query words >= 8  ->  top_k = number of distinct belief names in the query
```

0.7922 -> 0.8312. The rule does not understand the question; it counts how many
known names appear in it. The threshold is a plateau rather than a point -- every
cut from 50 characters to 8 words scored identically -- which is why it is
reported as a rule and not as a fitted constant.

The counting lives in the harness, not the store. `MemoryStore.search` takes a
`top_k`; it has no opinion about compound questions. That boundary is deliberate:
anything that needs the fixture to interpret a question belongs above the store,
not inside it.

**Result.** 34/43 active and 64/77 total, from a starting point of 0.3766. Best
third-party result on the benchmark; the authors' own implementation is 43/43.

**Correction, later.** The first version of this entry said "above the published
best of 0.4675" and sourced its comparison table from AMB's
`external_results.json`. Both were wrong. 0.4675 is open-knowledge-format's
**total** rate (36/77), while this store's headline was its **active** rate
(34/43) — two different denominators. And AMB's file disagrees with the
benchmark's own site, which records gbrain at 11/77 rather than 34/77. The
narrower claim, best third-party result on both columns, is what survives. Still no model.

**What remains is not rules.** 13 cases fail, and the residue is ordering and
fuzzy matching -- "k9s" for k8s, "mango" for MongoDB. Selection is largely solved;
ranking is not. Edit distance is the obvious next step and the last one that a
rule can plausibly close.
---

## D-027 Supersession redirects, it does not hide

**Context.** `supersede(old, new)` was implemented as a filter: a superseded
memory was excluded from search results. An exchange with the user exposed why
that is not just wrong but backwards.

```
old: 主库是 MySQL 8.0
new: 上个月迁到 PostgreSQL，MySQL 已下线

search("主库") before supersede  ->  [old]
search("主库") after  supersede  ->  []
```

The new memory never says 主库. The old one is the only thing that matched, and
dropping it dropped the answer. **A memory system that forgets a fact the moment
it is updated is worse than one that never had it** — the user asks a question
they have already answered, and gets silence instead of the correction.

I had this in place for several days and described it in the README as working.
It took one concrete example to show it was not.

**Decision.** A hit on a superseded memory resolves to its replacement. The score
stays the old memory's score, because the old memory is what matched the query.

Chain following is bounded by a visited set; a cycle returns nothing rather than
looping. `include_superseded=True` opts out of redirection entirely — that is the
"show me the history" path, and it should show what actually matched.

**Measured.** No change on PrecisionMemBench (64/77 total, 34/43 active). Its
supersession cases query the current belief directly, so the redirect never
fires. **The benchmark could not see this bug; thirty seconds of use could.**

That is worth recording as a limitation of the benchmark, not just of the code.
PrecisionMemBench asks "does the right belief come back"; it does not ask "does a
question phrased against the *old* belief still get answered". Those are different
questions and only one of them was being measured.

**What this is not.** It is not cross-session aggregation. Two memories still
exist; one points at the other. Nothing is merged, and no model decides that they
describe the same thing — the caller asserts it. The only change is what a search
does when it lands on the replaced one.

**The residual.** If the new memory shares no vocabulary with the old, this
recovers the *session* but the evidence's coverage score stays low, because the
session text genuinely does not contain the query's words. Indexing the old
memory's terms into the new one would fix that. Not done, and it is the next
piece of work in this area.
**Follow-up, done the same day.** Redirecting the hit recovered the session but
left `coverage` at 0.00, because coverage is computed from the session's raw text
and the replacement genuinely does not contain the old question's words. A search
that returns the right memory while reporting "this session lacks 主库" is worse
than an empty result — it looks like an answer and reads like a denial.

`supersede()` now copies the old row's *token column* into the new row's. Terms,
not content: the old row's tokens already carry anything **it** inherited, so a
chain A -> B -> C passes A's vocabulary through C without re-segmenting history
each time. `dict.fromkeys` keeps the merge idempotent, so re-linking does not
grow the row.

Coverage still reads raw text and stays low after a redirect. That is now honest
rather than contradictory: `coverage` answers "how much of the question is
literally in this session", and the answer is none. What changed is the *match*,
not the text. Making coverage credit inherited terms would need an inherited-text
column; the signal is not wrong, it is measuring something narrower than a reader
might assume.

One test was rewritten rather than repaired. `include_superseded=True` now
returns **both** rows: the old one matches on its own text, the new one on what it
inherited. The previous assertion had them mutually exclusive.
---

## D-028 Near-duplicates are reported, never merged

**Context.** The user asked what "cross-session aggregation" meant, and the
answer exposed a second gap besides D-027. Two sessions can say the same thing in
slightly different words, and nothing in the store noticed. A search returns both;
the reader has to work out which is current. D-027 handles the case where a
caller *declared* a replacement. This handles the case where nobody did.

nmem solves this inside its consolidate pass. Reading its implementation
(`engine/consolidation.py`) gives four operations — reinforce by access frequency,
prune by weight decay, merge overlapping fibres, summarise clusters into concept
neurons. Two of those are worth having here, and two are not:

- **Merge**: worth it, and it needs no model. nmem uses SimHash.
- **Summarise**: needs a model. The caller has one.
- **Reinforce by frequency**: needs an access counter, and nothing yet justifies
  letting it affect ranking.
- **Prune by weight**: D-022 already showed the naive version does not work.

**Decision, part one: detect, do not merge.** A SimHash fingerprint is computed
at write time and stored. `duplicates_of(id)` returns memories within a Hamming
distance of 10. `Evidence` keeps a `possible_duplicates` field for callers that
fill it, but search leaves it empty: the scan measured 5 ms against 2893
memories, more than half of a query that otherwise takes 4 ms, and it answers a
maintenance question rather than a retrieval one.

**No automatic merge, and this is the important half.** Merging is irreversible.
The store has no standing to decide that two memories state the same fact — that
requires reading both and judging, which is what the calling agent does for a
living. What the agent could not do was notice. Now it can.

The feedback loop is the point: the agent asks `duplicates_of(id)`, reads them, and if they are the same thing it calls `supersede()`. The store never
guesses.

**Decision, part two: count accesses, do not rank by them.** Every search
increments an `accesses` column on the memories it returned. nmem boosts synapse
weights for frequently-traversed paths; this records the same signal and does
nothing with it yet. Frequency-based ranking is a hypothesis, and this project's
rule is that a hypothesis needs a measurement before it changes behaviour.

**Measured.** Three benchmarks unchanged: PrecisionMemBench 64/77 and 34/43,
LoCoMo hit@1 0.6270, Chinese 0.9250. Nothing regressed, which is the only claim
available — the feature is additive and no benchmark tests it.

Two implementation notes worth keeping:

- SimHash is 64-bit unsigned and SQLite's INTEGER is signed. Storing the raw value
  overflows on insert. It is stored two's-complement and masked back on read.
- `memory()` re-reads the access count from the database instead of trusting the
  cached object, because `_annotate` bumps the counter without marking the index
  dirty. Without that, a counter that had just been incremented read back as 0.

**What is still open.** The 10-bit threshold was taken from the literature, not
measured against this corpus. On the one hand-constructed example (a changed IP
address) it separates correctly: 5 bits apart for a one-character edit, 30 bits
apart for a different topic. Whether that holds across the real corpus, and
whether 10 is the right cut, is unmeasured. Until then `duplicates_of` is a
hint, and the README says so.
---

## D-029 Weight: reinforcement, decay, and pruning on evidence

**Context.** D-028 counted accesses and explicitly did nothing with them. The
user's objection was fair: a counter that never affects anything is bookkeeping,
not memory. A weight is what makes the count and the decay mean something.

**Decision.** A `weight` column in [0, 1], initialised to `importance`.

- `reinforce(id, delta=0.03)` raises it, capped at 1.0.
- `decay(factor=0.95, floor=0.0)` multiplies every weight, explicitly, when the
  caller asks.
- `prune_by_weight(floor=0.05, dry_run=False)` deletes what has fallen below.
- `search(weight_blend=0.0)` optionally mixes weight into ranking.

**The four defaults that keep this from being a regression.**

1. **`weight_blend=0.0` by default.** BM25 is untouched. Every published number
   on this page was measured with that default, and a corpus that never calls
   `reinforce()` scores exactly as it did before the column existed.
2. **Weight 0.5 is neutral.** At blend 1.0 a memory still at the default keeps its
   score exactly; below 0.5 shrinks, above grows. So even a caller who turns the
   blend on gets no change until something is actually reinforced.
3. **`decay` is explicit, not scheduled.** A time-based decay has to decide what
   "age" means for a memory written once and read fifty times. Nothing here
   measures which answer is right, so the store does not pretend to know: the
   caller runs it weekly, after a sprint, or never.
4. **Pruning has a test, and it has no default floor.** `prune_by_weight()`
   requires the caller to name a threshold, and `dry_run=True` reports the count
   first. Deleting memories is irreversible.

**Why pruning by weight matters more than ranking by weight.** D-022 measured
that `prune(keep_sessions=N)` needs to delete 95% of a noisy corpus before
retrieval recovers — that is emptying, not forgetting, because write order says
nothing about usefulness. Weight is the first signal in this project that
distinguishes a memory worth keeping from one worth dropping on grounds other
than age: **one that is retrieved survives, one that is never retrieved decays
away.** That is a real answer to "what should be forgotten", and it is the first
one here that is not a heuristic about time.

**Why ranking by weight stays opt-in.** The mechanism works — a memory reinforced
five times outranks its twin at `weight_blend=1.0`, and I have the test to prove
it. But working is not the same as helping. Nothing here has measured whether
real retrieval patterns correlate with what should rank first, and letting an
unmeasured signal reorder results would be exactly the kind of unverified change
this project keeps refusing to make.

**Measured.** All three benchmarks unchanged at default settings:
PrecisionMemBench 64/77 and 34/43, LoCoMo hit@1 0.6270, Chinese 0.9250. That is
the intended outcome and also the limit of the claim — **no benchmark tests
whether the weight is useful, only that adding it broke nothing.**

**The honest gap.** The weight mechanics are verified; the weight *value* is not.
A recommendation to actually use `prune_by_weight` would need a corpus where some
memories are genuinely dead and some are genuinely alive, and a measurement that
the surviving set still answers the questions. That experiment has not been run.
---

## D-030 Pruning on a real corpus: age is anti-correlated with usefulness

**Context.** D-029 shipped `prune_by_weight` and `decay`, and said plainly that
the mechanism was tested while the *value* was not — it needed a corpus where
some memories are genuinely dead and some genuinely alive, plus a record of which
were which. The user pointed at the obvious source: the nmem brain on this
machine, 2925 fibers accumulated over months of real use.

**Decision.** Run the experiment. `bench/nmem/prune_probe.py` copies the database,
imports it, replays the recorded usage as reinforcement, and compares two pruning
policies at equal size.

**Measured.**

```
2891 fibers with usable text  (34 dropped as UTF-8-read-as-GBK mojibake)
 660 ever recalled            (22.8%)
2231 never recalled
```

Prune to 660 — the size of the genuinely-used set:

```
strategy                 kept   lived kept   self-recall
---------------------- ------ ------------ -------------
no pruning               2891       100.0%         64.6%
by weight (usage)         660       100.0%         31.5%
by age (newest first)     660         0.0%         90.2%
random                    660        22.1%         69.8%
```

**By weight keeps every memory that was ever used. By age keeps none of them, and
does it worse than random.** The reason is structural, not statistical: a memory
that has been useful for months is old, so age-ordered pruning deletes exactly
the memories that earned their keep. Age is not uncorrelated with usefulness
here — it is *anti*-correlated, and D-022's finding that pruning needs to delete
95% of a noisy corpus before retrieval recovers is the same fact measured a
different way.

**The number that argues the other way, kept rather than buried.** `self-recall`
— can a memory still be found by its own text — falls from 64.6% to 31.5% after
weight-based pruning. The surviving set is the memories that were used, and those
cluster around a few subjects, so they compete with each other on any query. Age
pruning keeps newer, more varied memories: individually easy to find, mostly
useless.

The two metrics answer different questions. `lived kept` asks whether the store
kept what mattered; `self-recall` asks whether each survivor is individually
locatable. A retrieval layer that exists to answer questions the user actually
asks should be judged on the first. An archive judged on browsing should be
judged on the second. This is the former, and the README says which one is being
quoted.

**What this does not establish.** One user, one corpus, one machine. The 0.0% is
a property of *this* usage pattern; someone whose work is uniformly recent would
see a different number and the policy might compare differently. And `frequency`
is nmem's counter, recorded by a different retrieval system with different
failures — a proxy for usefulness, not a measurement of it. Weight-based pruning
agreeing with observed usage is not the same as weight-based pruning agreeing with
value.

**What changed in the code.** Nothing. The experiment tested what D-029 shipped;
it did not reveal a defect. Two bugs were found and fixed in the *probe* itself,
both worth recording because they would have produced wrong conclusions:

1. The first version pruned without a decay pass. Every memory starts at the
   neutral weight 0.5, so "prune by weight" deleted nothing at all and the policy
   looked identical to no pruning.
2. The three strategies shared one database file, so each one pruned what the
   previous had already deleted. The last two looked catastrophic for the wrong
   reason. Each now runs on its own copy.
---

## D-031 First real agent session: two findings the benchmarks could not produce

**Context.** Every number in this repo came from a test I built. D-029's README
admitted that `b4a_ask`, `recall-context`, weight and SimHash had never been
exercised by an actual agent. The user pointed out the obvious gap: `codewhale
exec` exists, run one.

Setup: `b4a` added via `codewhale mcp add`, `BRAIN4AGENT_DB` and `PYTHONPATH`
written into `~/.codewhale/mcp.json`, a five-memory store built, then
`codewhale exec --auto "<question>"`.

### Finding 1: an agent with the source in front of it will not consult memory

First question: *"What does Brain4Agent use for retrieval, and why is there no
distillation at write time?"*

The agent answered correctly and **never called `b4a_recall`**. It ran `bash`,
read `store.py`, `cli.py` and `read.py`, and derived the answer from the code.

That is the right behaviour and it invalidates the test. A memory store competes
with the filesystem, and loses whenever the answer is in a file the agent can
open. Its territory is what is *not* in the workspace: verbal agreements,
decisions made in conversation, the reason a thing was rejected six weeks ago.

The second experiment used only such memories — "no comments unless asked", "we
do not publish to PyPI", a server address, a root-cause file and line from an old
bug. Those cannot be derived from `src/`.

### Finding 2: two memory servers, and the agent picks one

With both `neural-memory` and `b4a` enabled, the agent answered:

> first checked memory: Brain4Agent has only one Stage 0 conclusion in the memory
> store … no record of code style, release policy, or a test server

Five memories were in the `b4a` store at that moment. It had queried nmem,
received nothing relevant, and reported that as the state of memory — while
citing a test-server address belonging to an unrelated project, which is a
fact from the nmem brain.

Direct protocol test confirmed `b4a` was healthy: a hand-written MCP handshake
returned the correct two sessions. The failure was at selection, not at retrieval.

With `codewhale mcp disable neural-memory`, the same prompt worked:

- multiple `b4a_recall` calls with different queries (`注释 依赖 提交`,
  `服务器 部署 域名 端口 主机`)
- correct results, each attributed to a session id
- `corpus_missing` reported honestly — `ruff`, `格式化`, `风格` genuinely absent
- and a critique the store did not make: it noticed `198.51.100.23` is RFC 5737
  TEST-NET-2, non-routable, and concluded the memory was probably seed data

That last part is worth noting. The agent used the evidence as evidence — it did
not treat a retrieved fact as a verified one. That is exactly the posture
`corpus_missing` and the evidence/verdict split were built for.

### What this changes

**A deployment note, not a code change.** Any user running two memory servers
will hit Finding 2, and the symptom is the worst kind: the agent says "memory has
no record of X" when memory has a record of X. The store cannot fix this from
inside; it has to be documentation — one memory server per agent, or names that
make the choice unambiguous.

**A test-design lesson.** Benchmark questions that can be answered from the
corpus they are embedded in measure nothing. The Chinese set and PrecisionMemBench
both ask about content that exists only in the store, which is why they did not
expose this — but I had not articulated the rule, and a naive "let the agent try
it" experiment walked straight into it.

**No code changed.** `b4a` behaved correctly in every test. What failed was
integration, and it failed in a way only an integration test could show.

**Caveats.** One agent, one prompt pair, five memories, and an experimenter who
wrote the memories himself. This establishes that the mechanism is usable and
that co-existence is a hazard. It does not establish that a real user's memory
corpus is retrievable in practice — that needs weeks, not one session.
---

## D-032 Cross-harness comparison: dsh was clean, Codewhale was not

**Context.** D-031 found that Codewhale, with `neural-memory` and `b4a` both
enabled, queried the wrong memory server and reported "no record" while the
answer was in `b4a`. The user's follow-up was the right control: Codewhale has
used nmem for months, so try `dsh` — DeepSeek Harness — which is rarely used here
and has no nmem habit.

**Setup.** `dsh` boots from a profile stack. The MCP entry lives in
`$DSH_HOME/cordis.patch.yml`:

```yaml
- id: mcp-b4a
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: b4a
    transport: stdio
    command: 'python'
    args: ['-X', 'utf8', '-m', 'brain4agent.mcp_server']
    env: {BRAIN4AGENT_DB: ..., PYTHONPATH: ..., PYTHONUTF8: '1'}
```

Tools surface as `mcp__b4a__b4a_recall`, so a namespace collision with another
server is structurally impossible — the failure mode D-031 ran into cannot occur
in dsh's naming scheme.

**Measured.** Same five memories, same questions, `dsh --profile headless`.

Question 1 — code style, release policy, test server:

- cited `p1`, `d1`, `o1` with session ids and quoted text
- reported `corpus_missing` for `lint` / `格式化` / `风格` and said so explicitly
- flagged `198.51.100.23` as RFC 5737 TEST-NET-2, non-routable, probably seed data
- stated at the end: "以上是记忆库里查到的情况，未做推断"

Question 2 — what test framework CI uses and why:

- separated three sources without being asked: what the repo contains, what the
  decision log contains, and what has no record at all
- grepped `docs/decisions.md` for `pytest`, found zero hits, and said the *reason*
  is not archived anywhere
- offered to record it, rather than inventing one

Its own reasoning trace contains the line: *"I did NOT query the AGENTS.md/repo
conventions as memory."* That is the distinction the experiment was testing for,
and it made it unprompted.

**What this establishes.**

1. **The store works in a harness that has no competing memory server.** Two
   independent agents, two protocols, same result.
2. **D-031's failure was selection, not retrieval** — confirmed by the contrast.
   Same memories, same questions, different harness, no failure.
3. **Server-qualified tool names prevent the collision by construction.**
   `mcp__b4a__b4a_recall` cannot be confused with another server's `recall`. The
   codewhale failure happened because two servers both exposed memory tools and
   the agent chose one. dsh's naming makes the choice explicit in the tool name
   itself.

**What this does not establish.** Two agents and four questions is a smoke test,
not a trial. Both agents were asked about memories I wrote, with wording I chose,
ten minutes earlier. The interesting question — does a real corpus with hundreds
of memories surface the right one under real questions — is untouched.

**Action taken.** None in the store. The dsh config change lives outside the
repo; a copy is kept in this entry so it can be reconstructed. `cordis.patch.yml`
was backed up before editing and the temporary store was deleted afterwards.
---

## D-033 Real scale: 2893 memories, and the queries that worked

**Context.** Every previous test used five memories I wrote minutes earlier. The
user's instruction was direct: copy the nmem brain over and let the agent search
it with `b4a`.

**Setup.** `bench/nmem/import_to_b4a.py` reads the nmem database read-only, takes
each fiber's summary or essence as one memory, skips mojibake rows, and imports.

```
fibers:   2928
usable:   2893   (34 empty, 1 mojibake)
import:   8.9 s
store:    3.0 MB, 22799 index terms
```

The dsh profile points at the imported store; the question is asked in Chinese and
the agent is told to answer only from memory.

**Result, negative case.** *"How was the Oracle escaping issue solved?"*

> 没有。记忆库里查不到 Oracle 数据库转义问题的记录。
>
> 依据是检索给出的「整份语料缺失」信号，而不是单条会话不匹配：`corpus_missing`
> 直接列出了 `oracle`，说明整个记忆库中从未出现过这个词。

It then listed the three nearest sessions and named why each was not it (FTS
index corruption, a schema mismatch, a PowerShell escaping bug). **That is the
distinction `corpus_missing` exists to make**, used correctly and without
prompting: "not this session" versus "never recorded".

**Result, positive case.** *"What went wrong with PowerShell writing Chinese
files, and how was it solved?"*

It returned six memories across three categories — silent corruption on rewrite,
BOM injection by `Set-Content`/`Out-File`/`*>`, and the reverse failure of reading
UTF-8 as GBK — each with a session id, one of them noticing the source text is
itself truncated mid-sentence.

Then the part that matters more:

> 记忆里**没有**完整的替代写法记录：我特意查了 `WriteAllText`、`UTF8Encoding`
> 这两个词，返回的 `corpus_missing` 里就是它们。所以"用
> `[System.IO.File]::WriteAllText(...)` 写无 BOM"这种做法，b4a 里查不到，
> **不能算记忆内容**。

It checked whether the correct answer was in memory, found it was not, and
declined to supply it from its own knowledge — while stating what the correct
answer likely is. That is exactly the boundary this project is built around, and
the agent drew it without being told to.

**What this establishes.**

1. **2893 memories at 3 MB, 22.8k terms, 8.9 s import, 17–34 s per question**
   end-to-end including multiple retrievals and reading. That is a real if small
   corpus, and the store handled it without tuning.
2. **Retrieval is accurate enough to be useful on this corpus.** Six relevant
   memories surfaced from 2893 for a topic that occupied maybe 0.2% of it.
3. **The absence signal survived scale.** `oracle` returned zero across 2893
   sessions, and the agent read that as a hard fact rather than a failed search.

**What this does not establish.** The corpus is one user's notes, and the two
questions were chosen after seeing what was in it — which is a form of leakage:
I knew the PowerShell memories existed before asking about them. A blind set of
questions, or a corpus where the answer is genuinely unknown to the asker, would
be a stronger test. It also says nothing about latency under a thousand
concurrent queries, or about a corpus an order of magnitude larger.

**Incidental finding.** `is_mojibake` caught 1 row here against 34 in the pruning
probe. The earlier count came from a smaller sample with a different filter path;
neither number is authoritative. The detector matches common GBK-misread Han
characters and is a heuristic, not a validator — worth remembering before quoting
a "34 mojibake rows" figure again.

---

## D-034 The MCP config pointed at a database that did not exist

**Context.** Switching the machine's memory from nmem to b4a. Before editing the
config I checked where the two servers actually read from.

**Finding.** `mcp.json` set `BRAIN4AGENT_DB` to a path that did not exist. The
2893 imported memories were in a different file, and the configured one had
never been created. So every b4a tool call since the import had been creating an
empty database at that path and answering from it. The import in D-033 was real,
and the memories were real, but the MCP server was not reading them.

**Why it survived a session.** The empty store answers every query honestly:
zero results, `corpus_missing` listing the terms. An empty store and a store
that has never heard of something produce the same evidence. D-031 found that
two memory servers confuse tool selection; this is the same failure one layer
down -- the server was selected correctly and read the wrong file.

**Fix.** The configured path is now the main store, copied from the import file,
and `BRAIN4AGENT_DB` points at it. The import file stays untouched as the
experiment corpus for `bench/nmem/` and as a rollback point.

**Migration policy.** nmem is deprecated. Its 2893 usable fibers are imported
into b4a, and `neural-memory` is set to `disabled: true` in `mcp.json` so it is
no longer loaded into a session at all. The earlier wording here described it as
read-only history kept for a month as a rollback path; that was superseded.

Why disable rather than keep alongside: D-031 measured what two memory servers
do to tool selection, and this entry measured what one misconfigured server does
to answers. A memory source that is stale, complete in b4a, and still loadable
adds no capability and one more way to be wrong. Where b4a lacks something --
cross-session aggregation, recap-style summaries -- the answer is that it lacks
it, not that nmem should cover the gap.

The import file stays on disk as the corpus for `bench/nmem/` and as a rollback
artifact. It is not a runtime dependency and no code path reads it.

**Verified.** `b4a search` for a memory known to be in the import returns three
sessions from the main store; the MCP handshake reports the configured path and
lists `b4a_remember`, `b4a_recall`, `b4a_ask`, `b4a_session`, `b4a_stats`.

---

## D-035 The evidence tests cover the shipping implementation

**Date:** 2026-11-09
**Status:** Done

**Context.** D-015 left one exception on the record: `bench/zh/evidence.py`
kept its own `search`, because it took the benchmark's `MemoryItem` rather than
the store's `Memory`. The note said merging would move more code than it saved.

**Finding.** The exception outlived its reason. `bench/zh/e2e.py` now builds a
`MemoryStore` and calls `store.search`; `missing_terms_signal.py` reuses that
store. A repository-wide grep found exactly one importer of `bench.zh.evidence`:
`tests/test_evidence.py`. The file no benchmark script called was the file the
evidence tests covered, so the shipping `build_evidence` was only exercised
indirectly through the store and MCP tests.

**Decision.** Delete `bench/zh/evidence.py` and point `tests/test_evidence.py`
at `brain4agent.evidence` and `MemoryStore.search`. No benchmark script changes,
and no recorded number moves: nothing in `bench/` imported the deleted module.

**Consequences.** The tests now cover `question_terms`, per-session
`missing_terms`, `corpus_missing`, `coverage` bounds, `top_k`, session-level
evidence and score ordering against the code that ships. The bench copy's
`term_in_corpus` helper has no direct counterpart; `corpus_missing` is the
shipping expression of the same question.

**Not done.** No `MemoryItem` adapter. There is no caller left to justify one;
if a future benchmark needs to score a list of turns without a store, adding the
adapter then is cheaper than keeping a second implementation now.

---

## D-036 Access counts belong to the turn that matched

**Date:** 2026-11-09
**Status:** Done

**Context.** A review of PR #1 found that `_annotate` incremented
`Evidence.memory.id`, while `Evidence.memory` is the session's first turn.

**Finding.** Two-turn session, query matching only the second turn: after one
search `accesses` was 1 on the first turn and 0 on the turn that contained the
answer. The field is documented as "how many times a search has returned this
memory".

**Options.**

1. Count every turn in a returned session. Matches "the session is the unit of
   retrieval", but inflates the counter and credits turns that did not match.
2. Count only the turn that matched. Requires the evidence row to carry its id.

**Decision.** Option 2. The session is the retrieval unit, but the fact being
recorded is which memory row the retrieval is credited to. Reinforcement and
pruning are per-memory operations; crediting the opening turn puts both on the
wrong row. On a superseded hit the credit goes to the replacement, because that
is the row a caller can act on, even though the score came from the old one.

**Implementation.** `Evidence.matched_memory_id` is filled by `build_evidence`
and used by `_annotate`, with `memory.id` as the fallback for hand-built
evidence. The field is appended after `possible_duplicates` so the fifth
positional argument of `Evidence` keeps its meaning. No ranking change:
`accesses` still does not feed BM25.

**Verified.** `test_credits_the_matching_turn_not_the_first`: first turn 0,
matching turn 1. The existing single-turn access-count tests are unchanged.

## D-037 Query-side question words and session folding

**Date:** 2026-11-10
**Status:** Measured and shipped

**Context.** After D-019 the Chinese benchmark sat at hit@1 0.9250 / MRR 0.9404.
Two things had never been measured: whether question words belong in the query at
all, and whether a session's turns should be scored as one unit instead of taking
the best one. D-019 had tested a broader filter (`content`: every single-character
word and punctuation) and rejected it at 0.9000.

**Evidence.** `bench/zh/tuning_probe.py`, 40 answerable questions, one store.

```
query-side question words   hit@1 0.9500  hit@5 1.0000  MRR 0.9600
top-3 fold, beta 0.5        hit@1 0.9500  hit@5 1.0000  MRR 0.9613
```

1. **Question words are query-side noise.** Dropping 哪个/为什么/哪里/什么 from
the query moved hit@1 0.9250 → 0.9500 and MRR 0.9404 → 0.9600. The visible gain
is zh032, the only temporal question: 哪个 matched S29's "什么时候用哪个" while
the session that actually answers the question never contains it. This is
narrower than D-019's filter — 和/先/做 carry signal on this corpus, 哪个 does
not — and the two measurements are consistent rather than contradictory.

2. **A session is more than its best turn.** Folding the best three turns
(`max + 0.5 × next two`) added MRR 0.9613. Without the question-word filter the
same fold also lifted hit@1 from 0.9250 to 0.9500; the two overlap on zh032, and
the filter is the cheaper half. The beta curve is flat from 0.1 to 1.5 — 0.5 is
a point on a plateau, not a fitted constant.

3. **BM25's own parameters are flat.** A 25-cell k1/b grid spanned hit@1
0.9000–0.9250 and MRR 0.9294–0.9437, with every useful cell within 0.0025 of the
default. Recorded so it is not swept again.

**Shipped.** `tokenize.query_tokens()` drops interrogatives before matching;
`MemoryStore._fold_sessions` scores a session from its best three turns. The
interrogative list moved to `tokenize.py` so the query side and `corpus_missing`
share one definition instead of two that can drift.

**Cross-benchmark check.** LoCoMo moved hit@1 0.6270 → 0.6310 and MRR 0.7369 →
0.7402 from the fuzzy path in D-038; the question-word list is Chinese and does
not touch English queries. `bench/zh/missing_terms_signal` held at keep 0.875 /
refuse 0.632.

**Measured and rejected: LongMemEval on the store's tokenisers.** Wide index +
words + fuzzy scored 0.8596 hit@1 against 0.8660 for the harness's bigram/bigram,
and took 613 s against 78 s over the same 500 tasks. English already tokenises on
whitespace; the wide index and jieba add cost without a matching path to exploit.
The benchmark keeps bigrams.

**Measured and rejected: chunking LoCoMo sessions.** Chunks of 12 turns reached
hit@1 0.6434 against 0.6270 for whole sessions, but only with no session fold.
Under the product's fold every chunk size lost (0.6192 at 3 turns, 0.5911 at 8).
A benchmark setting cannot opt out of the product's scoring rule, so the idea is
recorded and not shipped. `bench/locomo/chunk_probe.py` reproduces it.

**Fallback.** Without jieba the query side is bigrams and the tokeniser split is
gone, but `_fold_sessions` still applies: measured 0.875 hit@1 against 0.825 for
the old fallback. The zero-dependency install is weaker, not broken.

**What this does not fix.** zh001 (rank 4) and zh013 (rank 5) remain wrong. Both
need the reader's judgement, not a better bag of words: zh001 asks why a search
failed, and the corpus contains both the symptom session and the root-cause
session; zh013 asks what is hard about abstention, and the corpus contains a
session about why abstention matters. D-020's limit is unchanged: a dense
same-topic corpus still collapses hit@1 to zero.

## D-038 Sizing the result set and matching one edit away

**Date:** 2026-11-10
**Status:** Measured and shipped

**Context.** D-026 left PrecisionMemBench at 34/43 active and 64/77 total and
named the residue: ordering and fuzzy matching ("k9s" for k8s, "mango" for
MongoDB). The harness also carried a rule that forced `top_k=1` for queries
shorter than eight words.

**Evidence.** `bench/pmb/run.py`, one change at a time, 43 active cases.

```
baseline (D-026)                       34/43 active   64/77 total
top_k from visible names, any length   36/43          66/77
whole-word name matching               38/43          68/77
one-edit fuzzy matching                42/43          72/77
```

1. **The length cut was truncating real multi-entity questions.** "What's Redis
doing?" names two beliefs that share the alias Redis; "explicit error returns in
pipe transforms" names two in six words. Counting names at every length fixed
both. Counting every belief regardless of the case's scope made it worse
(32/43): "Redis" matched a belief the tag filter would exclude, over-sized
top_k, and let noise in. The rule now counts only beliefs the case's scope can
see.
2. **Substring matching counted words the query did not say.** "pipe" inside
"pipeline" made b-functional-pipeline look named by a Kubernetes question;
"work" inside "working" did the same. Whole-word matching fixed both.
3. **One-edit matching is the last rule-shaped win.** An English word the corpus
has never contained is also matched against a corpus word one edit away: a
substitution, an insertion/deletion, or an adjacent transposition. That covers
`typscript` → `typescript`, `typsecript` → `typescript`, `mongod` → `mongodb` and
`mongoose` → `moongoose`. Four cases, +4 active.

**Shipped.** `BM25(fuzzy=True)`, armed by `MemoryStore`. Guards: ASCII letters
only, at least six letters, and only when the word is absent from the corpus.
Those guards are the difference between a spelling correction and rewriting half
of an English query (`test` → `tests` is a legitimate edit and not a misspelling).
The candidate with the highest document frequency wins, so the more common
spelling is chosen when two are one edit away.

**Cost.** The fuzzy vocabulary is built once per index from ASCII alphabetic
terms; a Chinese query never enters the loop. The candidate scan is linear in
that vocabulary and runs only for a query word absent from the corpus --
sub-millisecond at PMB's 35 beliefs and LoCoMo's 272 sessions, and it would need
length bucketing before a 10⁵-term vocabulary. That is a stated boundary, not a
measured one. LoCoMo gained 0.004 hit@1 and 0.003 MRR from the same path, so the
English effect is not PMB-specific.

**What remains.** Five cases. `filter-resolved-at-excluded` and
`scope-bleed-questions-code-excludes-writing` need a relevance verdict that
D-008/D-012 measured as unavailable without a model. `cap-stress` is an ordering
failure, not a selection one. `cold-start-user-no-beliefs` is the harness: it
scores every case as `test-user`, while the fixture expects a `brand-new-user`
empty context. The benchmark's own reference implementation still scores 43/43.

## D-039 Agent queries are keywords, not questions

**Date:** 2026-11-10
**Status:** Measured; one evidence field shipped, ranking unchanged

**Context.** D-037 optimised for the Chinese benchmark's questions
("中文查询为什么搜不到记忆？"). A user pointed out that an agent recalling a fact
does not ask a question — it types two or three keywords, often with an
identifier: "gh-oauth 403 上游 PR". If that is the real query distribution, the
benchmark may be measuring the wrong thing.

**Experiment.** Agent-style queries built from the real 2898-memory store: take a
memory's first distinctive nouns and identifiers, ask with two, three, or four of
them, and check whether the memory comes back. 300 queries, 100 per length.

```
queries        old code        new code (D-037/D-038)
2 words        hit@1 0.270     hit@1 0.270
3 words        hit@1 0.530     hit@1 0.530
4 words        hit@1 0.610     hit@1 0.610
```

**Finding 1: the D-037/D-038 changes do nothing for keyword queries.** No
interrogatives to drop, one memory per session so folding is a no-op, and the
keywords come from the corpus so nothing is misspelt. The improvements are real
for the benchmarks they were measured on; they are not improvements for an agent
typing keywords. Both facts can be true at once, and the README should not imply
otherwise.

**Finding 2: two-word queries fail on information, not ranking.** Gold position
for the 100 two-word queries: rank 1: 27, 2-5: 29, 6-10: 7, 11-50: 16, 51-200: 17,
missing: 4. Over a third of the gold memories sit outside the top ten. Coverage
re-ranking (`score * (1 + a * coverage)`) and MMR-style diversity were both
measured: three- and four-word hit@1 moved 0.53 -> 0.55 and 0.61 -> 0.63,
two-word hit@1 did not move at all. Neither is shipped.

**Finding 3: the failures are ambiguous queries, not retrieval bugs.** "GEO AI"
returns one GEO memory instead of another; "Reverse Web" returns the server
runbook instead of the character-art note. Both are correct matches to a query
that named a project and nothing else. The right response is a wider result set,
which the store already provides.

**What shipped.** `SearchResult.term_frequencies`: how many sessions each query
term appears in. Measured on the same store:

```
GEO AI                     geo=453  ai=250
gh-oauth 403 上游 PR       oauth=2  403=6  gh=90  pr=557
SpikeBot 005 Codespaces    codespaces=4  spikebot=137  005=14
```

The first query cannot be ranked better; it needs a third word. The second is
already precise. The evidence says which, without a model and without a verdict.
This is D-011 applied to the query itself: report the fact, let the caller decide
whether to rephrase.

**What remains.** The real store's `tags` are broken by the nmem import — all
2898 rows carry a literal `nmem-tag:["build" ...]` string instead of structured
labels, so scope filtering cannot help. Repairing that is a data migration, not a
retrieval change, and it needs the owner's approval.

## D-040 Structure at write time, evidence at read time

**Date:** 2026-11-10
**Status:** Shipped; ranking unchanged

**Context.** D-039 measured the real query distribution: two keywords, no
question, 2898 memories, hit@1 0.27. Coverage re-ranking and MMR both failed to
move it, because the failure is not ranking — the query does not name enough to
separate same-topic memories. Two things were missing: the memories had no
structured labels, and the caller had no evidence about which words would help.

**Write side.** `b4a_remember` now adds `date:`, `kind:` and `project:` tags
automatically. `project` comes from the `project` argument or the
`BRAIN4AGENT_PROJECT` environment variable; `kind` is stored both as a column and
a tag, so a tag filter can select decisions or errors without a separate index.
The store's own `add()` is unchanged: a library caller still gets exactly the
tags it passed, and only the agent-facing MCP tool adds the labels. That
boundary matters — automatic labels are a convenience for an agent, not a
silent rewrite of the storage API.

**Read side.** `b4a_recall` rows now carry `tags`, `created_at` and `kind` from
the turn that matched, and `SearchResult.related_terms` lists up to five words
the top sessions use that the query did not. That is the third word a two-word
query needs, taken from the corpus rather than guessed.

**Cost.** Related terms are computed from the top two sessions, capped at 300
characters and cached by text. Warm query latency on the Chinese fixture is
1.9 ms. Re-segmenting on every query measured 24 ms, which is why the cache
exists; the cache is keyed by text, not session id, so a changed session cannot
reuse a stale entry.

**Real-store probe.** `bench/real/probe.py` derives keyword queries from a store
and reports hit@1 by query length. On the local 2898-memory store: 2 words
0.275, 3 words 0.525, 4 words 0.650. Run it on a copy — `search` increments
access counters, so even a read writes. No memory text and no fixture is
committed.

**Not changed.** Ranking. The D-037/D-038 order stands; this entry adds what the
caller knows and what the store can say, not a new scorer.

**What remains.** The local store's tags are free-form (`trms`, `geo_3.0`,
`bug`). Nothing enforces a vocabulary, and nothing maps a new memory to an
existing project tag. A schema on `kind` remains unbuilt, as the README says.

## D-041 English query folding: stopwords, stems, and adjacency

**Date:** 2026-11-10
**Status:** Shipped. Chinese unchanged, LongMemEval unchanged, LoCoMo and
PrecisionMemBench up.

**Context.** LoCoMo was the weakest published number: hit@1 0.6310 over 1531
answerable questions. A rank histogram of the misses showed the shape clearly:
966 questions already rank the gold session first, then 200 rank it second, 105
third. It is mostly not a recall problem -- the answer is in the candidate set
and loses the top slot by a hair. Two of the failures sampled showed the same
mechanism twice:

- The question's words were mostly grammar. `When did Melanie sign up for a
  pottery class?` spends five of nine terms on `when/did/for/a` (plus the
  trailing `?`), none of which separates one session from another.
- The answer used a different inflection. `Where has Melanie camped?` asked in
  the past tense while the memory said `camping`; without folding they are two
  different tokens, so only one side of the question could match.

Chinese already had both fixes in a narrower form: interrogatives are dropped
from the query side (D-037), and character bigrams make every inflection
question moot. English had neither.

**What changed.** Three mechanisms, all model-free, all in `tokenize.py`:

1. **English stopwords on the query side only.** `EN_STOPWORDS` -- articles,
   auxiliaries, pronouns, prepositions, question words. The index keeps them: a
   memory containing `the` does not become more precise by hiding it.
2. **English inflection folding on both sides.** `stem_term` is a handful of
   suffix rules (`ies -> y`, `sses/shes/ches -> ss/sh/ch`, `ing`, `ed`, plural
   `s`), applied to ASCII alphabetic terms of four letters or more. It is not a
   linguist's stemmer; consistency is the goal, and both sides run it, so a
   crude stem that is wrong in the same way still matches. Applied at *index
   time* through `BM25(normalize=...)` rather than baked into the persisted
   token column, so an existing database needs no migration: stored tokens stay
   verbatim and are folded when the in-memory index is built.
3. **Adjacency bonus for ASCII word pairs.** A document containing two adjacent
   query terms in the same order earns `0.3 * (idf(a) + idf(b))`. A name like
   `pottery class` or `mental health` is one idea; BM25 discards the adjacency.
   Only ASCII alphabetic pairs are collected, which keeps the pair index small
   and makes the Chinese path a no-op by construction -- Chinese bigrams already
   index every adjacent character pair.

**Measured.** LoCoMo, through `bench.locomo.run`, one change at a time:

| configuration | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| before (D-037/D-038 state) | 0.6310 | 0.8870 | 0.9484 | 0.7402 |
| + English stopwords | 0.6532 | 0.8916 | 0.9425 | 0.7552 |
| + stem folding | 0.6525 | 0.8935 | 0.9556 | 0.7550 |
| stopwords + stems | **0.6702** | **0.9020** | **0.9556** | **0.7712** |
| + phrase bonus 0.3 | **0.6826** | 0.9020 | 0.9549 | **0.7784** |

The two filters interact: stopwords alone give back 0.006 of hit@10, stems
alone give back 0.013 of hit@10 and 0.015 of hit@5. Together they beat either
on every column. Hit@10 gave up one question to the phrase bonus (0.9556 ->
0.9549); it is 0.0065 above where this benchmark started.

Different corpora, different jobs, so the two implementations differ:

- Chinese retrieval is **unchanged** on every metric: 0.9500 / 1.0000 / 1.0000
  / 0.9613. The stopword list is English; the stemmer refuses non-ASCII; the
  pair index refuses non-ASCII. Measured, not hoped for.
- PrecisionMemBench **72/77 -> 74/77** total (active stays 42/43). Dropping
  `what/the/in` from the query made two scope-bleed cases fall out of the result
  set, which is the benchmark agreeing with D-025: a query's grammar is not a
  memory. `active` did not move because the two cases it gained were not
  retrieval-dependent.
- LongMemEval is **unchanged**: 0.8660 / 0.9511 / 0.9660 / 0.8990. Its harness
  uses `tokenize` directly, and that function was not touched; the folding lives
  in `BM25.normalize` and `query_tokens`.
- The real store (2,903 memories, D-040's leave-one-out keyword probe, 300
  queries): hit@1 0.460 -> 0.470, hit@5 0.650 -> 0.653, MRR 0.544 -> 0.551;
  two-keyword queries 0.27 -> 0.29. The queries an agent actually types are
  already nouns, so the stopword list does little there and nothing regressed.
  Measured on a copy of the store, since `search` writes access counters.

**Cost.** The pair index roughly doubles rebuild time on the LoCoMo fixture,
which is 272 documents of long conversation:

```
                    rebuild    query
phrase bonus off    267 ms     10.2 ms
phrase bonus on     520 ms     10.2 ms
```

Rebuild happens once per process or after a write, not per query, and the query
path is unchanged. For a 44-session Chinese corpus rebuild is 8.7 ms.

**Rejected.** Measured on the same fixture, discarded:

| change | hit@1 | verdict |
|---|---|---|
| BM25L length compensation (delta 0.5) | 0.6715 | +0.0013, noise |
| coverage re-rank (multiply by 1 + a(cov-1)) | 0.6688-0.6702 | flat or worse |
| distinct-term-count re-rank | 0.6728 | +0.0026, below the phrase bonus |

Coverage re-ranking failing is the third such result (D-020, D-024, D-039). It
says the same thing each time: BM25's order already contains the information a
post-hoc coverage term would add. The phrase bonus is different in kind -- it
recovers an ordering fact BM25 threw away, not a weighted copy of what it used.

**Caveats.**

- The stemmer is crude on purpose. `analysis -> analysi`, `boxes -> boxe`; both
  sides agree, so matching works and the token is not something a human reads.
- The stopword list is hand-written and English. A domain where `before` or
  `under` carries meaning would want a different list; none of the three
  benchmarks does, and LoCoMo's questions are ordinary conversation.
- The phrase bonus needs its pair index. It is off by default in `BM25` and on
  in `MemoryStore`, where the measured cost is paid once per rebuild.
- Every number above is retrieval, not answer quality. Nothing here changes the
  claim in D-011: the store reports evidence and the caller decides.

## D-042 Proximity: where the query words sit, not just whether they appear

**Date:** 2026-11-11
**Status:** Shipped. LoCoMo up, everything else measured unchanged.

**Context.** D-041 left LoCoMo at hit@1 0.6826 over 1531 answerable questions,
the weakest published number. A rank histogram of the misses (`bench/locomo/diagnose.py`)
narrowed it: 1045 questions already put the gold session first, **188 put it
second**, 75 third, and only 69 fell outside the top ten. On those second places
the gold and the winner had nearly the same coverage -- median 0.508 against
0.533 -- and scores within a few percent:

```
When did Caroline go to the LGBTQ conference?   gold rank 2
  session_5  18.65  coverage 0.96   <- returned
  session_7  18.15  coverage 0.96   <- gold
```

That is not a missing-word problem. Both sessions contain the question's words.
It is a *placement* problem: one conversation is about the conference, the other
mentions the same words in passing, and a bag of words cannot tell them apart.

**What changed.** One signal, `window_proximity` in `tokenize.py`: the fraction
of distinct query terms that share a window of 40 index tokens. Applied as
`score * (1 + alpha * proximity)`, so alpha = 0 is exactly the previous score.

The distinction from D-041's rejected coverage re-rank is the whole point. That
one asked *whether* the words are in the session and multiplied by a function of
the same BM25 already knows. This one asks *where they are* -- a fact the index
throws away -- and it is the third time in this project that recovering a
thrown-away ordering fact worked where reweighting known facts did not (the
phrase bonus in D-041 was the second, the read/write tokenizer split in D-019
the first).

**Measured.** `bench/locomo/proximity_sweep.py`, 5x6 grid over window and alpha.
Each cell re-ranks the store's own top-10 for that query, so the fold and the tag
filter are in the loop, but the store itself re-ranks all 272 candidates. That
difference is worth 0.0020, and it matters for how the grid is read -- see the
attribution note below the table:

| window | a=0.25 | a=0.5 | a=1 | a=1.5 | a=2 | a=3 |
|---|---|---|---|---|---|---|
| 20 | 0.6995 | 0.7002 | 0.7041 | 0.7002 | 0.7002 | 0.7022 |
| 30 | 0.6963 | 0.6995 | 0.7054 | 0.7061 | 0.7061 | 0.7080 |
| 40 | 0.6950 | 0.6989 | 0.7061 | **0.7080** | 0.7074 | 0.7067 |
| 60 | 0.6917 | 0.6976 | 0.7022 | 0.7061 | 0.7067 | 0.7061 |
| 80 | 0.6891 | 0.6969 | 0.6995 | 0.7015 | 0.7008 | 0.7008 |

Baseline 0.6826. **Every cell is above it**, and the weakest by a clear margin.

The block window 30-60 x alpha 1-3 is 12 cells spanning 0.7022 to 0.7080 -- a
0.0058 band whose floor is still 0.0196 above the baseline. `window=40,
alpha=1.5` is a point inside that band, tied for the top with `window=30,
alpha=3`, and the per-alpha spreads inside the block run 0.0013 to 0.0039. So
the exact value is a rounding decision rather than a fitted peak, which is the
rule D-037 applied to `SESSION_FOLD_BETA`. An earlier draft of this paragraph
said "30-60 all within 0.003", which the table does not support -- the alpha=1
column spans 0.0039.

**Three guards, each from a measurement that failed without it.** The full grid
above used a fixed window on ASCII words and would have shipped a regression on
the other two benchmarks if it had stopped there.

1. **Only ASCII words.** A Chinese query has no ASCII content words, so the
   signal cannot fire and Chinese is unchanged by construction -- the same scope
   rule, and the same structural reason, as the phrase index in D-041: Chinese
   indexes character bigrams, so adjacency is already in the index.
   Measured before the guard: Chinese hit@5 1.0000 -> 0.9750 and MRR
   0.9613 -> 0.9598. The losing question was `zh013` (`拒答难在哪里？`), where
   **every** candidate scored proximity 0.50 -- the query reduces to one content
   word plus a particle -- so the factor separated nothing while perturbing the
   turn-level fold. Chinese is now 0.9500 / 1.0000 / 1.0000 / 0.9613, unchanged.
2. **Documents under three windows are skipped.** A one-sentence belief has no
   passage, so proximity degenerates into document coverage while keeping a
   multiplier large enough to outvote BM25. Measured before the guard:
   PrecisionMemBench 74/77 -> 73/77, losing `negative-guidance-mongoose`, where
   a scope-bleed belief outranked the counter-signal one. With the guard every
   PMB belief is below the floor and PMB is 74/77, active 42/43, unchanged.
3. **Queries with fewer than five ASCII terms are skipped.** `bench/real/ab_probe.py`
   on a full snapshot of the real store (2,910 memories, 300 leave-one-out
   keyword queries): hit@1 **0.4867 -> 0.4067**, and every bucket fell -- 2-word
   0.230 -> 0.190, 3-word 0.580 -> 0.490, 4-word 0.650 -> 0.540. Two to four
   words is a lookup, not a question (D-039); proximity over two words has three
   possible values and is noise. `bench/term_count_probe.py` shows the boundary
   is clean: the real store's queries are 0% at five terms, LoCoMo's median is
   five (64.7% at five or more), PrecisionMemBench's is three.

   The stronger statement is the one that matters: **with the guard the store's
   retrieval is bit-identical to having the signal off entirely** --

   ```
   configuration      hit@1    hit@5    MRR      2-word  3-word  4-word
   proximity off      0.4867   0.6500   0.5584   0.230   0.580   0.650
   shipped (floor 5)  0.4867   0.6500   0.5584   0.230   0.580   0.650
   floor off          0.4067   0.6400   0.5035   0.190   0.490   0.540
   ```

   Every column of the shipped row equals the off row. On a corpus of keyword
   queries the feature costs nothing, which is what a guard is for.

   **Two of the three guards are now re-measurable, not just described.**
   `bench.guard_attribution_probe` switches the two constant-backed guards off
   and reports the benchmark each protects:

   ```
   guard             benchmark          shipped   guard off
   length floor      PMB pass             74/77       73/77
   term floor        LoCoMo hit@1        0.7054      0.7100
   term floor        real store hit@1    0.4867      0.4067
   ```

   The third guard is the ASCII filter on the query, which is a code path
   rather than a constant and so has no switch. Its cost is the measurement
   in item 1 above; reproducing it means deleting the filter.

   The real-store row has moved twice, both times for reasons worth recording.
   It first read 0.4433, measured before the ASCII filter existed; then 0.4267,
   after that filter but on a snapshot missing 13 memories (see the cost section);
   now 0.4067 on a complete snapshot. Each number was right for the code and data
   it was taken on, and none of them was right for the code and data that ship.
   A guard whose cost cannot be re-run is a story, which is why the switch
   exists. The ASCII filter
   is a code path rather than a constant and still has no switch; its
   measurement above is the record.

**Shipped numbers.**

| benchmark | metric | before | after | |
|---|---|---|---|---|
| LoCoMo | hit@1 | 0.6826 | **0.7054** | +0.0228 |
| LoCoMo | hit@5 | 0.9020 | **0.9086** | +0.0066 |
| LoCoMo | hit@10 | 0.9549 | 0.9549 | flat |
| LoCoMo | MRR | 0.7784 | **0.7921** | +0.0137 |
| Chinese | all four | 0.9500/1.0000/1.0000/0.9613 | identical | unchanged |
| PrecisionMemBench | pass / active | 74/77, 42/43 | identical | unchanged |
| LongMemEval-s | all four | 0.8660/0.9511/0.9660/0.8990 | identical | unchanged |
| real store | hit@1 | 0.4867 | identical | unchanged |

The shipped number is 0.7054, and neither the grid nor the first version of this
entry had it right. The grid re-ranks a top-10 window per query -- enough to
compare signals, not enough to price the configuration -- and had no term floor.
`bench/locomo/floor_attribution_probe.py` runs both through the store, which
re-ranks all 272 candidates:

```
                                             hit@1    hit@5   hit@10
no proximity (D-041 baseline)               0.6826   0.9020   0.9549
proximity, no term floor                    0.7100   0.9079   0.9556
proximity, term floor 5 (shipped)           0.7054   0.9086   0.9549
```

So the grid *understated* the signal by 0.0020 (top-10 window versus the whole
corpus) and the term floor costs **0.0046**, not the 0.0026 an earlier version
of this entry claimed from subtracting 0.7054 from the grid's 0.7080. That
subtraction was a guess between two differently-measured numbers, which is
exactly the error D-015 exists to prevent. Paying 0.0046 to leave the real
store's 0.4867 intact is still the right trade, and it is now a measured one.

By category, proximity helps where a question names an event and hurts where it
names a fact: open-domain 0.7586 -> 0.7919, temporal 0.6656 -> 0.6906,
multi-hop 0.4270 -> 0.4494, **single-hop 0.5552 -> 0.5445**. Single-hop is a
short factual lookup inside one session; the same boundary the term floor
encodes. The category table is published as measured, net positive.

**Cost.** `bench/locomo/cost_probe.py`, `bench/real/ab_probe.py` and
`bench/real/memory_probe.py`, three repeats each on an otherwise idle machine:

```
                              rebuild      query
LoCoMo (272 long sessions)    486-543 ms   12.2-13.2 ms   without proximity
                              474-631 ms   20.6-21.1 ms   with proximity  (~1.7x)
real store (2,910 short turns)   -           4.8-5.0 ms    either way (noise)
```

The rebuild column does not move because the sequence cache is a list append
alongside the `Counter` that was already being built. The query column nearly
doubles on LoCoMo and does not move on the real store, which is the same
boundary the guards draw: every LoCoMo session is 500-1100 index tokens and
clears the three-window floor, and only 9.8% of real memories do.

An earlier version of this table said 14.8 -> 46.9 ms. That was measured while a
reviewer was running benchmarks on the same box, and a 3.5x claim would have
been wrong by a factor of two. Repeats are cheap; the number is not.

Memory for the sequence cache on the real store: **+2.1 MB** (36.04 -> 38.15 MB
of index, 233,241 tokens over 2,910 documents). The tokens are references to the
strings the term counters already hold, not copies, so the delta is the list
slots and their overhead.

An earlier version of this paragraph said 2,897 documents for a stored 2,910,
and the discrepancy was a real bug rather than rounding: the probes copied the
database with `shutil.copy2`, which takes the main file and leaves the
write-ahead log behind. Thirteen committed memories lived only in the `-wal`.
`bench.real.probe.snapshot()` now copies through SQLite's backup API, which reads
the log, and every probe uses it. The retrieval numbers moved with the fix
(0.4700 -> 0.4867 on the same seed) not because those 13 memories were special
but because `build_queries` samples its queries from the store, so a different
snapshot is a different query set. Treat the real-store baseline as a property of
a snapshot and re-measure it, rather than as a constant.

The LoCoMo fixture is 272 entire conversations at 500-1100 index tokens each,
so every session clears the three-window floor and every question clears the
term floor. The real store is the other shape: median 63 index tokens, **9.8%
of memories above the floor**, and no query long enough to fire the signal. The
cost lands where the benefit is and nowhere else.

**Pruning: written, measured, removed.** The first version carried two bounds --
one from the terms a document contains at all, one from the maximum possible
boost -- and both were wrong in ways that returned fewer results than asked for:

- `>` where `>=` was needed meant `pool` was only truncated once it held
  `top_k + 1` entries, so the frontier was never assigned. It stayed `inf`, every
  later document passed the per-document test and was skipped, and `top_k=1`
  returned one document whatever the corpus held.
- `<=` where `<` was needed dropped documents that tie the k-th place. The final
  order breaks ties by insertion order, which is uncorrelated with the score, so
  a document whose best possible score equals the current k-th score can still
  belong ahead of it.

Both were caught in review, after a first set of tests passed them. That is the
part worth recording: the tests asserted "small `top_k` matches a full walk" over
documents that were 35-69 tokens, below the three-window floor, so proximity
never ran and all three alphas produced one ranking. A test that cannot reach the
code it names is worse than no test, because it reports success.

Fixing the comparisons was not the end. `bench/locomo.pruning_effect_probe`
instrumented `window_proximity` and counted calls: **271.6 per query out of 272
candidates, in both call shapes**. Neither bound fires at the shipped alpha. The
ceiling is 2.5x, so no realistic gap between neighbouring scores clears it, and
LoCoMo's sessions carry most of a question's terms, so the per-document bound is
near 1 as well. Sixty lines of bound arithmetic that never fire, and that had
already produced two wrong-answer bugs, is a worse trade than 8 ms -- so the
bounds are gone and `_apply_proximity` is a scoring loop and a sort.

**The tag filter does the work twice, and that is a real cost not fixed here.**
`MemoryStore.search` ranks the whole corpus and filters afterwards, because the
scope filter has to see the full ranking before truncating (D-025). On LoCoMo
that means every query scores proximity on all 272 sessions when only ~27 belong
to the sample it asked about -- about 90% of the work is on documents the caller
asked to exclude. Pushing the filter into the ranker would fix it; that changes
the `top_k` contract and is a separate decision.

**Rejected.** `bench/locomo/rerank_probe.py`, same fixture, each on top of the
published configuration:

| signal | best hit@1 | verdict |
|---|---|---|
| min-span (shortest window holding every term) | 0.6963 | +0.014, below the 40-token window |
| early-position (first match near the top) | 0.6826 | no effect at any alpha |
| window 100 tokens | 0.6982 | worse than 40 |

All three rows come from `bench/locomo/rerank_probe.py`, which takes its
`window_proximity` from the shipping tokeniser rather than reimplementing it.

`early-position` is recorded because it was the most plausible of the three and
did nothing: a session's matching passage is as often in the middle as at the
start.

A fourth variant was tried and is **not** in the table. Scaling the window as a
fraction of document length instead of fixing it in tokens scored 0.6937 and was
rejected -- but the parameter was removed with the rest of the pruning work, so
that number can no longer be reproduced. It is a note, not a result; anyone
reviving the idea should re-measure rather than cite 0.6937.

**Caveats.**

- The signal assumes a question names more than one thing. It is turned off
  below five ASCII terms, which means a long English question with a small
  vocabulary gets no proximity at all -- correct by the measurement, invisible
  to a benchmark that never asks one.
- The three guards are thresholds, and thresholds are where a future corpus will
  disagree. All three are now constants with the measurement attached
  (`PROXIMITY_WINDOW`, `PROXIMITY_BONUS`, `PROXIMITY_MIN_TERMS`,
  `PROXIMITY_MIN_DOC_TOKENS`, where 0 means three windows) rather than derived,
  so changing one is a decision with a number behind it -- and
  `bench.guard_attribution_probe` re-measures two of them on demand. The ASCII
  filter is a code path, not a constant, and has no switch.
- **A test that cannot reach the code it names reports success.** This file's
  first pruning tests asserted "small `top_k` matches a full walk" over 35-69
  token documents, below the 120-token floor, so proximity never ran and three
  alphas produced one identical ranking. The guard tests for the term floor and
  the ASCII filter had the same shape: deleting the guard they were named after
  left them green. Every guard test is now arranged so removing the guard flips
  its result, and that was checked by removing each one.
- Windows are counted in index tokens. Chinese indexes roughly two tokens per
  character, so a window that means "one passage" in English would mean "two
  sentences" there. The ASCII guard makes that moot rather than solving it: a
  Chinese proximity signal would need its own window, measured separately.
- **`Evidence.score` is no longer a BM25 score.** It is BM25 scaled by up to
  `(1 + PROXIMITY_BONUS)`, and whether that factor applies depends on the query's
  length, so the same session can carry a different score for two differently
  shaped questions about it. Ordering and within-result ratios are still valid;
  absolute thresholds and cross-query comparisons were already invalid (D-008)
  and are now more so. The field says this where it is declared.
  **`b4a_recall` returns it to clients** (`mcp_server.py`), rounded to three
  decimals and with no indication of scale, so an agent comparing a score from
  before this change with one from after is comparing two different units. The
  same warning already applied to the phrase bonus in D-041; this change widens
  it by a factor of up to 2.5 and deserves saying plainly rather than being left
  to be discovered.
- **A probe must isolate a bonus the store already ships.** `proximity_sweep`,
  `rerank_probe` and `diagnose` all read `store.search()` scores and apply a
  signal of their own. Once `PROXIMITY_BONUS` went live, the alpha=0 row stopped
  reproducing 0.6826 and started reporting the shipped result -- the probes were
  measuring their own output twice. `bench.retrieval.isolated_bonus` exists for
  this and all three use it. This is the D-015 failure in a new shape: the probe
  has to measure the shipping code, and it also has to *control* it.

## D-043 SQLite is a consequence of two constraints, not a comparison

**Date:** 2026-11-13
**Status:** In force. Revisit the index layer, not the database, when a local
write costs more than a second before the next query is answerable.

**Context.** After 42 entries, none records why the store is SQLite. It arrived
as the intersection of two earlier decisions -- D-004 (no dependency before a
measured win) and D-013 (position at "Chinese + no model + single file") -- and
was never compared against an alternative. The fair question is whether that was
reasoning or inertia from `neural-memory`. This entry answers with measurements
and states where the choice stops holding.

**The fact that decides most of it.** SQLite is not the retrieval engine. Rows
are read into an in-memory BM25 index, every search ranks that index, and the
only write a search makes is one transaction crediting the turns that matched
(`store.py`, `_ensure_fresh` and the `executemany` at the end of `search`). An
engine swap would therefore leave recall and ranking untouched. It would change
write cost, portability and the process model -- which is what is measured here.

**Measured.** `bench/storage/probe.py` against a 2,925-row snapshot of a real
store, repeated runs on an idle machine:

| | 2,925 rows |
|---|---|
| db file | 3.35 MB |
| index tokens (the `tokens` column) | 242,398 (83 per memory) |
| resident (whole store object graph) | 35.12 MB |
| resident per token | 145 bytes |
| resident per memory | 11.7 KB |
| full scan, 16 columns | 25-27 ms |
| open and build the index | 353-403 ms |
| **first search after a local write** | **402 ms** |
| query, 2-3 keywords | 3.8-4.6 ms |
| query, 40-character sentence | 20.7-25.8 ms |
| write per search (1 commit, 5 rows) | 0.12-0.17 ms |

**The row that matters is the one marked in bold.** `add()` sets `_dirty`; the
next `search()` calls `_ensure_fresh`, sees the flag and rebuilds the whole
index. A read-only benchmark never touches that path, so the first version of
this probe reported the open cost and left writes looking free. In an agent loop
that stores a turn and immediately asks a question -- the ordinary shape -- this
is the dominant latency, twice the open cost, and it has nothing to do with
SQLite: the cost is the in-memory index layer above it.

For the same reason `PRAGMA data_version` is not the 55us win an older docstring
in `store.py` implies. The pragma costs 6.4 us (measured, 200 samples), and when
it reports a sibling's commit it triggers the same full rebuild. A long-lived MCP
server and a CLI on one file therefore pay 400 ms on the first query after either
writes.

`resident` is the whole object graph, not an isolated index: `Memory` instances,
`_by_session`, `_by_id`, the per-session lowercase text, the term counters and
the sequence cache. That is the number competing for RAM, so it is the number a
budget should use; `tracemalloc` sees Python objects and not allocator overhead,
so treat it as a lower bound. The corpus is a live store: a re-run one memory
later reports 2,926 rows and 35.23 MB. These are properties of a snapshot (D-042
made the same point about the real-store baseline), not constants.

**Not attributable to proximity.** Keyword queries are ~5x faster than
40-character ones, and it is tempting to call that the proximity pass. It is not:
`PROXIMITY_MIN_TERMS` is 5, it counts only ASCII alphabetic terms
(`tokenize._phrasable`), and only 3 of 48 sampled queries clear that bar. The
difference is query length. The probe now prints the fired count so the two do
not get conflated again.

Linearity was checked on synthetic stores at 34 tokens per memory, shorter than
the real corpus:

| rows | resident | per token | open | first search after write |
|---|---|---|---|---|
| 3,000 | 17.5 MB | 169 B | 169 ms | 182 ms |
| 20,000 | 111.9 MB | 162 B | 1,289 ms | 1,424 ms |

Per-token cost moves 4% across a 6.7x range, and the rebuild scales with it. The
synthetic corpus is template-generated and far more repetitive than real
memories, so its query latency is pessimistic; its size and rebuild numbers are
the usable part.

**Projected.** At 83 tokens per memory and 145 bytes per token:

| memories | resident |
|---|---|
| 10,000 | 120 MB |
| 100,000 | 1.2 GB |
| 1,000,000 | 12 GB |

Read that against the rebuild measurements, not the file size: at 20k rows the
first query after a write already costs 1.4 s, which is where this design stops
being usable regardless of how much RAM is left. The two corpora do not convert
cleanly -- the real store packs 83 index tokens per memory against the synthetic
34 -- so the real store reaches a one-second rebuild nearer 7k memories than
20k. Re-measure rather than scale.

**An accounting gap, stated rather than smoothed.** D-042 reports 38.15 MB of
index over 233,241 tokens (163.6 bytes per token) for the same store. This entry
reports 35.12 MB over 242,398 tokens (145 bytes per token) on a later snapshot.
The two differ by 12% and the earlier measurement does not record its method, so
this entry does not claim D-042 was wrong; the extrapolation uses the numbers
from one run of one probe, and anyone comparing the two should re-measure rather
than subtract.

A second stale number turned up the same way. The README's rebuild column says
3.9 ms for 127 turns and 29 ms for 1,016; re-measured against one real store the
figures are **18.0, 101 and 393 ms** for 127, 1,016 and 2,925 turns. The table
predates the phrase pair index (D-041) and the sequence cache (D-042), both of
which live in the rebuild. The README now says so next to the table instead of
quietly carrying a number that contradicts the 402 ms above.

**Alternatives, each losing on a constraint rather than on taste.**

- **JSONL, or one file per session.** Loses transactions, the per-search commit
  that credits access counts, and cheap concurrent reads. Crash recovery becomes
  hand-rolled.
- **DuckDB.** An analysis engine with one writer, and not in the standard
  library. This workload is writes and point reads, not column scans.
- **LMDB / RocksDB.** No SQL, compiled dependency. Breaks the zero-dependency
  install that D-004 and `pyproject.toml` still hold to.
- **PostgreSQL / MySQL.** A service to install, a port and credentials to own.
  Breaks "single file" from D-013 and the `pip install` story in the README.
- **A vector store.** Breaks the zero-model core (D-004, `landscape.md`). At most
  an optional second retriever, never the store of record.

**Where SQLite earns its place, in shipping code rather than in principle.**

- `journal_mode=WAL` + `synchronous=NORMAL`: every search commits a write, and
  the default rollback journal would fsync per commit. Both are per connection;
  a bare `sqlite3.connect` defaults to `FULL`, which is why this probe sets it
  explicitly.
- `PRAGMA data_version`: two processes can hold the file, and a stale index is
  worse than a slow one. Cost 6.4 us, consequence 400 ms -- see above.
- `ORDER BY rowid`: write order, because `created_at` has second precision and
  ties are otherwise resolved arbitrarily.
- `backup()`: not on the hot path, but it is the only correct snapshot. A file
  copy drops the `-wal` and lost 13 committed memories (D-042);
  `bench.real.probe.snapshot` is where that lives.

**What the `neural-memory` lineage actually is.** The shape is shared; the
architecture is not. neural-memory's own brain file -- at the default location
`bench/nmem/import_to_b4a.py` already documents -- is also SQLite, with **37
application tables** next to the FTS5 shadow tables and **6 FTS5 triggers**
(`fibers_ai/ad/au`, `neurons_ai/ad/au`) whose bodies all call `cjk_spaced()`:
a user function that inserts spaces between Han characters at write time so
`unicode61` can index them. Counted from `sqlite_master` over a read-only URI;
the query is in the reproduce block below, because a number here is otherwise
unreproducible. The path is deliberately not repeated here -- the checker in
`tools/` refuses machine-local paths in published files, and it caught the
first draft of this entry.

`store.py` opens by rejecting that exact approach: FTS5's tokeniser cannot be
replaced from Python, `unicode61` treats a Han run as one token, and pre-spaced
text has to be kept in step with triggers. `bench/nmem/` exists because that
database is a real corpus with real recall history, not because it is a template.

**The gap this entry does not close.** "Single file" in D-013 was chosen because
it was the unoccupied position in `landscape.md`, not because it was measured as
the best option. No user has run this at 100k memories. The projection is
arithmetic on one 2,925-row corpus.

**Exit condition.** Revisit the index layer, not the database, when a local write
costs more than about a second before the next query can be answered. On this
machine and this corpus that lands near 7k memories: 402 ms at 2,925, scaled
proportionally. It is an extrapolation from one point, not a measurement, so
treat the crossover as approximate. It arrives well before the resident index
becomes a memory problem. The second trigger is non-technical: a requirement for
more than one writer process, which no measurement here addresses.

At that point the order is an on-disk inverted index or FTS5 with a compiled
tokeniser, with SQLite still holding text and metadata. Only a remote or
multi-writer deployment argues for a server database, and that is a different
product. This entry deliberately does not patch the rebuild: incremental index
maintenance is a design change with its own measurements, and it should not
arrive as a footnote to a storage choice.

**Reproduce.**

```bash
python -m bench.storage.probe --db <copy-of-store>
python -m bench.storage.probe --synth 20000
```

The `neural-memory` counts, read-only, without touching memory text. Point it
at that project's brain file, which is where `bench/nmem/` reads from:

```python
import sqlite3, sys

c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for typ in ("table", "trigger"):
    names = [n for (n,) in c.execute(
        "select name from sqlite_master where type=?", (typ,))]
    print(typ, len(names))
print([n for (n,) in c.execute(
    "select name from sqlite_master where type='trigger' and sql like '%cjk_spaced%'")])
```

Run it on an idle machine: an earlier draft of a D-042 latency number was
measured while benchmarks ran in parallel and was wrong by 2x.
