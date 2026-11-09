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

**Not done**

`bench/zh/evidence.py` keeps its own `search`, because it takes the benchmark's
`MemoryItem` rather than the store's `Memory`. Both `question_terms`
implementations agree but are written twice. Merging now would move more code
than it saves.

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
