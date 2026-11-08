# Landscape: Agent Memory Systems

Surveyed 2026-11-07. Star counts and activity verified through the GitHub API,
README files read directly, arXiv abstracts fetched per paper. Star counts are a
snapshot and will drift.

---

## Summary

1. **The field is crowded.** Tencent's project has 30k stars, `causal-memory` is
   at 104 and shipping weekly, and mem0 / Zep / Letta / Cognee each hold a niche.
2. **"No model + Chinese + local file" is a real gap.** Every leading project
   depends on a model: Tencent requires two sets of LLM parameters,
   `causal-memory` uses local ONNX embeddings, mem0 needs a vector store.
3. **Zero-model retrieval is sufficient.** BM25 over character bigrams reaches
   0.966 hit@10 on LongMemEval.
4. **Zero-model abstention is not.** Ceiling is 0.211 (13 features plus logistic
   regression, leave-one-out); a local LLM reaches 0.895. A factor of four.
5. **So the design is: retrieve without a model, leave the judgement to the
   caller.** The caller is an agent and already has a model.

---

## Comparison

### TencentDB-Agent-Memory

`TencentCloud/TencentDB-Agent-Memory` · **30,584 stars** · TypeScript · MIT

A team-level memory hub: conversation, documents, and code are organised into four
reusable asset types (Chat Memory / Skill / LLM-Wiki / Code-Graph) shared across
agents.

**Model dependency: mandatory.** Installation begins with
`$EDITOR .env  # Fill in two sets of LLM parameters (memory group + proxy group)`.

Architecture is three services — `memory-core`, `memory-hub`, `proxy` — integrated
by pointing an agent's base URL at it, with no plugin or MCP layer.

*It demonstrates both that demand is large and that the mainstream answer assumes
you have an LLM.*

### causal-memory

`JingxuanC/causal-memory` · 104 stars · Rust · Apache-2.0
Created 2026-07-26, still pushing as of 2026-11-04, 368 tests.

The closest overlap with this project's original scope:

- Single SQLite file, local-first
- MCP server, 14 tools
- Causal edges: `caused` +1.0 / `fact` +0.8 / `meta` +0.6 / `enabled` +0.5 /
  `prevented` **−0.3** (inhibitory, GABA analogue) / `no_effect` 0.0
- Hippocampal engine: CSR graph, spreading activation, SWR consolidation
  (LTP/LTD/GC), Q-values
- Hybrid retrieval: BM25 plus semantic cosine, fused with RRF
- **100% compression survival** against 45% for text

**Model dependency: local ONNX embeddings (offline).** Not an LLM, still a model.

Its benchmark split is revealing:

| benchmark | causal-memory | mem0 |
|---|---|---|
| LongMemEval-S | 76.4% @11.5K tok/q | 94.4% official / 73.8% independent |
| LoCoMo | 79.1% | 91.6% |
| CausalEval (own) | **78%** | 65% |

It says so itself: **"factual recall is not our strength"**; causality is.

*Implication:* do not rebuild the causal model. Its write path — raw turns into an
audit table, distilled output into facts and causal edges — is worth borrowing.

### Others

| project | shape | model dependency |
|---|---|---|
| mem0 | vector store + knowledge graph | embeddings (required) |
| Zep | temporal knowledge graph | embeddings |
| Letta (MemGPT) | three-tier memory, OS analogy | LLM (self-editing is the mechanism) |
| Cognee | local-first graph reasoning | embeddings |
| LangMem | LangChain ecosystem | optional |

---

## Abstention: why zero-model approaches fall short

The most time-consuming part of the survey, and the only hypothesis here with
hard data against it.

### Every signal tried, all measured

| signal | refusal at keep ≥ 0.95 | source |
|---|---|---|
| BM25 top-1 score | ~0 | this project |
| IDF-weighted coverage | ~0 | this project |
| unseen-word ratio (bigram) | ~0.05 | this project |
| unseen-word ratio (jieba) | 0.105 | this project |
| word-level coverage | ~0.10 | this project |
| NQC | 0.158 | Shtok et al. 2012 |
| UQC | 0.158 | same |
| WIG | 0.000 | Zhou & Croft 2007 |
| Clarity | 0.105 | Cronen-Townsend et al. 2002 |
| top1 − top10 margin | 0.105 | this project |
| **13 features + logistic regression + LOOCV** | **0.211** | this project |
| local LLM judgement | **0.895** | this project |

### The literature rejects QPP too

Query Performance Prediction is a mature IR subfield: predict retrieval quality
without looking at the answer. Its classical methods are purely statistical,
which is exactly the required shape.

They do not work here, and the literature expects that.

**arXiv:2504.01101** — *Uncovering the Limitations of Query Performance
Prediction* (Chifu, Déjean, Mothe, Garouani, Ortiz, Ullah, 2025-04)

Across ROBUST / GOV2 / WT10G / MS MARCO and BM25 / DFree / SPLADE / ColBERT:

- Predictors correlate poorly with actual performance
- Predictors are most sensitive to **collection**, then to retriever
- "QPP-driven selective query processing offers only marginal gains"

**Root cause:** QPP predicts **ranking quality**, not **whether an answer
exists**. A query can retrieve the single most relevant session perfectly while
that session contains no answer — which is exactly how abstention questions are
constructed (ask about a hamster when only a cat was mentioned).

### Why a classifier does not close the gap

Thirteen features reach 0.211 under leave-one-out cross-validation — better than
any single feature, still a quarter of a model's rate.

The strongest weight is `t_howmany` ("how many / how many times", +0.859):
questions asking for a specific number are more often unanswerable, because the
corpus covers one topic and has no such number.

That pattern cannot carry 0.9 accuracy. **Deciding whether 数据库 and 部署目录 name
the same concept requires semantic understanding, which counting cannot supply.**

---

## Where the gap is

| dimension | Tencent | causal-memory | mem0/Zep | available |
|---|---|---|---|---|
| Native Chinese segmentation | yes (Chinese product) | **no** | **no** | ✅ |
| No model | **no** (LLM) | **no** (embeddings) | **no** | ✅ |
| Single local file | partial (three services) | ✅ SQLite | no (vector store) | ✅ |
| Abstention | no | no | no | ⚠️ evidence only |
| Causal edges | no | ✅ strength | no | not worth duplicating |
| Compression survival | unknown | ✅ 100% | unknown | borrowable |

**No existing project occupies all of "Chinese + no model + single file".**

Stated honestly: this gap's commercial value is unknown. How many users would
trade accuracy to avoid a model has not been validated. It is a technical gap,
not a demonstrated demand.

---

## Positioning

1. **Do not rebuild causal-memory's causal model.** 368 tests and a dedicated
   140-question benchmark already exist; using it beats duplicating it.
2. **Narrow to "Chinese + no model + single SQLite file".** Not because it is
   best, but because it is the only unoccupied position.
3. **Leave abstention to the caller.** The evidence layer exposes
   `corpus_missing`, which a caller can act on directly.
4. **Evaluation-first was the right call.** A purpose-built Chinese set, with its
   limits stated (44 sessions, saturated hit@10), is more honest than quoting a
   number that was never run.

**If the goal is simply "get good agent memory as fast as possible", do not build
this.** Tencent's project, causal-memory, and mem0 are all usable today. Building
it only makes sense when "no model" is a hard constraint.
