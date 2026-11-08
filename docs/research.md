# Literature Review

Surveyed 2026-11-06. Every entry was checked by fetching the arXiv abstract page
and reading the title, authors, submission date, and abstract text — nothing
here is cited from memory.

---

## Surveys

**arXiv:2504.15965** — *From Human Memory to AI Memory: A Survey on Memory
Mechanisms in the Era of LLMs* (Yaxiong Wu et al., 8 authors, 2025-04-22, cs.IR,
26 pages)

Proposes a three-axis taxonomy — object, form, time — yielding eight quadrants,
and maps them against categories of human memory. Notes that prior surveys lack a
systematic treatment of the relationship between LLM-driven memory and human
memory.

*Relevance:* the eight quadrants are a usable basis for module boundaries.

**arXiv:2404.13501** — *A Survey on the Memory Mechanism of Large Language Model
based Agents* (Zeyu Zhang et al., 9 authors, 2024-04-21, cs.AI, 39 pages)

Covers what memory is, why it is needed, how to design and evaluate it,
applications, and open problems. States plainly that existing work is scattered,
lacks a holistic view, and has not abstracted reusable design patterns.

*Relevance:* the design and evaluation sections remove the most duplicated work.

---

## Benchmarks — the basis for Stage 0

**arXiv:2410.10813** — *LongMemEval: Benchmarking Chat Assistants on Long-Term
Interactive Memory* (Di Wu et al., 6 authors, 2024-10-14, v2 2025-03-04,
**ICLR 2025**)

Five core abilities, from the abstract: information extraction, multi-session
reasoning, temporal reasoning, knowledge updates, and **abstention**.

500 questions embedded in extensible conversation histories. Commercial
assistants and long-context LLMs lose **30% accuracy** under sustained
interaction.

Three-stage framework: `indexing → retrieval → reading`. Three optimisations:
session decomposition for value granularity, fact-augmented key expansion for
indexing, time-aware query expansion.

*Measured in this project (D-007, D-008):*

- Retrieval is not the bottleneck — BM25 with zero dependencies reaches 96.6%
  hit@10.
- Scores cannot implement abstention — best balanced threshold 0.688, at the cost
  of discarding 46% of normal queries.

**arXiv:2402.17753** — *Evaluating Very Long-Term Conversational Memory of LLM
Agents* (LoCoMo, Adyasha Maharana et al., 6 authors, 2024-02-27, cs.CL)

35 sessions, averaging 300 turns and 9K tokens, including shared images.
Evaluates question answering, event summarisation, and multimodal generation.
Finds long-range temporal and causal understanding still difficult; long context
and RAG help but remain far behind human performance.

*Relevance:* conversation-oriented. LongMemEval fits a coding agent's memory
better.

---

## Architecture references

**arXiv:2405.14831** — *HippoRAG: Neurobiologically Inspired Long-Term Memory for
LLMs* (Bernal Jiménez Gutiérrez et al., 5 authors, 2024-05-23, v3 2025-01-14,
**NeurIPS 2024**)

Based on hippocampal indexing theory: LLM plus a knowledge graph plus
**Personalized PageRank**, modelling the division of labour between neocortex and
hippocampus. Improves multi-hop QA by **20%** over prior state of the art;
single-step retrieval matches or beats IRCoT while being **10–30× cheaper and
6–13× faster**.

*Note:* a previously examined library implemented PPR in
`engine/ppr_activation.py`, but 79% of the nodes in its 34,048-edge graph were
fragments like `4G` and `5h` extracted from prose. The algorithm was right; the
graph was not. This is the origin of the "graph quality beats graph algorithm"
rationale.

**arXiv:2502.12110** — *A-MEM: Agentic Memory for LLM Agents* (Wujiang Xu et al.,
6 authors, 2025-02-17, v11 2025-10-08)

Built on Zettelkasten. Each new memory becomes a structured note — contextual
description, keywords, tags — and is linked against existing memories. The core
mechanism, **memory evolution**, lets a new memory update the contextual
representation and attributes of *older* memories. Outperforms prior state of the
art across six base models.

*Note:* rewriting old memories reduces traceability and requires versioning. Not
adopted here.

**arXiv:2305.10250** — *MemoryBank: Enhancing Large Language Models with Long-Term
Memory* (Wanjun Zhong et al., 5 authors, 2023-05-17, 10 pages)

Inspired by the Ebbinghaus forgetting curve: forgetting and reinforcement are
driven by **elapsed time** and **relative importance**.

*Relevance:* a memory store that only grows degrades monotonically in retrieval
quality. Forgetting is a computable feature, not a bug.

**arXiv:2304.03442** — *Generative Agents: Interactive Simulacra of Human
Behavior* (Joon Sung Park et al., 6 authors, 2023-04-07, v2 2023-08-03, cs.HC)

Memory stream (complete experience) plus reflection (periodic synthesis into
higher-level observations) plus retrieval-driven planning. Ablations show each
contributes materially.

*Relevance:* reflection is the missing layer — storing *what happened* and storing
*what was learned* are two different things.

---

## Candidates not individually verified

These appeared in search results but their arXiv pages were **not opened**.
Verify before citing:

- 2404.16130 — GraphRAG
- 2410.05779 — LightRAG
- 2504.13171 — Sleep-time Compute
- 2409.01317 — Language Models Need Sleep
- 2408.08921 / 2501.00309 / 2402.07630 — graph RAG variants

---

## Conclusions across the literature

1. **Indexing and querying must be separated** (LongMemEval's three stages).
   Writing wants broad coverage, querying wants precision, and one shared
   tokeniser necessarily loses on one side. Measured in D-019.
2. **Abstention is a first-class capability, not error handling** (LongMemEval's
   fifth ability). This project measured that score thresholds cannot implement
   it (D-008) and that corpus-absence evidence gets partway (D-018).
3. **Graph quality beats graph algorithms** (HippoRAG, plus the PPR observation
   above).
4. **Forgetting is a feature** (MemoryBank).
5. **A reflection layer is missing from most designs** (Generative Agents).
6. **Chinese is unaddressed.** None of the eight papers handle CJK; every
   benchmark is English.
