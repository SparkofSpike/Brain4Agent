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

---

## 2026-11-08 · 第二轮调研：范式级的想法

第一轮（见上文）看的八篇都是"怎么把记忆做得更好"。这一轮专门找**换一种做法的**，
不是工程改进，是范式。

结论先写：**三篇值得认真对待，其中一篇直接否定了本项目这两天做的事。**

### 1. STONE — 先存，按需提取（arXiv:2602.16192）

Kioxia 的研究中心，2026-02。作者自己说"不是提出新方法"，而是重新审视一个被广泛
忽略的范式。深读过正文与实验。

**它把主流做法命名为 "extract then store"**：经验到来时，按**当前任务**判断什么
有用，只存提取结果。形式化是 `M = f_T(E)`，f_T 是任务 T 的提取函数。

**它提出的替代是 STONE**（Store Then ON-demand Extract）：`M = E`，原样存经验；
检索到之后再用 `f_T'(M)` 按新任务提取。

**为什么 extract-then-store 必然丢信息**：f_T 只保留对 T 有用的部分。如果经验里有
对将来 T' 有用、对 T 无用的内容 x，x 会在提取时被丢掉。生物学上的类比是
**latent learning**：大鼠在没有找食物动机时闲逛迷宫，仍记住了食物位置；饥饿时能回想
起来。这在 extract-then-store 下不可能。

**一个反直觉的结论**：在"不得丢失任何潜在有用信息"这个约束下，STONE 的存储大小
**反而是最小的**（论文附录 A 有证明）。直觉上"存全文"更占空间，但为了避免丢失而
多存的那些提取版本，总量更大。

**实验**：内部规章问答，检索预算 100 次。extract-then-store 更快耗尽预算——同一份
文档的不同部分被反复取回；STONE 存整份文档，后续问题直接命中，答对更多。

**这对本项目的意义，是直接打脸**：

D-023 测出预先提炼把 hit@1 从 0.9250 打到 0.7750，我当时归因于"fact 之间词汇重叠"。
STONE 给出了更根本的解释：**我在写入时做了 f_T 的判断，而 f_T 不知道将来会被怎么问**。
提炼器把"根因在哪个文件哪一行"改写成"extraction/keywords.py 第 443 行"，
正是在丢掉当时看起来无用、但对检索至关重要的措辞。

我的数据与 STONE 的预测完全一致，只是我当时不知道这叫 extract-then-store。

**要做的不是删掉提炼，是挪位置**：写入存原文（现有 `add()` 路径，0.9250），
检索后按需提炼（尚未实现）。这正是 LongMemEval 三阶段里的 reading 阶段。

**保留意见**：作者是存储公司，"存得更多"对它有利。但论证本身站得住，且与我的独立
测量吻合——这一点比作者身份更重要。

### 2. 检索后装配是独立问题（arXiv:2606.01435）

Reddy & Challaram，2026-05。标题就是论点：**别让 LLM 去跟踪新鲜度**。

读的是摘要（正文页未渲染）。核心：

- 记忆系统常常"检索到了正确证据，仍然答错"，因为语义筛选、冲突解决、旧信息抑制、
  输出生成被塞进同一步。
- 作者把它定义为 **post-retrieval assembly**，主张拆成两段：一段做语义匹配，
  另一段执行答案策略。
- 在 MemoryAgentBench 的 FactConsolidation 上：**规格里明说新事实的序号更大**
  （一个确定性的、不需要模型判断的规则），但 22 个已报告系统里，最好的单跳 54%，
  **所有系统的多跳都没超过 7%**。

**两个可用的东西**：

1. **冲突用序号/时间戳解决，不问模型**。"哪条更新"是数据里带着的，不是需要推理的。
   本项目每条记忆都有 `created_at`，但检索排序完全没用它。
2. **检索与装配分离**。本项目已经是这样（`search` 只给证据，判断交调用方），
   算是无意中做对了。

**7% 这个数字值得记住**：多跳记忆是全行业未解的问题，不是本项目的短板。

### 3. FadeMem — 生物学遗忘曲线（arXiv:2601.18642）

只看了摘要。双层记忆架构 + 遗忘曲线 + 冲突解决 + 自适应融合，靠 LLM 引导融合。

**借鉴点**：遗忘不该是"按时间删最旧的"（本项目现在的 `prune`），而是按
**重要性 × 时间衰减**。D-022 测出 prune 要删掉 95% 才能恢复检索，正说明纯时间策略
没有区分能力。

**不采纳**：LLM 引导的融合与冲突解决——与本项目的零模型核心冲突，
且第 2 篇已经证明确定性规则更好。

### 其他（只看了标题与摘要）

- **MemoryArena**（He et al. 2026）：多会话相互依赖的任务评测。报告称在 LoCoMo 上
  接近饱和的模型在这里掉到 40–60%。如果本项目的 0.925 要对外声称，需要在这类
  基准上重新测——本项目的 44 会话语料同样可能已经饱和。
- **MemoryAgentBench**（arXiv:2507.05257）：增量多轮交互评测。
- **"Can Agents Forget When Asked?"**（arXiv:2610.18052）：记忆遗忘/取消学习评测，
  有 FAMA 指标惩罚对已失效记忆的依赖。
- **Structured Belief State**（arXiv:2605.11325）：精度感知的记忆检索基准。
- **Graph-based Agent Memory 综述**（arXiv:2602.05665）。
- **Memory for Autonomous LLM Agents**（arXiv:2603.07670）：分类学综述，
  提到 Agentic Memory 把记忆操作当作 RL 动作来训练——与零模型路线相反。
- **memorypapers.org**：持续更新的论文索引，以后找新论文从这里入手。

### 综合判断

**真正的范式级想法只有一条**：STONE。它解释了我测到的现象，并给出明确的修正方向
（把提炼从写入挪到读取）。另外两条（确定性冲突解决、遗忘曲线）是它的配套。

**这一轮最重要的收获不是新功能，是一个纠正**：本项目这两天在写路径上做的提炼，
方向是错的；正确的位置是读路径。

**下一步的具体动作**：

1. 实现 reading 阶段：`search` 之后，把 top-k 原文交给模型提取答案。
   这既是 STONE 的 on-demand extract，也补上了 LongMemEval 三阶段的最后一环。
2. 冲突用 `created_at` 解决：同一实体的不同取值，新的覆盖旧的。
   这是确定性的，不需要模型。
3. 遗忘改成"重要性 × 时间衰减"，而不是按时间删最旧。
4. 如果要对外声称检索指标，先在 MemoryArena 这类未饱和的基准上验证。
