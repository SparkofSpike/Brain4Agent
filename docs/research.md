# 论文调研

调研日期：2026-11-06
方法：每篇都实际抓取 arXiv abs 页面，读取标题、作者、投稿日期、摘要原文。
逐篇核实记录见 `/var/lib/b4a\research\agent-memory-papers-2026-11.md`。

---

## 综述

**arXiv:2504.15965** — *From Human Memory to AI Memory: A Survey on Memory
Mechanisms in the Era of LLMs*（Yaxiong Wu 等 8 人，2025-04-22，cs.IR，26 页）

提出 object / form / time 三维度 + 八象限的记忆分类法，并系统对照人类记忆类别。
摘要指出：现有综述缺少「LLM 驱动的 AI 系统记忆与人类记忆之间关系」的系统分析。

**用途**：八象限可作模块划分依据。

**arXiv:2404.13501** — *A Survey on the Memory Mechanism of Large Language Model
based Agents*（Zeyu Zhang 等 9 人，2024-04-21，cs.AI，39 页）

覆盖「是什么 / 为什么 / 怎么设计与评测 / 应用 / 局限与方向」。摘要明确点出现有工作
分散、缺整体视角、没能抽象出可复用设计模式。

**用途**：「设计与评测」两节最能省重复劳动。

---

## 评测基准 ← 本项目 Stage 0 的基础

**arXiv:2410.10813** — *LongMemEval: Benchmarking Chat Assistants on Long-Term
Interactive Memory*（Di Wu 等 6 人，2024-10-14 / v2 2025-03-04，**ICLR 2025**）

五项核心能力（摘要原文）：
information extraction / multi-session reasoning / temporal reasoning /
knowledge updates / **abstention**

500 道题，嵌入可自由扩展的对话历史。实测商业助手和长上下文 LLM 在持续交互中
**掉 30% 准确率**。

三阶段框架：`indexing → retrieval → reading`。
三个优化：session decomposition for value granularity；
fact-augmented key expansion for indexing；time-aware query expansion。

**本项目实测的两条（见 decisions.md D-007 / D-008）**
- 检索不是瓶颈（BM25 零依赖 hit@10 = 96.6%）
- 分数不能当拒答信号（最佳平衡 0.688，代价是丢 46% 正常查询）

**arXiv:2402.17753** — *Evaluating Very Long-Term Conversational Memory of LLM
Agents*（LoCoMo，Adyasha Maharana 等 6 人，2024-02-27，cs.CL）

35 会话 / 平均 300 轮 / 9K tokens，含图像分享。评测 QA、事件摘要、多模态生成。
结论：LLM 长程时序与因果理解仍困难；长上下文与 RAG 有改善但明显落后人类。

**用途**：偏对话。做 coding agent 记忆时 LongMemEval 更贴。

---

## 架构参考

**arXiv:2405.14831** — *HippoRAG: Neurobiologically Inspired Long-Term Memory for
LLMs*（Bernal Jiménez Gutiérrez 等 5 人，2024-05-23 / v3 2025-01-14，**NeurIPS 2024**）

依据海马体索引理论，LLM + 知识图谱 + **Personalized PageRank**，模拟新皮层与
海马体分工。多跳 QA 比 SOTA 提升 **20%**；单步检索达到或超过 IRCoT，
但**便宜 10–30 倍、快 6–13 倍**。

**注意**：neural-memory 有 `engine/ppr_activation.py` 实现过 PPR，但那个库
34048 条突触里 79% 神经元是 `4G`/`5h` 这类碎片。算法对，图脏。
**这是「控制图质量」这条设计理由的来源。**

**arXiv:2502.12110** — *A-MEM: Agentic Memory for LLM Agents*（Wujiang Xu 等 6 人，
2025-02-17 / **v11 2025-10-08**）

基于 Zettelkasten：新记忆写入时生成结构化笔记（上下文描述、关键词、标签），
分析历史记忆建立链接。核心机制 **memory evolution**——新记忆会触发对**旧记忆**
的上下文表示和属性更新。六个基础模型上优于 SOTA。

**注意**：改写旧记忆意味着可追溯性下降，需要保留版本。本项目暂不采用。

**arXiv:2305.10250** — *MemoryBank: Enhancing Large Language Models with Long-Term
Memory*（Wanjun Zhong 等 5 人，2023-05-17，10 页）

受艾宾浩斯遗忘曲线启发，依据**经过时间**与**相对重要性**决定遗忘与强化。

**用途**：只增不减的记忆库检索质量会单调下降。遗忘是可计算的功能，不是 bug。

**arXiv:2304.03442** — *Generative Agents: Interactive Simulacra of Human Behavior*
（Joon Sung Park 等 6 人，2023-04-07 / v2 2023-08-03，cs.HC）

memory stream（完整经历）+ reflection（随时间综合成更高层反思）+ 动态检索规划。
消融证明三者各自都有实质贡献。

**用途**：`reflection` 是缺失的一环——存「做了什么」和存「学到什么」是两件事。

---

## 未逐篇核实的候选

下列出现在搜索结果中，但**没有打开 arXiv 页面验证**，引用前必须补验：

- 2404.16130 — GraphRAG
- 2410.05779 — LightRAG
- 2504.13171 — Sleep-time Compute
- 2409.01317 — Language Models Need Sleep
- 2408.08921 / 2501.00309 / 2402.07630 — graph RAG 相关

---

## 跨论文的结论

1. **写入与查询必须分离**（LongMemEval 三阶段）。写入要覆盖全，查询要精准，
   共用一套键生成必然顾此失彼。
2. **拒答是一等能力，不是错误处理**（LongMemEval 第五项）。且本项目实测证明
   它不能用分数阈值实现。
3. **图的质量比图算法重要**（HippoRAG + 我们对 neural-memory 的实测）。
4. **遗忘是功能**（MemoryBank）。
5. **反思层是缺失的一环**（Generative Agents）。
6. **中文是空白**：8 篇论文零篇处理 CJK，评测集全英文。
