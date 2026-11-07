# Agent 记忆系统：领域全貌

调研日期：2026-11-07
方法：GitHub API 核实星数与活跃度，README 实际读取，arXiv 逐篇抓取。
星数为调研时刻快照，会变。

---

## 结论先行

1. **这个领域已经很卷。** 腾讯的项目 30k stars，`causal-memory` 104 stars 且每周在更新，
   mem0 / Zep / Letta / Cognee 各有生态位。
2. **但「零模型 + 中文 + 本地」是真实空白。** 头部项目全部依赖模型：
   腾讯要配两组 LLM 参数，causal-memory 用本地 ONNX embedding，mem0 用向量库。
3. **零模型检索是够用的。** 我们实测 BM25 + 字符 bigram 在 LongMemEval 上 hit@10 = 0.966。
4. **零模型拒答不够用。** 上限 0.211（组合 13 个特征 + 逻辑回归 + 留一交叉验证），
   而本地 LLM 能到 0.895。差四倍。
5. **所以定位应该是：零模型做检索，把判断交给调用方。** 调用方本来就是 agent，有模型。

---

## 竞品对比

### 腾讯 TencentDB-Agent-Memory

`TencentCloud/TencentDB-Agent-Memory` · **30,584 stars** · TypeScript · MIT

团队的记忆中枢：把对话、文档、代码整理成四类可复用资产
（Chat Memory / Skill / LLM-Wiki / Code-Graph），跨 agent 共享。

**模型依赖：强制。** 安装第一步就是
`$EDITOR .env  # Fill in two sets of LLM parameters (memory group + proxy group)`。

架构是三个服务：`memory-core` + `memory-hub` + `proxy`。
接入方式是改 agent 的 base URL，不用插件/MCP。

**它证明了需求很大，也证明了主流方案都假设你有一个 LLM。**

### causal-memory

`JingxuanC/causal-memory` · 104 stars · Rust · Apache-2.0
2026-07-26 创建，2026-11-04 仍在推送，368 个测试。

**这个项目的方向最值得看**，因为它和 Brain4Agent 的原始目标高度重合：

- SQLite 单库，本地优先
- MCP server，14 个工具
- 因果边：`caused` +1.0 / `fact` +0.8 / `meta` +0.6 / `enabled` +0.5 /
  `prevented` **−0.3**（抑制性，GABA 类比）/ `no_effect` 0.0
- 海马体引擎：CSR 图 + 扩散激活 + SWR 巩固（LTP/LTD/GC）+ Q-value
- 混合检索：BM25 + 语义余弦 → RRF 融合
- **压缩存活 100%**（文本对照 45%）

**模型依赖：本地 ONNX embedding（离线）。** 不是 LLM，但仍是模型。

它的 benchmark 分野很清楚：

| 基准 | causal-memory | mem0 |
|---|---|---|
| LongMemEval-S | 76.4% @11.5K tok/q | 94.4%（官方）/ 73.8%（独立复现） |
| LoCoMo | 79.1% | 91.6% |
| CausalEval（自造） | **78%** | 65% |

它自己说得很直白：**「事实召回不是我们的强项」**，强项在因果。

**对 Brain4Agent 的意义**：不要重复做因果。但它的写入分流
（raw → session_logs 审计，distill → facts + causal edges 可检索）值得抄。

### 其他

| 项目 | 形态 | 模型依赖 |
|---|---|---|
| mem0 | 向量库 + 知识图谱双存储 | embedding（必须） |
| Zep | 时序知识图谱 | embedding |
| Letta (MemGPT) | 三层记忆，OS 类比 | LLM（核心机制是自我编辑） |
| Cognee | 本地优先图推理 | embedding |
| LangMem | LangChain 生态 | 可选 |

---

## 拒答：为什么零模型做不到

这是本次调研最花时间的部分，也是唯一有硬数据否定的假设。

### 试过的信号，全部实测

| 信号 | keep≥0.95 时的拒答率 | 来源 |
|---|---|---|
| BM25 top-1 分数 | ~0 | 自建 |
| IDF 加权覆盖 | ~0 | 自建 |
| 未见词占比（bigram） | ~0.05 | 自建 |
| 未见词占比（jieba） | 0.105 | 自建 |
| 词级 coverage | ~0.10 | 自建 |
| NQC（归一化查询承诺） | 0.158 | Shtok et al. 2012 |
| UQC | 0.158 | 同上 |
| WIG（加权信息增益） | 0.000 | Zhou & Croft 2007 |
| Clarity | 0.105 | Cronen-Townsend et al. 2002 |
| top1 − top10 分差 | 0.105 | 自建 |
| **13 特征 + 逻辑回归 + LOOCV** | **0.211** | 自建 |
| 本地 LLM 判断 | **0.895** | 自建 |

### QPP 这条路，文献也否定了

Query Performance Prediction 是 IR 的成熟子领域，专门研究「不看成答案，
怎么预判这次检索的质量」。经典方法都是纯统计，理论上正合需要。

但实测落空，而且文献预期就是这样：

**arXiv:2504.01101** — *Uncovering the Limitations of Query Performance Prediction*
（Chifu, Déjean, Mothe, Garouani, Ortiz, Ullah，2025-04）

跨 ROBUST / GOV2 / WT10G / MS MARCO 四个集合、BM25 / DFree / SPLADE / ColBERT
四种检索器，结论原文：

- 预测器与真实性能的相关性不高
- 预测器对**集合**最敏感，其次才是检索器
- 「QPP-driven selective query processing offers only marginal gains」

**根因**：QPP 预测的是**排序质量**，不是**答案是否存在**。
一个查询可以完美检到最相关的会话，而那个会话里没有答案——
abstention 题正是这样设计的（问仓鼠但只提过猫）。

### 为什么分类器也不行

13 个特征组合后，留一交叉验证的拒绝率到 0.211，比任何单特征都好，但仍只有 LLM 的四分之一。

特征权重里有意义的一条：`t_howmany`（问「多少/几次」）+0.859 最高。
问具体数值的查询更容易无答案——因为语料里只有一个话题，没有那个数。

但这条规律不足以撑起 0.9 的准确率。**「数据库」和「部署目录」是不是同一个概念，
需要语义理解，不是统计能做到的。**

---

## 空白在哪

| 维度 | 腾讯 | causal-memory | mem0/Zep | Brain4Agent 可占 |
|---|---|---|---|---|
| 中文原生分词 | 有（产品面向中文） | **无** | **无** | ✅ |
| 零模型 | **否**（需 LLM） | **否**（需 embedding） | **否** | ✅ |
| 单文件本地 | 部分（三服务） | ✅ SQLite | 否（向量库） | ✅ |
| 拒答 | 无 | 无 | 无 | ⚠️ 只能给证据，判断交调用方 |
| 因果边 | 无 | ✅ 强项 | 无 | 不建议重复 |
| 压缩存活 | 未知 | ✅ 100% | 未知 | 可借鉴 |

**唯一同时满足「中文 + 零模型 + 单文件」的组合，目前没有现成项目。**

但要诚实：这个空白的商业价值不明。愿意为「不用模型」付出准确率代价的用户有多少，
不知道。技术上是空白，需求上没验证。

---

## 对 Brain4Agent 的建议

1. **不要重复 causal-memory 的因果模型。** 它做了 368 个测试、140 题的 CausalEval，
   再做一个不如直接用它的。
2. **定位收窄到「中文 + 零模型 + 单文件 SQLite」。**
   这不是因为它最好，是因为它是唯一没被占的位置。
3. **拒答交给调用方。** 我们已经有证据层（`bench/zh/evidence.py`），
   `missing_terms` 是调用方能直接用的字段。
4. **评测先行已经做对了。** 自造中文集 + 承认它只有 44 个会话、hit@10 饱和，
   比引用一个自己没跑过的数字诚实。

**如果目标是「最快用上好用的 agent 记忆」，答案是别自研**——
腾讯的 30k 星项目、causal-memory、mem0 都能直接用。
自研只在「零模型是硬约束」时才成立。
