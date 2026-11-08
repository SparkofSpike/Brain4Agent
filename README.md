# Brain4Agent

面向 agent 的长期记忆系统。中文优先，读写分离，拒答可区分。

## 为什么做

不是因为某个现有实现 bug 多。逐条查过的那些问题（中文分词、空壳记录、
参数未落库）都能在原代码里修掉，不需要换架构。

要做，是因为四件事在别人的结构里改不动：

**1. 写入与查询是两种不同的操作，不能共用一套键生成**
LongMemEval（ICLR 2025）把长期记忆拆成 `indexing → retrieval → reading` 三阶段。
写入要覆盖全（宁可多留线索），查询要精准（宁可少给噪音）。共用一个函数，
必然顾此失彼——这是结构冲突，不是实现疏忽。

**2. 拒答是能力，不是错误处理**
同一个基准把 `abstention` 列为五项核心能力之一。
系统必须能区分三种情况，并把区分暴露给调用方：

```
Found     有这条记忆，这是它
NotFound  确实没记过
Uncertain 可能有，但我不确定（检索不可靠 / 记忆过期 / 冲突未解决）
```

把「没找到」和「没有」混为一谈，是当前最贵的一类缺陷——调用方会重做
已经做过的事，而且永远不会知道。

**3. 图的质量比图的算法重要**
HippoRAG（NeurIPS 2024）用 Personalized PageRank 在知识图上做检索。
现有实现也抄了 PPR，但实测那个库里 79% 的神经元是 `4G` / `5h` / `2G`
这类从正文抓出来的碎片。同样的算法，喂不同的图，结果天差地别。

**4. 中文不是翻译问题**
调研的八篇论文，没有一篇处理 CJK 分词——它们的评测集都是英文。
中文没有空格，词边界不落在字符边界上，任何「按字符/按空白切」的默认假设
都会在真实查询上失效。这块没有现成答案。

## 非目标

明确不做，避免范围蔓延：

- **不追求功能数量。** 不做 60 个工具。工具少而可靠。
- **不做通用知识库。** 只服务 agent 的工作记忆——项目上下文、决策、踩过的坑。
- **不替代现有系统。** 目标是能与之并存（同一份数据，两套检索），不是吞掉它。
- **不承诺记忆是无限的。** 会遗忘是设计的一部分（MemoryBank 的艾宾浩斯思路），
  不是要修的 bug。

## 当前阶段：先有尺子

**Stage 0 — 评测先行。** 不写记忆系统，先能量出「好」和「坏」。

理由很直接：这几天刚经历过一次「以为修好了，实际是静默失败」。
没有基线数字之前，任何「做完了」都是自说自话。

Stage 1 起才写核心，且只在 Stage 0 指出问题的地方写。

## 参考

| arXiv | 用什么 |
|---|---|
| 2410.10813 | LongMemEval：五项能力、三阶段框架、abstention |
| 2402.17753 | LoCoMo：超长多轮对话评测 |
| 2404.13501 | 记忆机制综述：设计模式 |
| 2504.15965 | 人类记忆 → AI 记忆：八象限分类 |
| 2405.14831 | HippoRAG：PPR 图检索 |
| 2502.12110 | A-MEM：Zettelkasten + memory evolution |
| 2305.10250 | MemoryBank：遗忘曲线 |
| 2304.03442 | Generative Agents：memory stream + reflection |

| 2504.01101 | QPP 的局限：预测器不泛化、下游收益边际 |

逐篇核实记录在 `docs/research.md`。领域竞品与空白分析在 `docs/landscape.md`。

## 怎么用

命令行最快：

```bash
# 装：pip install -e .        （只需 Python 3.11+，无必装依赖）
$ brain4agent exchange \
    "中文查询搜不到记忆，nmem recall 返回空" \
    "根因在 keywords.py 第 443 行，正则不含汉字范围"

$ brain4agent search "中文查询为什么搜不到"
1 sessions for '中文查询为什么搜不到'
  [ 6.08 cov=1.00] e9d4c39a3790

--- e9d4c39a3790  score=6.08  coverage=1.00 ---
user: 中文查询搜不到记忆，nmem recall 返回空
assistant: 根因在 keywords.py 第 443 行，正则不含汉字范围

$ brain4agent search "杭州测试机上装的是什么数据库"
  [ 3.39 cov=0.34] 2287920753f2  missing: 数据库, 机上
```

`missing:` 那一行是重点。它不是「没找到」，是「语料里没有『数据库』这个词」。

### 给 agent 用（MCP）

```json
{"mcpServers": {"brain4agent": {
    "command": "python",
    "args": ["-m", "brain4agent.mcp_server"],
    "env": {"BRAIN4AGENT_DB": "/path/to/brain.db"}
}}}
```

四个工具：

- `memory_remember` — 存一轮对话，返回 session id
- `memory_recall` — 检索。返回证据，包含 `missing_terms`
- `memory_session` — 读整个会话
- `memory_stats` — 计数

`memory_recall` 不告诉你「有没有答案」，它告诉你「哪些词没找到」。
判断留给调用方——它手里已经有模型了。

实现是标准库写的 stdio JSON-RPC，没装 `mcp` 包：核心加一个依赖，
就要有一个实测数字证明它必要（D-004），而序列化四个 dict 不值得当最大的依赖。

### 作为库

```python
from brain4agent import MemoryStore

with MemoryStore("brain.db") as store:
    sid = store.new_session()
    store.add("中文查询搜不到记忆", session_id=sid, role="user")
    store.add("根因在 keywords.py 第 443 行", session_id=sid, role="assistant")

    result = store.search("中文查询为什么搜不到")
    best = result.best()
    print(best.coverage, best.missing_terms)
```

检索返回**证据**，不返回结论：

```
>>> store.search("杭州测试机上装的是什么数据库")
evidence=[Evidence(score=2.70, coverage=0.32, missing_terms=['数据库', '机上'])]
```

`missing_terms` 直接说明语料里没有「数据库」这个词。这算不算「没记过」，
由调用方结合上下文判断——它手里已经有模型了。

一次问答用 `add_exchange` 存成一个会话。检索以会话为单位，不是以轮为单位：
答案常常跨轮，按单轮判断会产生大量假拒答。

### 可选的云端 judge（默认关）

```python
from brain4agent import Judge, MemoryStore

judge = Judge.from_env()      # 读 BRAIN4AGENT_JUDGE_KEY；没配返回 None
if judge:
    r = store.search_judged("问题", judge)
    print(r.verdict)          # found / not_found / uncertain
```

`uncertain` 是独立的第三种状态：judge 超时、挂掉、返回无法解析时给这个，
不给 `not_found`。混为一谈会在网络故障时告诉用户「我从来没记过这件事」。

启用时必须同时有重试和超时——实测 53 次调用里出现过一次 363 秒，
而 5% 的调用会返回 500/503。

## 结构

```
src/brain4agent/
  tokenize.py   分词与 BM25。bigram 负责匹配，整词负责解释缺了什么
  store.py      SQLite 存储 + 内存索引，会话为检索单位
  evidence.py   证据结构。算分可以，下结论不行
  judge.py      可选云端判断。默认关，没 key 就没有网络调用
bench/          评测。复用 src 的代码，保证测的和用的是一套
docs/           设计决策与调研
tests/          测试
```

## 状态

**Stage 0 完成，Stage 1 核心已实现。**

### 1. 检索不是瓶颈

BM25（1994 年的算法，零依赖）在 LongMemEval cleaned 全量 470 道可答题上：

```
hit@1    0.866
hit@5    0.951
hit@10   0.966
MRR      0.899
```

每题平均 493 轮对话的真实 haystack，96.6% 的答案能在前十名里找到。
**「检索难」这个自建理由不成立。**

### 2. 分数不能当拒答信号

30 道「无答案」题（每道都带共享词汇的诱饵），BM25 top-1 分数分布：

```
abstention  min 9.6   median 19.7  max 43.4
answerable  min 9.4   median 25.2  max 135.5
```

几乎完全重叠。最佳平衡阈值只有 **0.688**（随机基线 0.5），而达到它要丢掉
**46%** 的正常查询。试过 IDF 覆盖率（0.697）和未见词占比（0.730），都不行。

### 3. 中文评测集建了，bigram 够用

59 道题（40 可答 / 19 abstention）、44 个会话的共享语料，全部中文原生。

| 指标 | bigram | jieba |
|---|---|---|
| hit@1 | 0.825 | **0.875** |
| hit@10 | **1.000** | 0.975 |
| MRR | 0.891 | **0.910** |

jieba 精确度略高但召回更低（未登录词），对拒答无帮助。保留 bigram：零依赖、
专有名词不会切错。**hit@10 已饱和在 1.0，hit@1 才是有区分度的指标。**

### 4. 拒答交给调用方，记忆系统不跑模型

试过本地 LLM 判断「这段证据里有没有回答问题所需的信息」，在 59 题上：

```
总是回答          keep 1.000   refuse 0.000
带模型判断        keep 0.950   refuse 0.895
```

目标达成了，但 `qwen3.5:9b` 常驻 **15.2 GB**，在这台 23.6 GB 的机器上直接把系统卡死。
4b 要 9 GB 且只能到 keep 0.92。

**更重要的是：调用方本来就是 agent，它自己就有模型。**
让记忆系统再起一个模型问「这条记忆相关吗」，是重复开销，不是功能。

所以拒答由调用方做，记忆系统只交证据：

```
记忆系统                          调用方（agent）
检索 → 排序                       拿到 evidence + missing_terms
每条带 score / coverage / 缺失词    用已有的模型判断，或直接展示给用户
零模型、零推理、几十 MB             这是它本来就该做的事
```

关键是 `missing_terms`——不是分数，是**具体的词**：

```
可答：杭州测试机的地址和部署目录是什么？  cov=1.00  无缺失
不可答：杭州测试机上装的是什么数据库？    cov=0.44  缺失 ['数据库']
```

**代价说清楚**：开箱即用的拒答没了。非模型方法的拒答率上限实测是 **21%**
（BM25 分数、IDF 覆盖、未见词、词级 coverage、NQC/WIG/Clarity、
13 特征逻辑回归全试过），而模型能到 89.5%–100%。差一个数量级。

### 5. 云端 API 能补上，但慢

如果非要内置拒答，云端是唯一不占内存的路：

| 方案 | keep | refuse | 单次延迟 | 常驻内存 |
|---|---|---|---|---|
| 纯 BM25 | 1.000 | 0.000 | 0.22 ms | 0 |
| 本地 2b | 0.750 | 1.000 | 0.77 s | ~1 GB |
| 本地 9b | 0.950 | 0.895 | 1.5–2 s | **15.2 GB** |
| 云端 deepseek-flash | 0.925 | 1.000 | **0.81 s** | **0** |

云端用零内存换到接近 9b 的准确率。但 0.81 秒 × 一个任务的 30 次记忆查询 = 24 秒。

**高频路径上，这个判断本可以不做**——调用方的 agent 正在推理，
顺手就能判断「这条记忆相不相关」。为它单独发起一次 API 调用，是串行化的第二次推理。

成本不是问题：214 token/次，约 0.0001 元。延迟才是。

完整决策记录在 `docs/decisions.md`（D-001 到 D-014），论文笔记在 `docs/research.md`。
这个 README 会随决策改变而更新——如果某条设计被实测否定，删掉它，不要留着好看。
