# Brain4Agent

**简体中文** | [English](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

给 agent 用的长期记忆。CJK 优先的检索，核心链路不调用模型，
用证据代替置信度分数。

`pip install "Brain4Agent[precise]"` 需要 Python 3.11+ 和一个可选词典。
不装词典，一切照常工作。

---

## 目录

- [是什么](#是什么)
- [安装](#安装)
- [快速开始](#快速开始)
- [工作原理](#工作原理)
- [基准测试](#基准测试)
- [其中有多少是样本噪声](#其中有多少是样本噪声)
- [与前沿对比](#与前沿对比)
- [状态](#状态)
- [设计决策](#设计决策)
- [参与贡献](#参与贡献)
- [特别致谢](#特别致谢)
- [许可证](#许可证)

## 是什么

在代码库上工作的 agent 会积累上下文：某个决定为什么这么做，
哪条路走不通，部署放在哪里。这些上下文的寿命比产生它的对话更长，
重新推导一遍代价很高。

Brain4Agent 把这些轮次存进一个 SQLite 文件，按会话检索。
设计围绕四条约束：

- **CJK 开箱可用。** 用空格或非词边界切分汉字、假名或谚文，
  整个查询会缩成一个 token；半角片假名既不属于 CJK 区段，
  也不属于 ASCII 词类。字符二元组在这几种情况下都不会失手。
  实测数字来自中文（见基准测试）；日文、韩文和越南文有支持、
  有测试，但没有基准数据，README 不会假装有。
- **核心不调用任何模型。** 检索用 BM25。不需要 API key，
  不需要加速器，没有冷启动，单次查询个位数毫秒。
- **它报告证据，不给结论。** 搜索不返回置信度分数，
  而是返回问题里哪些词在语料中从未出现过。这些意味着什么，
  交给调用方判断——调用方本来就有模型。
- **存储的内容不等于索引的内容。** 记忆保留全文，
  供人或 reader 阅读；索引可以只放更窄、更有辨识度的东西。
  把两者分开，PrecisionMemBench 从 0.6104 升到 0.7922，
  其他什么都没改。原因是解释性文字里的常用词和整个领域共享，
  排序会压过那个唯一有辨识度的名字。

每条记忆还带 `tags`（作用域）、`superseded_by`（替代关系）
和 `refs`（一跳链接）。这些都不靠推断，也不需要模型——
它们是写入方知道、但文本本身没说的事实。

## 安装

```bash
pip install "Brain4Agent[precise]"   # recommended: adds jieba, hit@1 0.925
pip install Brain4Agent              # zero dependencies: bigrams only, hit@1 0.825
```

Windows 和 macOS 都能用；需要 Python 3.11 或更高版本。
包已发布到 PyPI：<https://pypi.org/project/Brain4Agent/>。
要改代码本身，先 `git clone`，再 `pip install -e ".[precise,dev]"`。

`jieba` 是一个约 20 MB 的静态词典。它不是必需项——没有它，
代码在检索的读写两侧都退化为二元组，仍然可用。

**Windows 与非 ASCII 输入。** MCP server 和 CLI 把 stdin/stdout
固定为 UTF-8，所以无论控制台代码页是什么，CJK 内容都能正常工作。
但命令行*参数*在那之前就由 Python 解码了——如果要把非 ASCII
文本作为参数传入，请设置 `PYTHONUTF8=1`。

## 快速开始

### CLI

```bash
$ brain4agent exchange \
    "中文查询搜不到记忆，recall 返回空" \
    "根因在 keywords.py 第 443 行，正则不含汉字范围"
stored exchange in session 913bcac81341

$ brain4agent exchange \
    "杭州那台测试机的地址是 203.0.113.42" \
    "部署目录 /opt/demo-app，跑的是 PostgreSQL 14"
stored exchange in session c060a5294037

$ brain4agent search "中文查询为什么搜不到" -k 1
1 sessions for '中文查询为什么搜不到'
  [ 6.73 cov=1.00] 913bcac81341

--- 913bcac81341  score=6.73  coverage=1.00 ---
user: 中文查询搜不到记忆，recall 返回空
assistant: 根因在 keywords.py 第 443 行，正则不含汉字范围
```

一个存储答不了的问题：

```bash
$ brain4agent search "杭州测试机上装的是什么数据库" -k 1
1 sessions for '杭州测试机上装的是什么数据库'
  corpus has no: 数据库, 机上
  [ 4.81 cov=0.28] c060a5294037  session lacks: 数据库, 机上
```

从 JSONL 批量导入已有历史（每行一个对象，`content` 必填）：

```bash
$ brain4agent import conversations.jsonl
imported 4 turns across 2 sessions
```

```jsonl
{"session_id": "s1", "role": "user", "content": "中文查询搜不到记忆"}
{"session_id": "s1", "role": "assistant", "content": "根因在 keywords.py 第 443 行"}
```

### MCP server

```json
{"mcpServers": {"brain4agent": {
    "command": "python",
    "args": ["-m", "brain4agent.mcp_server"],
    "env": {"BRAIN4AGENT_DB": "/path/to/brain.db"}
}}}
```

四个工具：

| 工具 | 用途 |
|---|---|
| `b4a_remember` | 存一轮对话；返回会话 id |
| `b4a_recall` | 检索；返回会话，附带 `corpus_missing` 和每个会话的 `missing_terms` |
| `b4a_session` | 按顺序读出一个完整会话 |
| `b4a_stats` | 计数 |

`b4a_` 前缀是刻意的：`remember`、`recall` 这类名字会和同一个
客户端里加载的其他 MCP server 撞名。

这个 server 用标准库实现了基于 stdio 的 MCP——四个方法，
走换行分隔的 JSON-RPC。官方 SDK 为了序列化四个字典，
要拖进一棵庞大的依赖树；见
[D-017](docs/decisions.md#d-017-mcp-implemented-against-the-protocol-not-the-sdk)。

同一个库还打包成了 agent [skill](integrations/)——一份文档，
agent 读完后调用 CLI。

### 库

```python
from brain4agent import MemoryStore

with MemoryStore("brain.db") as store:
    sid = store.new_session()
    store.add("杭州那台测试机的地址是 203.0.113.42", session_id=sid, role="user")
    store.add("部署目录 /opt/demo-app", session_id=sid, role="assistant")

    result = store.search("杭州测试机上装的是什么数据库")
    print(result.corpus_missing)        # ['数据库', '机上']
    print(result.best().coverage)       # 0.28
```

`add_exchange(user=..., assistant=...)` 把一问一答存成一个会话。
检索以会话为单位，不是以轮次为单位：一个答案常常横跨整段对话，
按单轮判断会产生多得多的假阴性。

批量导入走 `batch()`，把提交推迟到每组一次：

```python
with store.batch():
    for turn in turns:
        store.add(turn.text, session_id=turn.sid, role=turn.role)
```

### 可选的托管 judge

默认关闭。只有存在 key 时才启用。

```python
from brain4agent import Judge, MemoryStore

judge = Judge.from_env()          # reads BRAIN4AGENT_JUDGE_KEY; None if unset
if judge:
    r = store.search_judged("question", judge)
    print(r.verdict)              # found / not_found / uncertain
```

`uncertain` 是独立的第三种状态。judge 超时或返回 503，
并不能证明记忆不存在；把两者合并，会把一次故障变成
一句笃定的"我从没记过这件事"。

## 工作原理

```
                write                          read
                  │                              │
        bigrams + dictionary words          dictionary words
                  │                              │
                  ▼                              ▼
          ┌──────────────────────────────────────────┐
          │  SQLite: id, session, role, content,     │
          │          created_at, tokens              │
          └──────────────────────────────────────────┘
                  │                              │
                  ▼                              ▼
           BM25 index (in memory)  →  session-level evidence
                                        + corpus_missing
                                        + per-session coverage
```

### 存储

一个 SQLite 文件。行是对话轮次，检索单位是会话。
索引 token 在写入时算好，缓存在一个列里，
所以重建内存索引只是数一遍，不用重新分词。

每一行还带 `tags`、`superseded_by`、`refs`，
以及一段可选的独立索引文本——这些元数据由调用方在写入时提供，
存储层在读取时据此过滤和扩展。PrecisionMemBench 能解，全靠这些字段；
没有它们，同一个 BM25 只有 0.3766。

写入按批提交。1,016 轮导入，逐条提交要 3.8 s，
放进 `batch()` 是 0.25 s——一千次单独提交比这一千次写入本身还贵。

```
                       import    rebuild   query
127 turns   44 sessions          3.9 ms   1.8 ms
1016 turns  352 sessions  0.25 s  29 ms   3.7 ms
```

### 读写分词器分离

建索引和查询是两件不同的事。写入希望留住所有线索——
包括二元组，它给未知词留了一条匹配路径。查询希望丢掉噪声——
不要从 `杭州测试机` 里切出 `州测`。

在索引策略 × 查询策略的 3×3 网格上实测：

| 索引 | 查询 | hit@1 | MRR |
|---|---|---|---|
| bigram | bigram | 0.825 | 0.891 |
| jieba | jieba | 0.875 | 0.911 |
| **bigrams + words** | **words** | **0.925** | **0.940** |

每一组非对称搭配都胜过对称搭配。见
[D-019](docs/decisions.md#d-019-index-wide-query-narrow)。

### 证据，而非结论

所有廉价的弃答信号都测过，没有一个够用：在保留 95% 可答查询的
前提下，按 BM25 分数或 IDF coverage 弃答，最多能拒掉约 10%；
用十三个特征做逻辑回归，能到 21%。

唯一有用的信号是另一类东西。它不是分数，是事实：

- **`corpus_missing`** — 问题里的某个实词在整个已存语料中从未出现。
  是每个会话都没有，不只是这一个。
- **`missing_terms`**（按会话）—— *这个*会话为什么排在这个位置。

区别很重要。`用户` 和 `记忆` 在大多数单个会话里都不出现，
但在语料里存在；这说明"不是这一条"，而不是"我不知道"。
把两者混为一谈，40 个可答问题里产生了 27 次误拒。
见 [D-018](docs/decisions.md#d-018-corpus-absence-is-not-session-absence)。

## 基准测试

所有数字都能用 `bench/` 下的脚本复现，
这些脚本导入的是发布代码本身，不是它的副本。

### LongMemEval（英文）— 检索

`s` 变体上 470 个可答问题（平均 haystack 493 轮），
BM25 基于字符二元组：

```
hit@1    0.8660
hit@5    0.9511
hit@10   0.9660
MRR      0.8990
```

用 `python -m bench.run_longmemeval --dataset s` 复现。
`bench/data/` 由 `bench/fetch_data.py` 拉取，不进仓库——
264 MB 的 JSON 不该放进 git，抓取脚本能逐字节重建它。

**在这个基准上，检索不是难点。** 96.6% 的答案落在前十。

它只对这个基准成立，不能外推：

| 基准 | 指标 | 本存储 | 最好的第三方 |
|---|---|---|---|
| LongMemEval `s` | hit@10 | 0.9660 | — |
| PrecisionMemBench | active | **34/43** | 18/43 (open-knowledge-format) |
| PrecisionMemBench | total | **64/77** | 36/77 (open-knowledge-format) |
| LoCoMo | hit@1 | **0.6270** | 不可比（见下文） |

三个基准用的是同一个检索器。不同的是问题在要什么。
LongMemEval 在大 haystack 上问宽泛的问题，命中前十就算对。
PrecisionMemBench 问的是不是恰好返回正确的那条信念、别的都不要——
多出一篇文档，这条用例就失败。前者说明检索已经解决；
后者上，公开成绩最好的系统正确率不到一半。

"检索很容易"和"检索很难"都成立。
你看到哪一个，完全取决于什么算命中。

### PrecisionMemBench — 检索精度

35 条信念上的 77 个单轮用例。没有生成模型，也没有 judge：
答案*就是*存储返回的信念 ID 集合，
每个用例断言哪些 ID 可以出现、哪些不可以。

```
active  34/43 = 0.7907     cases that need query-dependent retrieval
 total  64/77 = 0.8312     including cases any store passes
```

要两个数字，因为基准就报两个，而且两者不能互换。
**`active` 才是诚实的那个**：77 个用例里有 33 个在空存储上也能通过，
因为它们的断言只涉及 harness 从语料本身推导出的层级
（persona 前奏、固定事实、开放问题）。只引用 `total`，
等于把没做的工作算在本项目头上。

官方表格，来自基准自己的网站（`tenureai.dev/benchmark`）：

```
System                  Active    Total
----------------------  ------    -----
tenure (the authors)    43/43     77/77
this store              34/43     64/77
open-knowledge-format   18/43     36/77
atomicmemory             5/43     16/77
supermemory              2/43     15/77
yourmemory               0/43     21/77
gbrain                   0/43     11/77
hindsight                0/43     11/77
zep                      0/43      9/77
agentmemory              0/43      9/77
cognee                   0/43      9/77
mem0                     0/43      9/77
a-mem                    0/43      7/77
```

本项目在两列上都胜过所有第三方系统。
**但它没有赢过基准自己的参考实现**，后者是 43/43 和 77/77，
也没有理由指望它赢。

两列的分母不同——34/43 active 不是 34/77——所以两列都给出。
结论是**本基准上最好的第三方成绩**，不是"击败了最先进水平"。

### LoCoMo — 检索

272 个会话、5882 轮对话、十个独立对话样本上的 1540 个问题。

```
hit@1   0.6270
hit@5   0.8850
hit@10  0.9471
MRR     0.7369
```

`python -m bench.locomo.run`——不用模型，数据已提交进仓库。

**与公开的 LoCoMo 数字不可比。** 那些（0.67–0.92）测的是在自带
backbone 和 prompt 的 LLM judge 下的答案准确率；
这里测的是正确的会话有没有被检索出来。

提醒一个诱人的推论，因为本 README 做过这个推论，而且错了。
`hit@5 = 0.8850` **不是**任何人答案准确率的上限——
它只约束一个只看得到恰好五个会话、且不许越过它们推理的 reader。
本项目自己的 `hit@10 = 0.9471` 已经超过 MemMachine 的 0.9169，
而那是同一个存储，所以这个界对别的系统的检索什么也说明不了。
公开的 LoCoMo 数字两个方向都不能当作检索质量的证据。

这里的设计教训：检索单位必须和答案的单位一致。
索引轮次而不是会话，hit@1 掉 8.2 个点（0.5447 → 0.6270），
其他什么都没改。

### 中文基准

59 个问题（40 个可答，19 个不可答），44 个共享语料会话，
从零构建——调研的八篇论文没有一篇评估 CJK 分词。

| 指标 | bigram/bigram | bigrams+words / words |
|---|---|---|
| hit@1 | 0.825 | **0.925** |
| hit@5 | 0.950 | **0.975** |
| hit@10 | 1.000 | 1.000 |
| MRR | 0.891 | **0.940** |

hit@10 已经饱和；只有 hit@1 和 MRR 有区分度。

### 这个数字在哪里失效

**0.925 描述的是 44 个会话。它扛不住密集的近重复。**
加入 300 个同主题但陈述不同事实的噪声会话，hit@1 掉到零：

| 语料 | hit@1 | hit@5 |
|---|---|---|
| 44 个会话 | 0.925 | 0.975 |
| +300 个同主题会话 | 0.000 | 0.000 |

机制是字面匹配。问*"根因在哪个文件哪一行"*，一个说
`keywords.py line 443` 的会话含"根因"，但不含"文件"也不含"行"；
另一个说"看了某个文件、某一行"的会话三个词全有。
含糊的非答案排到了精确答案前面。

补上缺失的字面量也修不好，这一点同样测过。oracle 查询——
问题加上金标会话自己的文件名、行号和 commit hash——
只找回 15%：

| 查询 | hit@1 |
|---|---|
| 仅问题 | 0.000 |
| + 3 个 oracle 实体 | 0.150 |

仍然低，有两个原因。**40 个答案里有 23 个根本不含具体 token**
（"下游组件需要改吗？"——答"不需要"），没有东西可补。
剩下 17 个里，BM25 仍有一半以上的时候没把正确会话排在第一。

四个修法都测过并被否掉：BM25 长度归一化（无效果）、
按时间新近打破平局（干净语料上损失 0.25）、
具体性加分（找回一个问题，损失 0.025）、oracle 实体注入。见
[D-020](docs/decisions.md#d-020-bag-of-words-retrieval-degrades-on-dense-same-topic-corpora)
和 [D-021](docs/decisions.md#d-021-oracle-entities-recover-15-the-failure-is-not-missing-literals)。

实际结论：一个主题在语料里
出现一两次时，BM25 够用。出现得更多时，修的地方在写入侧，
不在查询侧。

PrecisionMemBench 就是反例：35 条信念，主语几乎一样，
同一个词 `redis` 出现在两条不同记忆里。一旦记忆带上作用域标签、
替代链接和关系 refs，一旦索引里放的是信念的名字而不是它的叙述，
同一个 BM25 在那里能拿 0.8312。查询侧的四个修法都测过，
全部失败——BM25 参数扫描、coverage 阈值、分数比、missing-term 比率。

上面的 0.000 是真的。这就是*不带标签*的 BM25 面对一个
每篇文档都讲同一主题的语料时的表现。读前五条，
把 `corpus_missing` 看得比原始分数重，给记忆打标签——
作用域、kind、谁替代谁。三个独立测量都指向同一件事，
所以这个系统返回证据而不是结论。

### 弃答

在 `corpus_missing` 的自然工作点上：

| 策略 | 保留 | 弃答 |
|---|---|---|
| 全部回答 | 1.000 | 0.000 |
| **corpus_missing** | **0.875** | **0.632** |
| 本地 9B 模型 | 0.950 | 0.895 |
| 托管 `deepseek-flash` | 0.925 | 1.000 |

0.875 的保留率意味着 12.5% 的可答问题会被自动拒掉。
这个水平不足以替调用方做决定，所以信号只报告，不执行。

### 托管 judge

如果需要内置弃答，托管模型是唯一不占内存的选项：

| 方案 | 保留 | 弃答 | 延迟 | 常驻内存 |
|---|---|---|---|---|
| 仅 BM25 | 1.000 | 0.000 | 0.22 ms | 0 |
| 本地 2B | 0.750 | 1.000 | 0.77 s | ~1 GB |
| 本地 9B | 0.950 | 0.895 | 1.5–2 s | 15.2 GB |
| `deepseek-flash` | 0.925 | 1.000 | 0.81 s | 0 |

一个任务里查 30 次记忆，0.81 s × 30 就是多出 24 秒，
花在调用方 agent 本来就能边推理边做的判断上。

### 其中有多少是样本噪声

77 个用例是很小的基准，小基准给出的区间很宽。
所以本页每个数字都配一个 Wilson 95% 区间：

| 结果 | 分数 | 95% CI | 宽度 |
|---|---|---|---|
| PrecisionMemBench **active**，本存储 | 34/43 = 0.7907 | 0.6479 – 0.8858 | 23.8 pp |
| PrecisionMemBench **active**，open-knowledge-format | 18/43 = 0.4186 | 0.2838 – 0.5667 | 28.3 pp |
| PrecisionMemBench **total**，本存储 | 64/77 = 0.8312 | 0.7323 – 0.8986 | 16.6 pp |
| PrecisionMemBench **total**，open-knowledge-format | 36/77 = 0.4675 | 0.3603 – 0.5779 | 21.8 pp |
| 中文 hit@1（40 个可答） | 37/40 = 0.9250 | 0.8014 – 0.9742 | 17.3 pp |
| LoCoMo hit@1 | 960/1531 = 0.6270 | 0.6025 – 0.6509 | 4.8 pp |
| LoCoMo hit@5 | 1355/1531 = 0.8850 | 0.8681 – 0.9001 | 3.2 pp |

区间很宽，这就是 77 个用例的诚实代价。
比较在**两个**分母上都成立，这一点重要，因为两者不能互换：

- active：下界 0.6479 对 上界 0.5667，相差 8 个点
- total：下界 0.7323 对 上界 0.5779，相差 15 个点

两组都不重叠。中文那个数字要小心对待：37/40 给出 [0.80, 0.97]；
点估计 0.925 不该被当成精确值引用。它是一个方向，不是一个值。

这张表修不了两件事。**PrecisionMemBench 只是一个基准**；
再来一个形状相同的语料，要么确认这个效应，
要么说明它只对这 35 条信念成立。另外，
**没有任何独立第三方复现过 0.8312**——数据和 harness 都在仓库里，
所以它可复现，但"我能复现"不等于"别人复现过"。

## 与另一个记忆 server 并存

**不要在同一个 agent 里启用两个记忆 MCP server。** 实测（D-031）：
`neural-memory` 和 `b4a` 同时启用时，agent 被问到代码风格约定，
去查了另一个 server，什么也没拿到，然后报告"记忆里没有这件事"——
而 `b4a` 里躺着五条相关记忆。存储层无法从内部发现这种情况；
它看起来和一个空记忆一模一样。

一次直接的 MCP 握手证明 `b4a` 是健康的，所以故障出在选择环节，
不在检索。每个 agent 只配一个记忆 server，或者让名字把选择
变得没有歧义。按 server 给 MCP 工具加命名空间的 harness——
mcp__b4a__b4a_recall——碰不到这个问题；撞名需要两个 server
都暴露名字相似的记忆工具。

## 与前沿对比

本项目在当下 agent 记忆版图里的位置。直说，因为只列优点的
README 是推销页。

| 能力 | 本项目的状态 |
|---|---|
| 结构化记忆（带类型的事实、偏好、实体） | **部分实现。** `kind` 是自由文本标签，`tags` 是作用域，`refs` 是链接。没有 schema，没有校验——kind 是什么意思由调用方决定。 |
| 时序 / 图记忆（有效期窗口、关系演化） | **未实现。** 有 `created_at`，`superseded_by` 记录替代关系，但没有有效期区间，查询时也没有图遍历。图建过也测过；在中文语料上什么也没改变（D-024），所以没有接进主路径。`refs` 提供的是一跳扩展，不是遍历。 |
| 冲突消解 | **已实现，确定性。** `supersede(old, new)`；被替代的记忆不再出现。不涉及模型。PrecisionMemBench 替代类别：3/3。 |
| 写入侧去重 / 模式抽取 | **未实现。** 抽取功能存在（`remember()`），但**默认关闭**，因为实测在写入时做会损害检索（D-023）。 |
| 多检索器融合 | **未实现。** 只有一个检索器：BM25，基于二元组，可选加 jieba 词。 |
| 遗忘 / 归并 | **粗糙。** `prune(keep_sessions=N)` 是按新近程度控制体积，不是提升检索。要把噪声语料救回来，得删掉 95%——那是清空，不是遗忘（D-022）。 |
| 跨会话聚合 | **未实现。** 目前明确不在范围内。 |
| 主动召回 | **部分实现。** `recall-context` 从 stdin 读 prompt，打印格式化后的记忆，没有匹配时保持静默——它是为客户端 hook 做的，因为 MCP 只能拉取，server 无法主动献上一条记忆。没有调度器：什么时候跑由客户端决定。见 `integrations/hooks/`。 |
| 多模态 | **未实现。** |
| 幻觉防御 | **部分实现。** 证据与结论分离、`corpus_missing`、judge 的三态结果。没有任何东西能检测出"检索对了但答案错了"。 |
| 公开数字的可复现性 | **强。** 三个基准，其中两个完全不涉及模型。这是本项目领先于大多数同行的那一维，而它只是"不依赖模型"这条规则的副产品，本身不是目标。 |

诚实的总结：**这是一层检索，测量得格外认真，不是认知架构。**
它只做一件事——找到正确的记忆，并清楚说明找到了什么、
没找到什么——其余的一律有意不做。这个取舍对不对，取决于调用方。
已经有模型的 agent，要的是证据和干净的弃答信号，
不是第二个模型对自己记忆的意见。

## 状态

库、CLI 和 MCP server 已经完成。元数据层——作用域标签、
替代链接、关系 refs 和独立的索引文本——是在第一批基准
写完之后才加上的。

**有意不做，也没有计划：**

- **跨会话自动合并。** 检索本来就跨会话——一次查询搜遍所有
  已存会话，返回其中最好的。没做的是把多个会话*综合*成
  一份摘要。那需要模型，或者一条判断"两条记忆在说同一件事"
  的规则；合得不好，会丢掉读者本可自己权衡的细节。
- **多模态。** 不在范围内。

**最近新增：**

- **近重复检测。** 每条记忆一个 SimHash 指纹；两条记忆可能在说
  同一件事时，`search` 返回 `possible_duplicates`。只报告，不合并——
  调用方 agent 自己读两条再决定（D-028）。
- **weight（权重）。** `reinforce` 提高它，`decay` 降低它，
  `prune_by_weight` 删除跌破阈值的记忆。这是本项目第一个
  与时间无关的遗忘信号：被检索到的记忆留下来，
  没被检索到的衰减（D-029）。

**最近修复：**

- **被替代的记忆曾经会消失。** `supersede(old, new)` 把旧的藏起来，
  但新记忆换了词时——"主库是 MySQL"被"迁移到 PostgreSQL"替代——
  查"主库"只匹配到那条被藏起来的，搜索什么也不返回。
  一条记忆在更新的瞬间就忘掉事实，比从来没有过更糟。
  现在命中会重定向到当前记忆，不再直接丢弃（D-027）。

**未实现，在清单上：**

- **按时间衰减的遗忘。** `prune` 是按新近程度控制体积；
  D-022 实测，要把噪声语料救回来得删掉 95%。
  按重要性而不是按时间加权的衰减，是下一件真正要做的事。
- **召回调度器。** `recall-context` 已经给 hook 用了，
  但没有任何东西决定什么时候跑。每轮都跑很便宜（1.5 ms），
  但大概太吵。
- **基于频率的排序。** `accesses` 有计数，`weight` 可以通过
  `weight_blend` 混进排序，但默认关闭。机制测过；它是否*有用*
  没有测过，而让一个未测量的信号改变结果顺序，
  正是这个项目一直拒绝做的那类改动。
- **多检索器融合。** 目前只有一个 BM25。可以再试一个词法检索器，
  不违反"不依赖模型"这条规则。
- **`kind` 的 schema。** 它是自由文本字符串，没有校验。

**测过并否决——不是推迟，是定了：**

- **查询时的图遍历。** 建过，在中文语料上测过，什么也没改变
  （D-024）。一跳 `refs` 扩展确实有用——PrecisionMemBench 上 4/4——
  但那是一条链接，不是遍历。
- **写入时抽取。** 默认关闭，因为它损害检索（D-023）。

**没有模型就做不到：**

- **检测"检索正确、答案错误"。** 这里所有信号都只说明正确的记忆
  有没有回来；没有一个说明 reader 用得对不对。那需要一个 judge，
  而一个会判错的 judge 比没有更糟。

[决策日志](docs/decisions.md)记录了 26 条，两个方向都有——
包括测过并失败的假设。[研究笔记](docs/research.md)和
[版图分析](docs/landscape.md)覆盖周边领域。

已知局限，全部有测量。不带标签的检索在密集近重复上退化（D-020），
查询侧没有任何技巧能救回来（D-021）；修法在写入侧（D-025、D-026）。
LoCoMo hit@1 是 0.6270，没有超过该数据集上公开的最好答案准确率——
而那里的差距在检索，不在阅读。

## 设计决策

26 条决策，每条都附背后的测量或推理，见
[`docs/decisions.md`](docs/decisions.md)。两个方向都记录——
包括测过并失败的假设。

## 参与贡献

验证门禁：

```bash
python -m pytest tests/ -q
python -m ruff check src bench tests
python -m ruff format --check src bench tests
python -m bench.zh.e2e
python -m bench.pmb.run
python -m bench.locomo.run
```

`bench/zh/e2e.py` 跑发布代码路径，应报告 `hit@1 0.9250`。
`bench.pmb.run` 应报告 `pass 64/77`；`bench.locomo.run`
应报告 `hit@1 0.6270`。检索、分词或评分上的改动，
只要让其中任何一个下降，就是回归。后两个不需要 API key，
也不需要网络。

改动某条设计决策，应该在 `docs/decisions.md` 里新增一条，
附上支撑它的测量。

## 特别致谢

感谢 [nhadaututtheky](https://github.com/nhadaututtheky) 的
[neural-memory](https://github.com/nhadaututtheky/neural-memory)。
本项目始于那里的一次 bug 修复
（[PR #228](https://github.com/nhadaututtheky/neural-memory/pull/228)）——
中文查询返回空，因为关键词正则没有汉字范围。修的过程说明
问题出在假设，不在字符类，所以这是对同一个问题的另一个答案，
不是替代品。它的 brain 也是 `bench/nmem/` 的语料——
这里唯一记录了用户真正问了什么的数据集。

## 许可证

Apache-2.0。见 [LICENSE](LICENSE)。
