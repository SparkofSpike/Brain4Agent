# Brain4Agent

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

给 agent 的长期记忆。CJK 优先检索，核心路径不调用模型，
用证据代替置信度分数。

`pip install "Brain4Agent[precise]"` 需要 Python 3.11+，外加一个 20 MB 的可选词典。
不装也完全可用。

---

## 目录

- [是什么](#是什么)
- [安装](#安装)
- [快速开始](#快速开始)
- [工作原理](#工作原理)
- [基准测试](#基准测试)
- [与另一个记忆 server 并存](#与另一个记忆-server-并存)
- [与前沿对比](#与前沿对比)
- [状态](#状态)
- [参与贡献](#参与贡献)
- [特别致谢](#特别致谢)
- [许可证](#许可证)

## 是什么

在代码库上工作的 agent 会积累上下文：某个决定为什么这么做，
哪条路走不通，部署放在哪里。Brain4Agent 把这些轮次存进一个 SQLite 文件，
按会话检索。

- **CJK 开箱可用。** 字符二元组不需要词典，也不需要分词模型。
  实测数字来自中文；日文、韩文和越南文有支持、有测试。
- **核心不调用任何模型。** 检索用 BM25：不需要 API key，
  不需要加速器，没有冷启动，单次查询个位数毫秒。
- **它报告证据，不给结论。** 搜索返回问题里哪些词在语料中从未出现过
  （`corpus_missing`），以及某条会话缺了哪些词（`missing_terms`）。
  这些意味着什么，由调用方判断。
- **存储的内容不等于索引的内容。** 记忆保留全文，索引可以只放更窄、
  更有辨识度的版本。

每条记忆还带 `tags`（作用域）、`superseded_by`（替代关系）
和 `refs`（一跳链接），由写入方提供。

## 安装

```bash
pip install "Brain4Agent[precise]"   # 推荐：加上 jieba，hit@1 0.950
pip install Brain4Agent              # 零依赖：只用二元组，hit@1 0.875
```

Windows 和 macOS 都可用；需要 Python 3.11 或更新版本。
包在 PyPI 上：<https://pypi.org/project/Brain4Agent/>。
要改代码本身，`git clone` 后执行 `pip install -e ".[precise,dev]"`。

`jieba` 是一个 20 MB 的静态词典。不装时，代码回退到二元组。

**Windows 下的非 ASCII 输入。** MCP server 和 CLI 把 stdin/stdout
固定为 UTF-8，所以 CJK 内容不受控制台代码页影响。但命令行*参数*
在那之前就由 Python 解码了——如果要把非 ASCII 文本作为参数传入，
设置 `PYTHONUTF8=1`。

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

一个存储层回答不了的问题：

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

五个工具：

| 工具 | 用途 |
|---|---|
| `b4a_remember` | 存一轮；返回 session id |
| `b4a_recall` | 搜索；返回会话，带 `corpus_missing` 和每条会话的 `missing_terms` |
| `b4a_ask` | 检索并一步作答；需要模型，没配置时会明说 |
| `b4a_session` | 按顺序读完整会话 |
| `b4a_stats` | 计数 |

同一个库也打包成了 agent [skill](integrations/)。

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
检索单位是会话，不是单轮。

批量导入走 `batch()`，每组只提交一次：

```python
with store.batch():
    for turn in turns:
        store.add(turn.text, session_id=turn.sid, role=turn.role)
```

### 可选的托管 judge

默认关闭。只有存在 key 时才启用。

```python
from brain4agent import Judge, MemoryStore

judge = Judge.from_env()          # 读取 BRAIN4AGENT_JUDGE_KEY；未设置时为 None
if judge:
    r = store.search_judged("question", judge)
    print(r.verdict)              # found / not_found / uncertain
```

`uncertain` 是独立于 `not_found` 的第三种状态：judge 超时不等于记忆不存在。

## 工作原理

```
                写                             读
                  │                              │
          二元组 + 词典词                   词典词
                  │                              │
                  ▼                              ▼
          ┌──────────────────────────────────────────┐
          │  SQLite: id, session, role, content,     │
          │          created_at, tokens              │
          └──────────────────────────────────────────┘
                  │                              │
                  ▼                              ▼
           BM25 索引（内存中）  →  会话级证据
                                    + corpus_missing
                                    + 每条会话的 coverage
```

### 存储

一个 SQLite 文件。行是轮次，检索单位是会话。索引 token 在写入时算好，
缓存在列里。

```
                       导入      重建     查询
127 轮    44 会话               3.9 ms   1.8 ms
1016 轮   352 会话    0.25 s   29 ms    3.7 ms
```

写入按批提交。1016 轮的导入，逐条提交要 3.8 秒，放在 `batch()` 里是 0.25 秒。

「重建」那一列是在 D-041 和 D-042 之前测的，那两次分别加了短语词对索引和
序列缓存。在同一个机器、同一个真实库上重测，现在的重建是 **127 轮 18.0 毫秒、
1016 轮 101 毫秒、2925 轮 393 毫秒**——是上表的三到四倍。表格留着是因为形状
仍然成立、导入列仍是现值；凡是依赖打开延迟的判断请用新数字（D-043）。

之所以是 SQLite，是因为写入而不是读取：检索发生在内存索引里，而每次搜索都要
提交一次访问计数；MCP server 和 CLI 也能同时持有同一个文件。WAL 模式意味着
「一个文件」在磁盘上是三个——`brain.db`、`brain.db-wal`、`brain.db-shm`——快照
必须走 SQLite 的 backup API，文件复制会漏掉日志、丢掉已提交的行。

要知道的成本不在文件上。一次本地写入会把 store 标脏，下一次搜索重建整个索引：
2925 条记忆时是 **402 毫秒**，而预热后的关键词查询是 4 毫秒。常驻内存是另一个
上限——每个索引 token 约 145 字节，10 万条约 1.2 GB。两者都属于内存索引层，
不属于 SQLite（D-043）。

### 读写分词器分离

索引和查询用不同的分词器。3×3 网格实测（D-019），以及之后的两次改进（D-037）：

| 索引 | 查询 | hit@1 | MRR |
|---|---|---|---|
| bigram | bigram | 0.825 | 0.891 |
| jieba | jieba | 0.875 | 0.911 |
| 二元组 + 词 | 词 | 0.925 | 0.940 |
| **二元组 + 词** | **词 − 疑问词** | **0.950** | **0.961** |

最后一行在 3×3 网格之外多做了两件事：查询侧去掉疑问词，会话分数取最好的
三轮折叠。

### 证据，而非结论

- **`corpus_missing`**——问题里的实词在整个已存语料中从未出现。
- **`missing_terms`**（每条会话）——这条会话缺了哪些问题词。
- **`term_frequencies`**——每个问题词出现在多少条会话里。一个词出现在 2898 条
  记忆中的 453 条，既不是缺失也不是拼错，而是太泛、区分不了任何东西，查询需要
  再加一个词（D-039）。
- **`related_terms`**——top 会话里用了、而查询没说的词，最多五个。这就是两词
  查询需要的第三个词，从语料里取，不是猜的（D-040）。

是否拒答由调用方决定；存储层只报告事实。

## 基准测试

所有数字都能用 `bench/` 下的脚本复跑，脚本导入的是发布代码，不是副本。

### LongMemEval（英文）— 检索

`s` 变体上 470 个可答问题（平均 haystack 493 轮）：

```
hit@1    0.8660
hit@5    0.9511
hit@10   0.9660
MRR      0.8990
```

`python -m bench.run_longmemeval --dataset s`。`bench/data/` 由
`bench/fetch_data.py` 拉取（264 MB，不进 git）。

三个基准放在一起：

| 基准 | 指标 | 本存储 | 最好的第三方 |
|---|---|---|---|
| LongMemEval `s` | hit@10 | 0.9660 | — |
| PrecisionMemBench | active | **42/43** | 18/43（open-knowledge-format） |
| PrecisionMemBench | total | **74/77** | 36/77（open-knowledge-format） |
| LoCoMo | hit@1 | **0.7054** | 不可比 |

### PrecisionMemBench — 检索精度

77 个单轮用例，覆盖 35 条信念。不涉及生成模型，也没有 judge：
答案就是存储返回的信念 ID 集合。`active` 是诚实的口径，
因为 77 个用例里有 33 个空库也能通过。

```
active  42/43 = 0.9767     需要查询相关检索的用例
 total  74/77 = 0.9610     包含任何存储都能过的用例
```

这里提升了两处：结果集大小改为按查询中出现的名称数量决定，不再对八个词以下的
查询硬编码 `top_k=1`；名称只统计当前 scope 可见的信念。名称匹配从子串改为整词，
`pipe` 不会再命中 `pipeline`。另外加入了一次编辑距离的模糊匹配
（`typscript` → `typescript`，`mongod` → `mongodb`）。见
[D-038](docs/decisions.md#d-038-sizing-the-result-set-and-matching-one-edit-away)。

官方表格，来自基准自己的站点（`tenureai.dev/benchmark`）：

```
System                  Active    Total
----------------------  ------    -----
tenure (the authors)    43/43     77/77
this store              42/43     74/77
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

两列都是第三方最好成绩。基准自带的参考实现是 43/43 和 77/77。

### LoCoMo — 检索

1540 个问题，272 个会话，5882 轮。

```
hit@1   0.7054
hit@5   0.9086
hit@10  0.9549
MRR     0.7921
```

`python -m bench.locomo.run`——不涉及模型，数据已提交。
这些是会话检索命中率，不是 LLM judge 下的答案准确率，
所以和公开的 LoCoMo 数字不可比。

索引单轮而不是会话，会损失 2.5 个点的 hit@1（0.6022 → 0.6310），分词器和
评分不变。

按类别（hit@1）：open-domain 0.7919，temporal 0.6906，single-hop 0.5445，
multi-hop 0.4494。后两类是在单个会话内找一个事实，这也是
[D-042](docs/decisions.md#d-042-proximity-where-the-query-words-sit-not-just-whether-they-appear)
的近邻加权对短查询自动关闭的原因：它帮一个点名事件的问题，伤一个点名事实的问题。

### 中文基准

59 个问题（40 个可答，19 个不可答），44 个会话。

| 指标 | bigram/bigram | 当前 |
|---|---|---|
| hit@1 | 0.825 | **0.950** |
| hit@5 | 0.950 | **1.000** |
| hit@10 | 1.000 | 1.000 |
| MRR | 0.891 | **0.961** |

`当前` = 索引 `二元组 + 词`，查询 `词 − 疑问词`，会话取最好的三轮折叠（D-037）。
hit@10 已经饱和；只有 hit@1 和 MRR 有区分度。

### 这个数字在哪里失效

**0.950 描述的是 44 个会话。密集的近重复会让它崩掉。**

| 语料 | hit@1 | hit@5 |
|---|---|---|
| 44 个会话 | 0.950 | 1.000 |
| +300 个同主题会话 | 0.000 | 0.000 |

问「根因在哪个文件哪一行」时，一个说「看了某个文件、某一行」的含糊会话，
会排到说 `keywords.py line 443` 的会话前面。补上金标会话自己的文件名、
行号和 commit hash，只找回 15%：

| 查询 | hit@1 |
|---|---|
| 仅问题 | 0.000 |
| + 3 个 oracle 实体 | 0.150 |

读前五条，把 `corpus_missing` 看得比原始分数重，给记忆打标签——
作用域、kind、谁替代谁。修法在写入侧；查询侧试过的四种都在
[D-020](docs/decisions.md#d-020-bag-of-words-retrieval-degrades-on-dense-same-topic-corpora)
和 [D-021](docs/decisions.md#d-021-oracle-entities-recover-15-the-failure-is-not-missing-literals)。

### 弃答

在 `corpus_missing` 的自然工作点上：

| 方案 | 保留 | 弃答 | 延迟 | 常驻内存 |
|---|---|---|---|---|
| 仅 BM25 | 1.000 | 0.000 | 0.22 ms | 0 |
| **corpus_missing** | **0.875** | **0.632** | ~0 | 0 |
| 本地 2B | 0.750 | 1.000 | 0.77 s | ~1 GB |
| 本地 9B | 0.950 | 0.895 | 1.5–2 s | 15.2 GB |
| 托管 `deepseek-flash` | 0.925 | 1.000 | 0.81 s | 0 |

0.875 的保留率意味着 12.5% 的可答问题会被自动拒掉，所以信号只报告，
不自动执行。托管 judge 每次查询多 0.81 秒——一个任务查 30 次就是 24 秒。

## 与另一个记忆 server 并存

**不要在同一个 agent 里启用两个记忆 MCP server。** `neural-memory` 和
`b4a` 同时启用时，agent 去查了另一个 server，报告「记忆里没有这件事」，
而 `b4a` 里躺着五条相关记忆。每个 agent 只配一个记忆 server，
或者让名字把选择变得没有歧义。

## 与前沿对比

| 能力 | 本项目的状态 |
|---|---|
| 结构化记忆（带类型的事实、偏好、实体） | **部分实现。** `kind` 是自由文本标签，`tags` 是作用域，`refs` 是链接。没有 schema，没有校验。 |
| 时序 / 图记忆（有效期窗口、关系演化） | **未实现。** 只有 `created_at` 和 `superseded_by`；没有有效期区间，查询时也没有图遍历。一跳 `refs` 在 PrecisionMemBench 上是 4/4。 |
| 冲突消解 | **已实现。** `supersede(old, new)`；被替代的记忆不再出现。PrecisionMemBench 替代类别：3/3。 |
| 写入侧去重 / 模式抽取 | **未实现。** 抽取功能存在（`remember()`），**默认关闭**（D-023）。 |
| 多检索器融合 | **未实现。** 只有一个检索器：BM25。 |
| 遗忘 / 归并 | **粗糙。** `prune(keep_sessions=N)` 是按新近程度控制体积；`decay` 和 `prune_by_weight` 加了权重信号。要把噪声语料救回来，得删掉 95%（D-022）。 |
| 跨会话聚合 | **未实现。** |
| 主动召回 | **部分实现。** `recall-context` 供客户端 hook 使用；没有调度器。 |
| 多模态 | **未实现。** |
| 幻觉防御 | **部分实现。** 证据与结论分离、`corpus_missing`、judge 三态。没有任何东西能检测「检索对了但答案错了」。 |
| 公开数字的可复现性 | **强。** 三个基准，其中两个完全不涉及模型。 |

## 状态

库、CLI 和 MCP server 已经完成。

**新增：** 近重复检测（`duplicates_of`，SimHash；只报告不合并，
不放在检索路径上）和 weight（`reinforce`、`decay`、`prune_by_weight`）。
检索侧会从查询里去掉疑问词，把会话最好的三轮折叠成一个分数，并对一个
语料里从未出现过的英文词做一次编辑距离的模糊匹配（D-037、D-038）。

**未实现：** 跨会话综合、多模态、`kind` 的 schema、多检索器融合、
召回调度器、按时间衰减的遗忘。

**测过并否决：** 查询时图遍历（D-024）、写入时抽取（D-023）、把 LoCoMo 会话
切块检索（D-037）、用 store 自己的分词器跑 LongMemEval（D-037）。

发布历史见 [CHANGELOG.md](CHANGELOG.md)。[决策日志](docs/decisions.md)记录了
43 条；[研究笔记](docs/research.md)和[版图分析](docs/landscape.md)覆盖周边领域。

## 参与贡献

```bash
python -m pytest tests/ -q
python -m ruff check src bench tests
python -m ruff format --check src bench tests
python -m bench.zh.e2e
python -m bench.pmb.run
python -m bench.locomo.run
```

`bench/zh/e2e.py` 应报告 `hit@1 0.9500`；`bench.pmb.run` 应报告 `pass 74/77`；
`bench.locomo.run` 应报告 `hit@1 0.7054`。低于这些就是回归。后两个不需要
API key，也不需要网络。改动某条设计决策，应该在 `docs/decisions.md` 里
新增一条，附上支撑它的测量。

## 特别致谢

感谢 [nhadaututtheky](https://github.com/nhadaututtheky) 的
[neural-memory](https://github.com/nhadaututtheky/neural-memory)。
本项目始于那里的一次 bug 修复：
中文查询返回空，因为关键词正则没有汉字范围。它的 brain 也是
`bench/nmem/` 的语料。

## 许可证

Apache-2.0。见 [LICENSE](LICENSE)。
