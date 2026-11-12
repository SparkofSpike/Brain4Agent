# Brain4Agent

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

> **Agents: start here.** [FOR_AGENTS.md](FOR_AGENTS.md) is this page written
> for you — what the store does for an agent, how to install it, how to call
> it, and where it stops being useful.

Long-term memory for agents. CJK-first retrieval, no model in the core path,
and evidence instead of confidence scores.

`pip install "Brain4Agent[precise]"` needs Python 3.11+, plus a 20 MB optional
dictionary. Everything works without it.

---

## Contents

- [What it is](#what-it-is)
- [Install](#install)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Running alongside another memory server](#running-alongside-another-memory-server)
- [Against the frontier](#against-the-frontier)
- [Status](#status)
- [Contributing](#contributing)
- [Special thanks](#special-thanks)
- [License](#license)

## What it is

An agent working on a codebase accumulates context: why a decision was made,
which approach failed, where a deployment lives. Brain4Agent stores those turns
in one SQLite file and retrieves them by session.

- **CJK works out of the box.** Character bigrams need no dictionary and no
  segmentation model. The benchmarked numbers are Chinese; Japanese, Korean and
  Vietnamese are supported and tested.
- **The core makes no model calls.** Retrieval is BM25: no API key, no
  accelerator, no cold start, single-digit milliseconds per query.
- **It reports evidence, not verdicts.** A search returns which question words
  the corpus never contained (`corpus_missing`) and which ones a given session
  lacked (`missing_terms`). The caller decides what that means.
- **What is stored is not what is indexed.** A memory keeps its full text; the
  index can hold a narrower, more distinctive version.

Memories also carry `tags` (scope), `superseded_by` (replacement) and `refs`
(one-hop links), supplied by the writer.

## Install

```bash
pip install "Brain4Agent[precise]"   # recommended: adds jieba, hit@1 0.950
pip install Brain4Agent              # zero dependencies: bigrams only, hit@1 0.875
```

Windows and macOS both work; Python 3.11 or newer is required. The package is on
PyPI as <https://pypi.org/project/Brain4Agent/>. To work on the code itself,
`git clone` and `pip install -e ".[precise,dev]"`.

`jieba` is a 20 MB static dictionary. Without it the code falls back to bigrams.

**Windows, non-ASCII input.** The MCP server and the CLI pin their stdin/stdout
to UTF-8, so CJK content works regardless of the console code page. Command
line *arguments* are decoded by Python before any of that runs, though — if you
pass non-ASCII text as an argument, set `PYTHONUTF8=1`.

## Quick start

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

A question the store cannot answer:

```bash
$ brain4agent search "杭州测试机上装的是什么数据库" -k 1
1 sessions for '杭州测试机上装的是什么数据库'
  corpus has no: 数据库, 机上
  [ 4.81 cov=0.28] c060a5294037  session lacks: 数据库, 机上
```

Bulk-load an existing history from JSONL (one object per line, `content`
required):

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

Five tools:

| Tool | Purpose |
|---|---|
| `b4a_remember` | Store one turn; returns a session id |
| `b4a_recall` | Search; returns sessions with `corpus_missing` and per-session `missing_terms` |
| `b4a_ask` | Retrieve and answer in one step; needs a model, and says so when none is configured |
| `b4a_session` | Read a whole session in order |
| `b4a_stats` | Counts |

The same library is also packaged as an agent [skill](integrations/).

### Library

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

`add_exchange(user=..., assistant=...)` stores a pair as one session. Retrieval
is session-level, not turn-level.

Bulk imports go through `batch()`, which defers the commit to one per group:

```python
with store.batch():
    for turn in turns:
        store.add(turn.text, session_id=turn.sid, role=turn.role)
```

### Optional hosted judge

Off by default. Enabled only when a key is present.

```python
from brain4agent import Judge, MemoryStore

judge = Judge.from_env()          # reads BRAIN4AGENT_JUDGE_KEY; None if unset
if judge:
    r = store.search_judged("question", judge)
    print(r.verdict)              # found / not_found / uncertain
```

`uncertain` is a third state, distinct from `not_found`: a judge that times out
is not evidence that the memory does not exist.

## How it works

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

### Storage

One SQLite file. Rows are turns; the unit of retrieval is the session. Index
tokens are computed at write time and cached in a column, so rebuilding the
in-memory index is a counting exercise.

```
                       import    rebuild   query
127 turns   44 sessions          3.9 ms   1.8 ms
1016 turns  352 sessions  0.25 s  29 ms   3.7 ms
```

Writes commit in batches. A 1,016-turn import takes 3.8 s one commit at a time
and 0.25 s inside `batch()`.

Rebuild also builds the phrase pair index when English summaries are in play
(D-041), which roughly doubles that column: 8.9 ms for the 127-turn corpus and
60 ms for 1,016 turns on the machine this was last measured on.

### Read/write tokenizer split

Indexing and querying use different tokenizers. Measured over a 3×3 grid of
index strategy against query strategy (D-019), then two changes on top of the
best cell (D-037):

| index | query | hit@1 | MRR |
|---|---|---|---|
| bigram | bigram | 0.825 | 0.891 |
| jieba | jieba | 0.875 | 0.911 |
| bigrams + words | words | 0.925 | 0.940 |
| **bigrams + words** | **words − question words** | **0.950** | **0.961** |

The last row adds two things the 3×3 grid did not test: interrogatives are
removed from the query side, and a session's best three turns are folded into
one score. Both are measured in
[D-037](docs/decisions.md#d-037-query-side-question-words-and-session-folding);
the first 3×3 grid is [D-019](docs/decisions.md#d-019-index-wide-query-narrow).

English gets the same treatment one language over (D-041): query-side stopwords
(`when`, `the`, `for`), inflection folding on both sides (`camped` matches
`camping`), and a small bonus when two adjacent query words appear adjacent in a
memory (`pottery class` outranks the same two words scattered). The stemmer
runs at index time, not in the stored token column, so an existing database
needs no migration. Chinese is untouched by all three — its bigrams already
index adjacency, and the rules refuse non-ASCII terms.

### Where the words sit

BM25 counts occurrences. It cannot tell a session that is *about* an event from
one that mentions the same words in passing. On LoCoMo's 486 misses, 417 had the
right session inside the top ten and **188 had it at rank 2**, usually within a
few percent of the winner (D-042).

So a document also earns a bonus for keeping the question's words together: the
fraction of distinct query terms that share a window of 40 index tokens,
multiplied into the score. It is not coverage reweighted — coverage asks
*whether* a session has the words, and four separate attempts to use it as a
post-hoc term were flat or worse (D-020, D-024, D-039, D-041). This asks
*where* they are, which is a fact the index throws away.

Three guards, each from a measurement that regressed without it: only ASCII
words take part (Chinese bigrams already index adjacency), documents under
three windows are skipped (a one-sentence belief has no passage), and queries
under five terms are skipped (two to four words is a lookup — on a real store
the bonus cost 0.043 hit@1 there and nothing anywhere else, D-042).

### Evidence, not verdicts

- **`corpus_missing`** — a content word from the question appears nowhere in the
  stored corpus.
- **`missing_terms`** (per session) — which question words that session lacked.
- **`term_frequencies`** — how many sessions each question word appears in. A
  word present in 453 of 2898 memories is neither missing nor misspelt; it is
  too common to separate anything, and the query needs another word (D-039).
- **`related_terms`** — up to five words the top sessions use that the query did
  not. The third word a two-word query needs, taken from the corpus (D-040).

Whether to refuse is the caller's decision; the store reports the fact.

## Benchmarks

All numbers are reproducible from the `bench/` scripts, which import the
shipping code rather than a copy of it.

### LongMemEval (English) — retrieval

470 answerable questions over the `s` variant (average haystack 493 turns):

```
hit@1    0.8660
hit@5    0.9511
hit@10   0.9660
MRR      0.8990
```

`python -m bench.run_longmemeval --dataset s`. `bench/data/` is fetched by
`bench/fetch_data.py` (264 MB, not committed).

These use the harness's bigram/bigram configuration. The store's own
configuration (wide index, words minus question words, fuzzy) scored 0.8596 on
the same 500 tasks and took 8× longer, so LongMemEval keeps bigrams — measured,
not assumed.

Across three benchmarks:

| benchmark | metric | this store | best third-party |
|---|---|---|---|
| LongMemEval `s` | hit@10 | 0.9660 | — |
| PrecisionMemBench | active | **42/43** | 18/43 (open-knowledge-format) |
| PrecisionMemBench | total | **74/77** | 36/77 (open-knowledge-format) |
| LoCoMo | hit@1 | **0.7054** | not comparable |

### PrecisionMemBench — retrieval precision

77 single-turn cases over 35 beliefs. No generation model and no judge: the
answer is the set of belief IDs the store returns. `active` is the honest
column, because 33 of the 77 cases pass with an empty store.

```
active  42/43 = 0.9767     cases that need query-dependent retrieval
 total  74/77 = 0.9610     including cases any store passes
```

Three changes lifted this from 34/43 and 64/77. The result set is now sized from
the names a query contains at every length, and only from beliefs the case's
scope can see — the old rule forced `top_k=1` below eight words and counted
out-of-scope beliefs. Whole-word name matching replaced substring matching, so
`pipe` no longer matches inside `pipeline`. A fuzzy query path then matches a
misspelt English word against a corpus word one edit away (`typscript` →
`typescript`, `mongod` → `mongodb`). See
[D-038](docs/decisions.md#d-038-sizing-the-result-set-and-matching-one-edit-away).
Two more cases came with the English query folding in
[D-041](docs/decisions.md#d-041-english-query-folding-stopwords-stems-and-adjacency):
a query's grammar is not a memory, so dropping `what` and `the` from the query
side keeps a scope-bleed belief out of the result set.

Official table, from the benchmark's own site (`tenureai.dev/benchmark`):

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

Best third-party result on both columns. The benchmark's own reference
implementation scores 43/43 and 77/77.

### LoCoMo — retrieval

1540 questions over 272 sessions, 5882 turns.

```
hit@1   0.7054
hit@5   0.9086
hit@10  0.9549
MRR     0.7921
```

`python -m bench.locomo.run` — no model, data committed. These are session
retrieval rates, not answer accuracy under an LLM judge, so they are not
comparable to published LoCoMo numbers.

Indexing turns instead of sessions costs 2.5 points of hit@1 (0.6022 → 0.6310),
same tokenisers and scoring — measured before D-041 and D-042 raised both.

By category (hit@1): open-domain 0.7919, temporal 0.6906, single-hop 0.5445,
multi-hop 0.4494. The last two are lookups of a single fact inside a session,
which is why the proximity bonus in [D-042](docs/decisions.md#d-042-proximity-where-the-query-words-sit-not-just-whether-they-appear)
turns itself off for short queries: it helps a question that names an event and
hurts one that names a fact.

### Chinese benchmark

59 questions (40 answerable, 19 unanswerable), 44 sessions.

| metric | bigram/bigram | current |
|---|---|---|
| hit@1 | 0.825 | **0.950** |
| hit@5 | 0.950 | **1.000** |
| hit@10 | 1.000 | 1.000 |
| MRR | 0.891 | **0.961** |

`current` is index `bigrams + words`, query `words minus interrogatives`, and
sessions folded from their best three turns (D-037). hit@10 is saturated; only
hit@1 and MRR discriminate.

### Where the Chinese number stops holding

**0.950 describes 44 sessions. Dense near-duplicates collapse it.**

| corpus | hit@1 | hit@5 |
|---|---|---|
| 44 sessions | 0.950 | 1.000 |
| +300 same-topic sessions | 0.000 | 0.000 |

Asked which file and line caused a bug, a vague session saying "some file, some
line" outranks the one saying `keywords.py line 443`. Supplying the gold
session's own file names, line numbers and commit hashes recovers 15%:

| query | hit@1 |
|---|---|
| question only | 0.000 |
| + 3 oracle entities | 0.150 |

Read the top five, weight `corpus_missing` over raw score, and tag your
memories — scope, kind, what supersedes what. The fix is at write time; four
query-side attempts are in [D-020](docs/decisions.md#d-020-bag-of-words-retrieval-degrades-on-dense-same-topic-corpora)
and [D-021](docs/decisions.md#d-021-oracle-entities-recover-15-the-failure-is-not-missing-literals).

### Abstention

At the natural operating point of `corpus_missing`:

| approach | keep | refuse | latency | resident |
|---|---|---|---|---|
| BM25 only | 1.000 | 0.000 | 0.22 ms | 0 |
| **corpus_missing** | **0.875** | **0.632** | ~0 | 0 |
| local 2B | 0.750 | 1.000 | 0.77 s | ~1 GB |
| local 9B | 0.950 | 0.895 | 1.5–2 s | 15.2 GB |
| hosted `deepseek-flash` | 0.925 | 1.000 | 0.81 s | 0 |

A 0.875 keep rate means 12.5% of answerable questions would be refused
automatically, so the signal is reported rather than acted on. A hosted judge
adds 0.81 s per lookup — 24 seconds across 30 lookups in one task.

## Running alongside another memory server

**Do not enable two memory MCP servers in the same agent.** With
`neural-memory` and `b4a` both loaded, an agent queried the other server and
reported "memory has no record of this" while five relevant memories sat in
`b4a`. Use one memory server per agent, or names that make the choice
unambiguous.

## Against the frontier

| Capability | Status here |
|---|---|
| Structured memory (typed facts, preferences, entities) | **Partial.** `kind` is a free-form label, `tags` are scopes, `refs` are links. No schema, no validation. |
| Temporal / graph memory (validity windows, relation evolution) | **Not built.** `created_at` and `superseded_by` only; no validity interval, no query-time traversal. One-hop `refs` scores 4/4 on PrecisionMemBench. |
| Conflict resolution | **Built.** `supersede(old, new)`; the replaced memory stops surfacing. PrecisionMemBench supersession category: 3/3. |
| Write-path deduplication / pattern extraction | **Not built.** Extraction exists (`remember()`) and is **off by default** (D-023). |
| Multi-retriever fusion | **Not built.** One retriever: BM25. |
| Forgetting / consolidation | **Crude.** `prune(keep_sessions=N)` is recency-based size control; `decay` and `prune_by_weight` add a weight signal. Recovering a noisy corpus needs deleting 95% of it (D-022). |
| Cross-session aggregation | **Not built.** |
| Proactive recall | **Partial.** `recall-context` for client-side hooks; no scheduler. |
| Multimodal | **Not built.** |
| Hallucination defence | **Partial.** Evidence/verdict separation, `corpus_missing`, three-state judge. Nothing detects a wrong answer produced from a correct retrieval. |
| Reproducibility of published numbers | **Strong.** Three benchmarks, two with no model in the loop. |

## Status

Library, CLI and MCP server are complete.

**New:** near-duplicate detection (`duplicates_of`, SimHash; reports, does not
merge, and is not on the search path) and weight (`reinforce`, `decay`,
`prune_by_weight`). Retrieval drops question words from the query side, folds a
session's best three turns into one score, and matches an unknown English word
one edit away from a corpus word (D-037, D-038). English queries also drop
stopwords, fold inflections, and reward word adjacency (D-041) — LoCoMo hit@1
0.6310 → 0.6826 and PrecisionMemBench total 72/77 → 74/77, with the Chinese
number unchanged. A session is also rewarded for keeping the question's words
in one passage (D-042) — LoCoMo hit@1 0.6826 → 0.7054 and MRR 0.7784 → 0.7921,
with the Chinese benchmark, PrecisionMemBench, LongMemEval and keyword queries
against the real store all measured unchanged.

**Not built:** cross-session synthesis, multimodal, schema on `kind`,
multi-retriever fusion, a recall scheduler, time-decayed forgetting.

**Measured and rejected:** query-time graph traversal (D-024), write-time
extraction (D-023), chunking LoCoMo sessions for retrieval, running LongMemEval
on the store's own tokenisers (D-037), chunking LoCoMo sessions (D-037),
minimum-span and early-position re-ranks (D-042).

Release history is in [CHANGELOG.md](CHANGELOG.md). The
[decision log](docs/decisions.md) records 42 entries; the
[research notes](docs/research.md) and [landscape analysis](docs/landscape.md)
cover the field.

## Contributing

```bash
python -m pytest tests/ -q
python -m ruff check src bench tests
python -m ruff format --check src bench tests
python -m bench.zh.e2e
python -m bench.pmb.run
python -m bench.locomo.run
```

`bench/zh/e2e.py` should report `hit@1 0.9500`; `bench.pmb.run` `pass 74/77`;
`bench.locomo.run` `hit@1 0.7054`. Lower is a regression. The last two need no
API key and no network. Changes to a design decision belong in
`docs/decisions.md` as a new entry, with the measurement that justifies it.

## Special thanks

To [nhadaututtheky](https://github.com/nhadaututtheky) for
[neural-memory](https://github.com/nhadaututtheky/neural-memory). This project
began as a bug fix there: a Chinese query returned nothing because the keyword
regex had no Han range. Its brain is also the corpus for `bench/nmem/`.

## License

Apache-2.0. See [LICENSE](LICENSE).
