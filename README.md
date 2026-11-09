# Brain4Agent

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

Long-term memory for agents. CJK-first retrieval, no model in the core path,
and evidence instead of confidence scores.

`pip install "Brain4Agent[precise]"` requires Python 3.11+ and one optional
dictionary. Without it, everything still works.

---

## Contents

- [What it is](#what-it-is)
- [Install](#install)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Benchmarks](#benchmarks)
- [How much of this is sampling noise](#how-much-of-this-is-sampling-noise)
- [Against the frontier](#against-the-frontier)
- [Status](#status)
- [Design decisions](#design-decisions)
- [Contributing](#contributing)
- [Special thanks](#special-thanks)
- [License](#license)

## What it is

An agent that works on a codebase accumulates context: why a decision was made,
which approach failed, where a deployment lives. That context outlives the
conversation it came from, and re-deriving it is expensive.

Brain4Agent stores those turns in a single SQLite file and retrieves them by
session. It is designed around four constraints:

- **CJK works out of the box.** Segmenting Han, kana or Hangul with a whitespace
  or non-word boundary collapses a query to one token, and halfwidth katakana
  matches neither a CJK range nor an ASCII word class. Character bigrams fail
  in none of those ways. The measured numbers are Chinese (see Benchmarks);
  Japanese, Korean and Vietnamese are supported and tested but not benchmarked,
  and the README does not pretend otherwise.
- **The core makes no model calls.** Retrieval is BM25. No API key, no
  accelerator, no cold start, single-digit milliseconds per query.
- **It reports evidence, not verdicts.** Instead of a confidence score, a search
  returns which words in the question the corpus never contained. Deciding what
  that means is left to the caller, which already has a model.
- **What is stored is not what is indexed.** A memory keeps its full text for a
  human or a reader to consume; the index can hold something narrower and more
  distinctive. Separating the two moved PrecisionMemBench from 0.6104 to 0.7922
  with no other change, because explanatory prose shares its ordinary words with
  the whole domain and outranks the one distinctive name.

Memories also carry `tags` (scope), `superseded_by` (replacement) and `refs`
(one-hop links). None of these is inferred and none needs a model — they are facts
the writer knows and the text does not say.

## Install

```bash
pip install "Brain4Agent[precise]"   # recommended: adds jieba, hit@1 0.925
pip install Brain4Agent              # zero dependencies: bigrams only, hit@1 0.825
```

Windows and macOS both work; Python 3.11 or newer is required. The package is on
PyPI as <https://pypi.org/project/Brain4Agent/>. To work on the code itself,
`git clone` and `pip install -e ".[precise,dev]"`.

`jieba` is a ~20 MB static dictionary. It is not required — the code degrades to
bigrams on both sides of retrieval and keeps working.

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

The `b4a_` prefix is deliberate: names like `remember` or `recall` collide with
other MCP servers loaded into the same client.

The server is a standard-library implementation of MCP over stdio — four methods
of newline-delimited JSON-RPC. The official SDK pulls in a large dependency tree
to serialise four dictionaries; see
[D-017](docs/decisions.md#d-017-mcp-implemented-against-the-protocol-not-the-sdk).

The same library is also packaged as an agent [skill](integrations/) — a document
the agent reads, then calls the CLI.

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
is session-level, not turn-level: an answer often spans an exchange, and judging
single turns produces far more false negatives.

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

`uncertain` is a distinct third state. A judge that times out or returns 503 is
not evidence that the memory does not exist; collapsing the two would turn an
outage into a confident "I never recorded this".

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
in-memory index is a counting exercise rather than a re-segmentation.

Each row also carries `tags`, `superseded_by`, `refs` and an optional separate
index text — metadata the caller supplies at write time and the store filters and
expands on at read time. These are what make PrecisionMemBench solvable at all;
without them the same BM25 scores 0.3766.

Writes commit in batches. A 1,016-turn import took 3.8 s one commit at a time
and 0.25 s inside `batch()` — a thousand individual commits costs more than the
thousand writes.

```
                       import    rebuild   query
127 turns   44 sessions          3.9 ms   1.8 ms
1016 turns  352 sessions  0.25 s  29 ms   3.7 ms
```

### Read/write tokenizer split

Indexing and querying are different jobs. Writing wants every clue kept —
including bigrams, which give unknown words a matching path. Querying wants
noise dropped — no `州测` from `杭州测试机`.

Measured over a 3×3 grid of index strategy against query strategy:

| index | query | hit@1 | MRR |
|---|---|---|---|
| bigram | bigram | 0.825 | 0.891 |
| jieba | jieba | 0.875 | 0.911 |
| **bigrams + words** | **words** | **0.925** | **0.940** |

Every asymmetric pairing beats the symmetric one. See
[D-019](docs/decisions.md#d-019-index-wide-query-narrow).

### Evidence, not verdicts

Every cheap abstention signal was measured and none was sufficient: at 95% of
answerable queries kept, refusal tops out around 10% for BM25 score or IDF
coverage and 21% for a logistic regression over thirteen features.

The one useful signal is a different kind of thing. It is not a score, it is a
fact:

- **`corpus_missing`** — a content word from the question appears nowhere in the
  stored corpus. Absent from every session, not just this one.
- **`missing_terms`** (per session) — why *this* session ranked where it did.

The distinction matters. `用户` and `记忆` are absent from most individual
sessions and present in the corpus; that means "this isn't the one", not "I
don't know". Conflating them produced 27 false refusals out of 40 answerable
questions. See [D-018](docs/decisions.md#d-018-corpus-absence-is-not-session-absence).

## Benchmarks

All numbers are reproducible from the `bench/` scripts, which import the
shipping code rather than a copy of it.

### LongMemEval (English) — retrieval

470 answerable questions over the `s` variant (average haystack 493 turns), BM25
over character bigrams:

```
hit@1    0.8660
hit@5    0.9511
hit@10   0.9660
MRR      0.8990
```

Reproduce with `python -m bench.run_longmemeval --dataset s`. `bench/data/` is
fetched by `bench/fetch_data.py` rather than committed — 264 MB of JSON does not
belong in git, and the fetcher reproduces it byte-for-byte.

**On this benchmark, retrieval is not the hard part.** 96.6% of answers are in
the top ten.

It holds for this benchmark and does not transfer:

| benchmark | metric | this store | best third-party |
|---|---|---|---|
| LongMemEval `s` | hit@10 | 0.9660 | — |
| PrecisionMemBench | active | **34/43** | 18/43 (open-knowledge-format) |
| PrecisionMemBench | total | **64/77** | 36/77 (open-knowledge-format) |
| LoCoMo | hit@1 | **0.6270** | not comparable (see below) |

The retriever is the same in all three. What differs is what the question asks
for. LongMemEval asks broad questions over a large haystack and accepts a hit in
the top ten. PrecisionMemBench asks whether exactly the right belief comes back
and nothing else — one extra document fails the case. On the first, retrieval is
solved; on the second, the best published system gets less than half right.

"Retrieval is easy" and "retrieval is hard" are both true. Which one you get
depends entirely on what counts as a hit.

### PrecisionMemBench — retrieval precision

77 single-turn cases over 35 beliefs. No generation model and no judge: the
answer *is* the set of belief IDs the store returns, and each case asserts which
IDs may and may not be in it.

```
active  34/43 = 0.7907     cases that need query-dependent retrieval
 total  64/77 = 0.8312     including cases any store passes
```

Two numbers, because the benchmark reports two and they are not interchangeable.
**`active` is the honest one**: 33 of the 77 cases pass with an empty store,
because their assertions only touch tiers the harness derives from the corpus
itself (persona prelude, pinned facts, open questions). Quoting `total` alone
would credit this store for work it did not do.

Official table, from the benchmark's own site (`tenureai.dev/benchmark`):

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
a-mem                     0/43      7/77
```

This store beats every third-party system on both columns. **It does not beat
the benchmark's own reference implementation**, which scores 43/43 and 77/77,
and there would be no reason to expect it to.

The two columns have different denominators — 34/43 active is not 34/77 — so
both are given. The claim is **best third-party result on this benchmark**, not
"beat the state of the art".

### LoCoMo — retrieval

1540 questions over 272 sessions, 5882 turns, ten independent conversation
samples.

```
hit@1   0.6270
hit@5   0.8850
hit@10  0.9471
MRR     0.7369
```

`python -m bench.locomo.run` — no model, data committed.

**Not comparable to published LoCoMo numbers.** Those (0.67–0.92) measure answer
accuracy under an LLM judge with its own backbone and prompt; these measure
whether the right session was retrieved.

`hit@5 = 0.8850` is **not** a ceiling on anyone's answer accuracy — only on a
reader that sees exactly five sessions and is forbidden to reason beyond them.
This store's own `hit@10 = 0.9471` already exceeds MemMachine's 0.9169, so the
bound says nothing about a different system's retrieval.
Published LoCoMo numbers are not evidence about retrieval quality in either
direction.

The design lesson here: retrieval unit must match the answer's unit. Indexing
turns instead of sessions costs 8.2 points of hit@1 (0.5447 → 0.6270) with no
other change.

### Chinese benchmark

59 questions (40 answerable, 19 unanswerable), 44 sessions of shared corpus,
built from scratch — none of the eight surveyed papers evaluate CJK
segmentation.

| metric | bigram/bigram | bigrams+words / words |
|---|---|---|
| hit@1 | 0.825 | **0.925** |
| hit@5 | 0.950 | **0.975** |
| hit@10 | 1.000 | 1.000 |
| MRR | 0.891 | **0.940** |

hit@10 is saturated; only hit@1 and MRR discriminate.

### Where that number stops holding

**0.925 describes 44 sessions. It does not survive dense near-duplicates.**
Adding 300 noise sessions that share the topic but state a different fact drops
hit@1 to zero:

| corpus | hit@1 | hit@5 |
|---|---|---|
| 44 sessions | 0.925 | 0.975 |
| +300 same-topic sessions | 0.000 | 0.000 |

The mechanism is literal matching. Asked *"which file and line is the root cause
in"*, a session saying `keywords.py line 443` contains 根因 but neither 文件 nor
行, while one saying `looked at some file, some line` contains all three. The
vague non-answer outranks the precise answer.

Supplying the missing literals does not fix it, and that was measured too. An
oracle query — the question plus the gold session's own file names, line numbers
and commit hashes — recovers 15% of the set:

| query | hit@1 |
|---|---|
| question only | 0.000 |
| + 3 oracle entities | 0.150 |

Two reasons it stays low. **23 of the 40 answers contain no concrete token at
all** ("did the downstream components need changes?" — answered "no"), so there
is nothing to supply. And of the 17 that do, BM25 still ranks the right session
first less than half the time.

Four query-side fixes were tried and none helped; the list is in
[D-020](docs/decisions.md#d-020-bag-of-words-retrieval-degrades-on-dense-same-topic-corpora)
and [D-021](docs/decisions.md#d-021-oracle-entities-recover-15-the-failure-is-not-missing-literals).

Practical consequence: BM25 is adequate when a topic appears once or twice in the
corpus. When it does not, the fix is at write time, not at query time.

PrecisionMemBench is the counter-example: 35 beliefs, near-identical subjects, the
same word `redis` in two different memories. The same BM25 scores 0.8312 there
once memories carry scope tags, supersession links and relation refs, and once the
index holds the belief's names rather than its prose.

The 0.000 above is real. It is what *untagged* BM25 does to a corpus where every
document is about the same subject. Read the top five, weight `corpus_missing`
over raw score, and label your memories — scope, kind, what supersedes what. That
is why the system returns evidence rather than a verdict, from three independent
measurements.

### Abstention

At the natural operating point of `corpus_missing`:

| strategy | keep | refuse |
|---|---|---|
| answer everything | 1.000 | 0.000 |
| **corpus_missing** | **0.875** | **0.632** |
| local 9B model | 0.950 | 0.895 |
| hosted `deepseek-flash` | 0.925 | 1.000 |

A 0.875 keep rate means 12.5% of answerable questions would be refused
automatically. That is not good enough to decide for the caller, which is why
the signal is reported rather than acted on.

### Hosted judge

If built-in abstention is required, a hosted model is the only option that costs
no memory:

| approach | keep | refuse | latency | resident |
|---|---|---|---|---|
| BM25 only | 1.000 | 0.000 | 0.22 ms | 0 |
| local 2B | 0.750 | 1.000 | 0.77 s | ~1 GB |
| local 9B | 0.950 | 0.895 | 1.5–2 s | 15.2 GB |
| `deepseek-flash` | 0.925 | 1.000 | 0.81 s | 0 |

0.81 s × 30 memory lookups in a task is 24 seconds of added latency, spent on a
judgement the calling agent could make while it is already reasoning.

### How much of this is sampling noise

77 cases is a small benchmark, and small benchmarks produce wide intervals. Every
number on this page therefore gets a Wilson 95% interval:

| result | fraction | 95% CI | width |
|---|---|---|---|
| PrecisionMemBench **active**, this store | 34/43 = 0.7907 | 0.6479 – 0.8858 | 23.8 pp |
| PrecisionMemBench **active**, open-knowledge-format | 18/43 = 0.4186 | 0.2838 – 0.5667 | 28.3 pp |
| PrecisionMemBench **total**, this store | 64/77 = 0.8312 | 0.7323 – 0.8986 | 16.6 pp |
| PrecisionMemBench **total**, open-knowledge-format | 36/77 = 0.4675 | 0.3603 – 0.5779 | 21.8 pp |
| Chinese hit@1 (40 answerable) | 37/40 = 0.9250 | 0.8014 – 0.9742 | 17.3 pp |
| LoCoMo hit@1 | 960/1531 = 0.6270 | 0.6025 – 0.6509 | 4.8 pp |
| LoCoMo hit@5 | 1355/1531 = 0.8850 | 0.8681 – 0.9001 | 3.2 pp |

The intervals are wide, and that is the honest cost of 77 cases. The comparison
still holds on **both** denominators, which matters because they are not
interchangeable:

- active: 0.6479 lower bound against 0.5667 upper bound, an 8-point gap
- total: 0.7323 lower bound against 0.5779 upper bound, a 15-point gap

Neither pair overlaps. The Chinese number is the one to treat carefully: 37 of 40
gives [0.80, 0.97]; the point estimate 0.925 should not be quoted as though it
were precise. It is a direction, not a value.

Two things this table cannot fix. **PrecisionMemBench is one benchmark**; a second
corpus with the same shape would either confirm the effect or show it was specific
to these 35 beliefs. And **no independent party has reproduced 0.8312** — the
data and harness are committed, so it is reproducible, but reproducible-by-me is
not the same as reproduced-by-someone-else.

## Running alongside another memory server

**Do not enable two memory MCP servers in the same agent.** Measured (D-031):
with `neural-memory` and `b4a` both enabled, an agent asked about code-style
conventions queried the other server, got nothing, and reported "memory has no
record of this" — while five relevant memories sat in `b4a`. The store cannot
detect this from inside; it looks exactly like an empty memory.

A direct MCP handshake proved `b4a` was healthy, so the failure was at selection,
not retrieval. One memory server per agent, or names that make the choice
unambiguous. Harnesses that namespace MCP tools by server — mcp__b4a__b4a_recall
— cannot hit this; the collision needs two servers exposing similarly named
memory tools.

## Against the frontier

Where this sits relative to the current agent-memory landscape, stated plainly
because a README that only lists strengths is a sales page.

| Capability | Status here |
|---|---|
| Structured memory (typed facts, preferences, entities) | **Partial.** `kind` is a free-form label, `tags` are scopes, `refs` are links. No schema, no validation — the caller decides what a kind means. |
| Temporal / graph memory (validity windows, relation evolution) | **Not built.** `created_at` exists and `superseded_by` records replacement, but there is no validity interval and no graph traversal at query time. A graph was built and measured; it changed nothing on the Chinese corpus (D-024), which is why it is not wired in. `refs` gives one-hop expansion, not traversal. |
| Conflict resolution | **Built, deterministic.** `supersede(old, new)`; the replaced memory stops surfacing. No model involved. PrecisionMemBench supersession category: 3/3. |
| Write-path deduplication / pattern extraction | **Not built.** Extraction exists (`remember()`) and is **off by default**, because measured at write time it costs retrieval (D-023). |
| Multi-retriever fusion | **Not built.** One retriever: BM25 over bigrams and optionally jieba words. |
| Forgetting / consolidation | **Crude.** `prune(keep_sessions=N)` is recency-based size control, not retrieval improvement. Recovering a noisy corpus needs deleting 95% of it — emptying, not forgetting (D-022). |
| Cross-session aggregation | **Not built.** Explicitly out of scope so far. |
| Proactive recall | **Partial.** `recall-context` reads a prompt from stdin and prints formatted memory, silent when nothing matches — built for a client-side hook, since MCP is pull-only and a server cannot volunteer a memory. No scheduler: the client decides when to run it. See `integrations/hooks/`. |
| Multimodal | **Not built.** |
| Hallucination defence | **Partial.** Evidence/verdict separation, `corpus_missing`, three-state judge results. Nothing that detects a wrong answer produced from a correct retrieval. |
| Reproducibility of published numbers | **Strong.** Three benchmarks, two with no model in the loop at all. This is the axis where the project is ahead of most of the field, and it is a consequence of the model-free rule rather than a goal in itself. |

The honest summary: **this is a retrieval layer with unusually careful measurement,
not a cognitive architecture.** It does one thing — find the right memory and say
clearly what it found and what it did not — and declines the rest on purpose.
Whether that is the right trade depends on the caller. An agent that already has
a model wants evidence and a clean abstention signal, not a second model's
opinion about its own memory.

## Status

Library, CLI and MCP server are complete. The metadata layer — scope tags,
supersession links, relation refs, and a separate index text — landed after the
first benchmarks were written.

**Deliberately not built, and not planned:**

- **Automatic cross-session merging.** Retrieval already spans sessions — one
  query searches every stored session and returns the best of them. What is not
  built is *synthesising* several sessions into one summary. That needs a model,
  or a rule for deciding that two memories describe the same thing, and merging
  badly loses detail the reader could have weighed themselves.
- **Multimodal.** Out of scope.

**Recently added:**

- **Near-duplicate detection.** A SimHash fingerprint per memory; `duplicates_of(id)`
  returns the memories that may state the same thing. It reports and does not
  merge — the calling agent reads both and decides (D-028). It is not on the
  search path: a corpus-wide scan cost 5 ms against 2893 memories, more than half
  of a query.
- **Weight.** `reinforce` raises it, `decay` lowers it, `prune_by_weight` deletes
  what fell below a threshold. This is the first forgetting signal here that is
  not about age: a retrieved memory survives, an unretrieved one decays (D-029).

**Recently fixed:**

- **A replaced memory used to vanish.** `supersede(old, new)` hid the old one, but
  when the new one used different words — "main store is MySQL" replaced by
  "migrated to PostgreSQL" — a query for "main store" matched only the hidden
  one, and the search returned nothing at all. A memory system that forgets a
  fact the moment it is updated is worse than one that never had it. Hits are now
  redirected to the current memory instead of dropped (D-027).

**Not built, on the list:**

- **Time-decayed forgetting.** `prune` is recency-based size control; D-022
  measured that recovering a noisy corpus needs deleting 95% of it. A decay that
  weighs importance rather than age is the next real piece of work.
- **A recall scheduler.** `recall-context` exists for hooks, but nothing decides
  when to run it. Every-turn is cheap (1.5 ms) and probably too noisy.
- **Frequency-based ranking.** `accesses` is counted and `weight` can blend into
  ranking via `weight_blend`, but the default is off. The mechanism is tested;
  whether it *helps* is unmeasured, and an unmeasured signal reordering results is
  the kind of change this project keeps refusing to make.
- **Multi-retriever fusion.** One BM25 today. A second lexical retriever could be
  tried without breaking the model-free rule.
- **Schema on `kind`.** It is a free-form string with no validation.

**Measured and rejected — not deferred, decided:**

- **Graph traversal at query time.** Built, measured on the Chinese corpus,
  changed nothing (D-024). One-hop `refs` expansion does help — 4/4 on
  PrecisionMemBench — but that is a link, not a traversal.
- **Write-time extraction.** Off by default because it costs retrieval (D-023).

**Out of reach without a model:**

- **Detecting a wrong answer produced from a correct retrieval.** Every signal
  here says whether the right memory came back; none says whether the reader used
  it well. That needs a judge, and a judge that is wrong is worse than none.

The [decision log](docs/decisions.md) records 34 entries in both directions —
including the hypotheses that were tested and failed. The
[research notes](docs/research.md) and [landscape analysis](docs/landscape.md)
cover the surrounding field.

Known limits, all measured. Untagged retrieval degrades on dense near-duplicates
(D-020) and no query-side trick recovers it (D-021); the fix is at write time
(D-025, D-026). LoCoMo hit@1 is 0.6270, which does not beat the best published
answer accuracy on that dataset — and the gap there is retrieval, not reading.

## Design decisions

Thirty-four decisions, each with the measurement or reasoning behind it, in
[`docs/decisions.md`](docs/decisions.md). Both directions are recorded —
including the hypotheses that were tested and failed.

## Contributing

The verification gate:

```bash
python -m pytest tests/ -q
python -m ruff check src bench tests
python -m ruff format --check src bench tests
python -m bench.zh.e2e
python -m bench.pmb.run
python -m bench.locomo.run
```

`bench/zh/e2e.py` runs the shipping code path and should report `hit@1 0.9250`.
`bench.pmb.run` should report `pass 64/77`; `bench.locomo.run` should report
`hit@1 0.6270`. A change to retrieval, tokenisation or scoring that lowers any of
them is a regression. The last two need no API key and no network.

Changes to a design decision belong in `docs/decisions.md` as a new entry, with
the measurement that justifies it.

## Special thanks

To [nhadaututtheky](https://github.com/nhadaututtheky) for
[neural-memory](https://github.com/nhadaututtheky/neural-memory). This project
began as a bug fix there ([PR #228](https://github.com/nhadaututtheky/neural-memory/pull/228)) —
a Chinese query returned nothing because the keyword regex had no Han range.
Fixing it showed that the assumption, not the character class, was the problem, so
this is a different answer to the same question rather than a replacement. Its
brain is also the corpus for `bench/nmem/`, the only dataset here that records
what a user actually asked for.

## License

Apache-2.0. See [LICENSE](LICENSE).
