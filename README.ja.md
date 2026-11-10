# Brain4Agent

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

エージェントのための長期記憶。CJK 優先の検索、コア経路にモデルなし、
信頼スコアの代わりに証拠を返す。

`pip install "Brain4Agent[precise]"` には Python 3.11+ と、20 MB の任意の
辞書が要る。辞書がなくてもすべて動く。

---

## 目次

- [これは何か](#これは何か)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [仕組み](#仕組み)
- [ベンチマーク](#ベンチマーク)
- [別のメモリサーバーとの併用](#別のメモリサーバーとの併用)
- [最前線との比較](#最前線との比較)
- [現状](#現状)
- [コントリビュート](#コントリビュート)
- [謝辞](#謝辞)
- [ライセンス](#ライセンス)

## これは何か

コードベースで作業するエージェントは文脈を溜め込む。なぜその決定をしたのか、
どのやり方が失敗したのか、デプロイはどこにあるのか。Brain4Agent はそうした
やり取りを単一の SQLite ファイルに保存し、セッション単位で検索する。

- **CJK はそのまま動く。** 文字バイグラムは辞書も分割モデルも要らない。
  測定値は中国語のもの。日本語・韓国語・ベトナム語は対応しテストもしている。
- **コアはモデルを呼ばない。** 検索は BM25。API キーもアクセラレータも
  コールドスタートもなく、クエリあたり一桁ミリ秒。
- **返すのは判定ではなく証拠だ。** 検索は、質問中のどの語がコーパスに一度も
  現れなかったか（`corpus_missing`）、あるセッションがどの語を欠いていたか
  （`missing_terms`）を返す。その意味をどう取るかは呼び出し側が決める。
- **保存するものと索引するものは別だ。** 記憶は全文を保ち、索引はより狭く
  識別性の高いものを持てる。

記憶はさらに `tags`（スコープ）、`superseded_by`（置換）、`refs`
（一ホップのリンク）を持ち、これらは書き手が与える。

## インストール

```bash
pip install "Brain4Agent[precise]"   # 推奨：jieba を追加、hit@1 0.925
pip install Brain4Agent              # 依存ゼロ：バイグラムのみ、hit@1 0.825
```

Windows と macOS で動く。Python 3.11 以降が必要。パッケージは PyPI にある:
<https://pypi.org/project/Brain4Agent/>。コード自体を触るなら `git clone` して
`pip install -e ".[precise,dev]"`。

`jieba` は 20 MB の静的辞書。なければコードはバイグラムに退避する。

**Windows の非 ASCII 入力。** MCP サーバーと CLI は stdin/stdout を UTF-8 に
固定しているので、CJK の内容はコンソールのコードページに関係なく動く。
ただしコマンドライン*引数*はその前に Python がデコードする。非 ASCII テキストを
引数で渡すなら `PYTHONUTF8=1` を設定すること。

## クイックスタート

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

ストアが答えられない質問:

```bash
$ brain4agent search "杭州测试机上装的是什么数据库" -k 1
1 sessions for '杭州测试机上装的是什么数据库'
  corpus has no: 数据库, 机上
  [ 4.81 cov=0.28] c060a5294037  session lacks: 数据库, 机上
```

既存の履歴を JSONL から一括読み込み（1 行 1 オブジェクト、`content` 必須）:

```bash
$ brain4agent import conversations.jsonl
imported 4 turns across 2 sessions
```

```jsonl
{"session_id": "s1", "role": "user", "content": "中文查询搜不到记忆"}
{"session_id": "s1", "role": "assistant", "content": "根因在 keywords.py 第 443 行"}
```

### MCP サーバー

```json
{"mcpServers": {"brain4agent": {
    "command": "python",
    "args": ["-m", "brain4agent.mcp_server"],
    "env": {"BRAIN4AGENT_DB": "/path/to/brain.db"}
}}}
```

ツールは五つ:

| ツール | 用途 |
|---|---|
| `b4a_remember` | 1 ターン保存し、セッション ID を返す |
| `b4a_recall` | 検索。`corpus_missing` とセッションごとの `missing_terms` を返す |
| `b4a_ask` | 検索と回答を一度に。モデルが必要で、未設定ならそう言う |
| `b4a_session` | セッション全体を順に読む |
| `b4a_stats` | 件数 |

同じライブラリはエージェント [skill](integrations/) としても配布している。

### ライブラリ

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

`add_exchange(user=..., assistant=...)` は一問一答を 1 セッションとして保存する。
検索単位はセッションで、ターンではない。

一括インポートは `batch()` を通し、グループごとに 1 回だけコミットする:

```python
with store.batch():
    for turn in turns:
        store.add(turn.text, session_id=turn.sid, role=turn.role)
```

### 任意のホスト型ジャッジ

既定では無効。キーがあるときだけ有効になる。

```python
from brain4agent import Judge, MemoryStore

judge = Judge.from_env()          # BRAIN4AGENT_JUDGE_KEY を読む。未設定なら None
if judge:
    r = store.search_judged("question", judge)
    print(r.verdict)              # found / not_found / uncertain
```

`uncertain` は `not_found` とは別の第三の状態。ジャッジのタイムアウトは、
記憶が存在しない証拠ではない。

## 仕組み

```
                書き込み                       読み取り
                  │                              │
          バイグラム + 辞書語              辞書語
                  │                              │
                  ▼                              ▼
          ┌──────────────────────────────────────────┐
          │  SQLite: id, session, role, content,     │
          │          created_at, tokens              │
          └──────────────────────────────────────────┘
                  │                              │
                  ▼                              ▼
           BM25 索引（メモリ上）  →  セッション単位の証拠
                                    + corpus_missing
                                    + セッションごとの coverage
```

### ストレージ

SQLite ファイル一つ。行はターン、検索の単位はセッション。索引トークンは
書き込み時に計算して列にキャッシュする。

```
                       インポート  再構築    クエリ
127 ターン   44 セッション          3.9 ms   1.8 ms
1016 ターン  352 セッション  0.25 s  29 ms   3.7 ms
```

書き込みはバッチでコミットする。1016 ターンのインポートは 1 件ずつなら 3.8 秒、
`batch()` 内なら 0.25 秒。

### 読み書きで分割したトークナイザ

索引とクエリは別のトークナイザを使う。3×3 グリッドの実測:

| 索引 | クエリ | hit@1 | MRR |
|---|---|---|---|
| bigram | bigram | 0.825 | 0.891 |
| jieba | jieba | 0.875 | 0.911 |
| **バイグラム + 語** | **語** | **0.925** | **0.940** |

[D-019](docs/decisions.md#d-019-index-wide-query-narrow) を参照。

### 判定ではなく証拠

- **`corpus_missing`**——質問の内容語が、保存済みコーパスのどこにも現れない。
- **`missing_terms`**（セッションごと）——そのセッションが欠いていた質問語。

棄権するかどうかは呼び出し側が決める。ストアは事実を報告するだけだ。

## ベンチマーク

すべての数値は `bench/` のスクリプトで再現できる。スクリプトはコピーではなく
出荷コードをインポートする。

### LongMemEval（英語）— 検索

`s` 変体の回答可能な質問 470 件（平均 haystack 493 ターン）:

```
hit@1    0.8660
hit@5    0.9511
hit@10   0.9660
MRR      0.8990
```

`python -m bench.run_longmemeval --dataset s`。`bench/data/` は
`bench/fetch_data.py` が取得する（264 MB、git には入れない）。

三つのベンチマークを並べる:

| ベンチマーク | 指標 | このストア | 最良のサードパーティ |
|---|---|---|---|
| LongMemEval `s` | hit@10 | 0.9660 | — |
| PrecisionMemBench | active | **34/43** | 18/43 (open-knowledge-format) |
| PrecisionMemBench | total | **64/77** | 36/77 (open-knowledge-format) |
| LoCoMo | hit@1 | **0.6270** | 比較不可 |

### PrecisionMemBench — 検索精度

35 の belief に対する 77 の単一ターンケース。生成モデルもジャッジもない。
答えはストアが返す belief ID の集合そのものだ。`active` が正直な列で、
77 ケースのうち 33 は空のストアでも通る。

```
active  34/43 = 0.7907     クエリ依存の検索が必要なケース
 total  64/77 = 0.8312     どのストアでも通るケースを含む
```

ベンチマーク自身のサイト（`tenureai.dev/benchmark`）による公式表:

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

両列でサードパーティ最良。ベンチマーク自身の参照実装は 43/43 と 77/77。

### LoCoMo — 検索

272 セッション、5882 ターンに対する 1540 の質問。

```
hit@1   0.6270
hit@5   0.8850
hit@10  0.9471
MRR     0.7369
```

`python -m bench.locomo.run`——モデルなし、データはコミット済み。これは
セッションの検索命中率であり、LLM ジャッジ下の回答精度ではない。公開されている
LoCoMo の数値とは比較できない。

セッションではなくターンを索引すると、hit@1 が 8.2 ポイント落ちる
（0.5447 → 0.6270）。

### 中国語ベンチマーク

59 問（回答可能 40、回答不能 19）、44 セッション。

| 指標 | bigram/bigram | バイグラム+語 / 語 |
|---|---|---|
| hit@1 | 0.825 | **0.925** |
| hit@5 | 0.950 | **0.975** |
| hit@10 | 1.000 | 1.000 |
| MRR | 0.891 | **0.940** |

hit@10 は飽和している。判別力があるのは hit@1 と MRR だけだ。

### その数字が成り立たなくなる範囲

**0.925 が記述しているのは 44 セッションだ。密な近似重複で崩れる。**

| コーパス | hit@1 | hit@5 |
|---|---|---|
| 44 セッション | 0.925 | 0.975 |
| +同一話題の 300 セッション | 0.000 | 0.000 |

「根本原因はどのファイルの何行目か」と問うと、「some file, some line」と言う
曖昧なセッションが `keywords.py line 443` と言うセッションより上位に来る。
正解セッション自身のファイル名、行番号、コミットハッシュを足しても 15% しか
回復しない:

| クエリ | hit@1 |
|---|---|
| 質問のみ | 0.000 |
| + オラクル実体 3 つ | 0.150 |

上位 5 件を読み、生スコアより `corpus_missing` を重く見て、記憶にラベルを
付ける——スコープ、kind、何が何を置き換えるか。修正は書き込み側にある。
クエリ側で試した四つは
[D-020](docs/decisions.md#d-020-bag-of-words-retrieval-degrades-on-dense-same-topic-corpora)
と [D-021](docs/decisions.md#d-021-oracle-entities-recover-15-the-failure-is-not-missing-literals) にある。

### 棄権

`corpus_missing` の自然な動作点で:

| 手法 | 維持 | 拒否 | レイテンシ | 常駐 |
|---|---|---|---|---|
| BM25 のみ | 1.000 | 0.000 | 0.22 ms | 0 |
| **corpus_missing** | **0.875** | **0.632** | ~0 | 0 |
| ローカル 2B | 0.750 | 1.000 | 0.77 s | ~1 GB |
| ローカル 9B | 0.950 | 0.895 | 1.5–2 s | 15.2 GB |
| ホスト型 `deepseek-flash` | 0.925 | 1.000 | 0.81 s | 0 |

維持率 0.875 は、答えられる質問の 12.5% が自動的に拒否されることを意味する。
だからこのシグナルは報告するだけで、自動では作用させない。ホスト型ジャッジは
1 回あたり 0.81 秒を足す——1 タスクで 30 回なら 24 秒。

## 別のメモリサーバーとの併用

**同じエージェントに二つのメモリ MCP サーバーを有効にしてはいけない。**
`neural-memory` と `b4a` の両方を有効にすると、エージェントはもう一方の
サーバーに問い合わせ、「メモリに記録はない」と報告した。`b4a` には関連する
記憶が五つあったのに。エージェントごとにメモリサーバーは一つ、あるいは選択が
曖昧にならない名前を付けること。

## 最前線との比較

| 能力 | ここでの状態 |
|---|---|
| 構造化メモリ（型付き事実、選好、実体） | **部分的。** `kind` は自由形式のラベル、`tags` はスコープ、`refs` はリンク。スキーマも検証もない。 |
| 時間的・グラフ的メモリ（有効期間、関係の変化） | **未実装。** `created_at` と `superseded_by` のみ。有効区間もクエリ時のグラフ走査もない。一ホップの `refs` は PrecisionMemBench で 4/4。 |
| 衝突解決 | **実装済み。** `supersede(old, new)`。置き換えられた記憶は浮かび上がらない。PrecisionMemBench の supersession カテゴリ: 3/3。 |
| 書き込み経路の重複排除・パターン抽出 | **未実装。** 抽出（`remember()`）は存在するが**既定で無効**（D-023）。 |
| 複数検索器の統合 | **未実装。** 検索器は一つ。BM25。 |
| 忘却・統合 | **粗い。** `prune(keep_sessions=N)` は新近性によるサイズ制御。`decay` と `prune_by_weight` が重みのシグナルを足す。ノイズの多いコーパスの回復には 95% の削除が要る（D-022）。 |
| セッション横断の集約 | **未実装。** |
| 能動的リコール | **部分的。** クライアント側フック用の `recall-context`。スケジューラはない。 |
| マルチモーダル | **未実装。** |
| ハルシネーション防御 | **部分的。** 証拠と判定の分離、`corpus_missing`、三状態のジャッジ。正しい検索から生まれた誤った回答を検出するものはない。 |
| 公開数値の再現性 | **強い。** ベンチマークは三つ、うち二つはモデルをループに入れていない。 |

## 現状

ライブラリ、CLI、MCP サーバーは完成している。

**追加済み:** 近似重複の検出（`duplicates_of`、SimHash。報告のみでマージせず、
検索経路には載らない）と weight（`reinforce`、`decay`、`prune_by_weight`）。

**未実装:** セッション横断の合成、マルチモーダル、`kind` のスキーマ、
複数検索器の統合、リコールのスケジューラ、時間減衰による忘却。

**測定して却下:** クエリ時のグラフ走査（D-024）、書き込み時の抽出（D-023）。

**最近の修正:** 新しい記憶が別の語を使うとき、`supersede(old, new)` は
何も返さずに終わるのではなく、ヒットを新しい記憶へリダイレクトする（D-027）。

[決定ログ](docs/decisions.md) は 36 項目を記録している。
[リサーチノート](docs/research.md) と
[ランドスケープ分析](docs/landscape.md) が周辺領域を扱う。

## コントリビュート

```bash
python -m pytest tests/ -q
python -m ruff check src bench tests
python -m ruff format --check src bench tests
python -m bench.zh.e2e
python -m bench.pmb.run
python -m bench.locomo.run
```

`bench/zh/e2e.py` は `hit@1 0.9250`、`bench.pmb.run` は `pass 64/77`、
`bench.locomo.run` は `hit@1 0.6270` を報告するはずだ。下回れば回帰。
最後の二つは API キーもネットワークも要らない。設計上の決定を変えるときは、
根拠となる測定とともに `docs/decisions.md` へ新しい項目を追加すること。

## 謝辞

[nhadaututtheky](https://github.com/nhadaututtheky) の
[neural-memory](https://github.com/nhadaututtheky/neural-memory) に感謝する。
本プロジェクトはそこのバグ修正
（[PR #228](https://github.com/nhadaututtheky/neural-memory/pull/228)）から
始まった: 中国語のクエリが、キーワード正規表現に漢字の範囲がないために
何も返さなかった。その brain は `bench/nmem/` のコーパスでもある。

## ライセンス

Apache-2.0。[LICENSE](LICENSE) を参照。
