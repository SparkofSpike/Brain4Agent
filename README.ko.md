# Brain4Agent

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

에이전트를 위한 장기 기억. CJK 우선 검색, 코어 경로에 모델 없음,
신뢰도 점수 대신 증거.

`pip install "Brain4Agent[precise]"`에는 Python 3.11+와 20 MB의 선택적
사전이 필요하다. 사전이 없어도 전부 동작한다.

---

## 목차

- [무엇인가](#무엇인가)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [동작 방식](#동작-방식)
- [벤치마크](#벤치마크)
- [다른 메모리 서버와 함께 실행하기](#다른-메모리-서버와-함께-실행하기)
- [최전선과 비교](#최전선과-비교)
- [현황](#현황)
- [기여](#기여)
- [특별 감사](#특별-감사)
- [라이선스](#라이선스)

## 무엇인가

코드베이스에서 일하는 에이전트는 맥락을 쌓는다. 어떤 결정을 왜 내렸는지,
어떤 접근이 실패했는지, 배포 환경이 어디에 있는지. Brain4Agent는 그런 턴들을
SQLite 파일 하나에 저장하고 세션 단위로 검색한다.

- **CJK는 별도 설정 없이 동작한다.** 문자 바이그램은 사전도 분절 모델도
  필요로 하지 않는다. 측정값은 중국어 기준이며, 일본어·한국어·베트남어는
  지원하고 테스트한다.
- **코어는 모델을 호출하지 않는다.** 검색은 BM25. API 키도, 가속기도,
  콜드 스타트도 없고 질의당 한 자릿수 밀리초.
- **판정이 아니라 증거를 돌려준다.** 검색은 질문의 어떤 단어가 코퍼스에 한 번도
  없었는지(`corpus_missing`), 어떤 세션이 어떤 단어를 빠뜨렸는지
  (`missing_terms`)를 반환한다. 그 의미는 호출자가 판단한다.
- **저장되는 것과 인덱싱되는 것은 다르다.** 기억은 전체 텍스트를 보관하고,
  인덱스는 더 좁고 더 특징적인 것을 담을 수 있다.

기억에는 `tags`(범위), `superseded_by`(대체), `refs`(한 홉 링크)도 붙으며,
이들은 기록하는 쪽이 제공한다.

## 설치

```bash
pip install "Brain4Agent[precise]"   # 권장: jieba 추가, hit@1 0.950
pip install Brain4Agent              # 의존성 제로: 바이그램만, hit@1 0.875
```

Windows와 macOS 모두 동작한다. Python 3.11 이상이 필요하다. 패키지는 PyPI에
있다: <https://pypi.org/project/Brain4Agent/>. 코드 자체를 다루려면 `git clone`
후 `pip install -e ".[precise,dev]"`.

`jieba`는 20 MB 정적 사전이다. 없으면 코드는 바이그램으로 퇴화한다.

**Windows의 비 ASCII 입력.** MCP 서버와 CLI는 stdin/stdout을 UTF-8로
고정하므로 CJK 내용은 콘솔 코드 페이지와 무관하게 동작한다. 하지만 명령줄
*인수*는 그 전에 Python이 디코딩한다. 비 ASCII 텍스트를 인수로 넘긴다면
`PYTHONUTF8=1`을 설정하라.

## 빠른 시작

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

저장소가 답할 수 없는 질문:

```bash
$ brain4agent search "杭州测试机上装的是什么数据库" -k 1
1 sessions for '杭州测试机上装的是什么数据库'
  corpus has no: 数据库, 机上
  [ 4.81 cov=0.28] c060a5294037  session lacks: 数据库, 机上
```

기존 기록을 JSONL에서 일괄 적재(한 줄에 객체 하나, `content` 필수):

```bash
$ brain4agent import conversations.jsonl
imported 4 turns across 2 sessions
```

```jsonl
{"session_id": "s1", "role": "user", "content": "中文查询搜不到记忆"}
{"session_id": "s1", "role": "assistant", "content": "根因在 keywords.py 第 443 行"}
```

### MCP 서버

```json
{"mcpServers": {"brain4agent": {
    "command": "python",
    "args": ["-m", "brain4agent.mcp_server"],
    "env": {"BRAIN4AGENT_DB": "/path/to/brain.db"}
}}}
```

도구는 다섯 개:

| 도구 | 용도 |
|---|---|
| `b4a_remember` | 턴 하나 저장, 세션 id 반환 |
| `b4a_recall` | 검색. `corpus_missing`과 세션별 `missing_terms` 반환 |
| `b4a_ask` | 검색과 답변을 한 번에. 모델이 필요하며 없으면 그렇다고 말한다 |
| `b4a_session` | 세션 전체를 순서대로 읽기 |
| `b4a_stats` | 개수 |

같은 라이브러리는 에이전트 [skill](integrations/)로도 배포된다.

### 라이브러리

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

`add_exchange(user=..., assistant=...)`는 질문과 답변을 하나의 세션으로
저장한다. 검색 단위는 세션이며 턴이 아니다.

일괄 가져오기는 `batch()`를 거치며, 그룹마다 한 번만 커밋한다:

```python
with store.batch():
    for turn in turns:
        store.add(turn.text, session_id=turn.sid, role=turn.role)
```

### 선택적 호스팅 심사자

기본값은 꺼짐. 키가 있을 때만 활성화된다.

```python
from brain4agent import Judge, MemoryStore

judge = Judge.from_env()          # BRAIN4AGENT_JUDGE_KEY를 읽는다. 없으면 None
if judge:
    r = store.search_judged("question", judge)
    print(r.verdict)              # found / not_found / uncertain
```

`uncertain`은 `not_found`와 구별되는 세 번째 상태다. 심사자의 타임아웃은
기억이 존재하지 않는다는 증거가 아니다.

## 동작 방식

```
                 쓰기                            읽기
                  │                              │
           바이그램 + 사전 단어              사전 단어
                  │                              │
                  ▼                              ▼
          ┌──────────────────────────────────────────┐
          │  SQLite: id, session, role, content,     │
          │          created_at, tokens              │
          └──────────────────────────────────────────┘
                  │                              │
                  ▼                              ▼
           BM25 인덱스(메모리)  →  세션 단위 증거
                                    + corpus_missing
                                    + 세션별 coverage
```

### 저장

SQLite 파일 하나. 행은 턴이고, 검색 단위는 세션이다. 인덱스 토큰은 쓰기 시점에
계산해 열에 캐시한다.

```
                       가져오기  재구축    질의
127턴    44세션                  3.9 ms   1.8 ms
1016턴   352세션       0.25 s   29 ms    3.7 ms
```

쓰기는 배치로 커밋한다. 1016턴 가져오기는 한 건씩이면 3.8초,
`batch()` 안이면 0.25초.

### 읽기/쓰기 토크나이저 분리

인덱싱과 질의는 서로 다른 토크나이저를 쓴다. 3×3 격자 실측(D-019)과
최선 셀 위에 얹은 두 가지 변경(D-037):

| 인덱스 | 질의 | hit@1 | MRR |
|---|---|---|---|
| bigram | bigram | 0.825 | 0.891 |
| jieba | jieba | 0.875 | 0.911 |
| 바이그램 + 단어 | 단어 | 0.925 | 0.940 |
| **바이그램 + 단어** | **단어 − 의문사** | **0.950** | **0.961** |

마지막 행은 3×3 격자 밖에서 두 가지를 더한다. 질의 쪽에서 의문사를 빼고,
세션의 상위 3턴을 하나의 점수로 접는다.

### 판정이 아니라 증거

- **`corpus_missing`**——질문의 내용어가 저장된 코퍼스 어디에도 없다.
- **`missing_terms`**(세션별)——그 세션이 빠뜨린 질문 단어.
- **`term_frequencies`**——각 질문 단어가 몇 세션에 나타나는지. 2898개 중 453개에
  나타나는 단어는 누락도 오타도 아니다. 구분할 수 없을 만큼 흔하며, 질의에 단어가
  하나 더 필요하다(D-039).

기권 여부는 호출자가 정한다. 저장소는 사실만 보고한다.

## 벤치마크

모든 수치는 `bench/` 스크립트로 재현할 수 있다. 스크립트는 복사본이 아니라
출하 코드를 임포트한다.

### LongMemEval(영어) — 검색

`s` 변형의 답 가능 질문 470개(평균 haystack 493턴):

```
hit@1    0.8660
hit@5    0.9511
hit@10   0.9660
MRR      0.8990
```

`python -m bench.run_longmemeval --dataset s`. `bench/data/`는
`bench/fetch_data.py`가 가져온다(264 MB, git에 넣지 않음).

세 벤치마크를 나란히:

| 벤치마크 | 지표 | 이 저장소 | 최고 서드파티 |
|---|---|---|---|
| LongMemEval `s` | hit@10 | 0.9660 | — |
| PrecisionMemBench | active | **42/43** | 18/43 (open-knowledge-format) |
| PrecisionMemBench | total | **72/77** | 36/77 (open-knowledge-format) |
| LoCoMo | hit@1 | **0.6310** | 비교 불가 |

### PrecisionMemBench — 검색 정밀도

35개 신념에 대한 77개 단일 턴 케이스. 생성 모델도 심사자도 없다. 답은 저장소가
반환하는 신념 ID 집합 자체다. `active`가 정직한 열이며, 77개 중 33개는 빈
저장소로도 통과한다.

```
active  42/43 = 0.9767     질의 의존 검색이 필요한 케이스
 total  72/77 = 0.9351     어떤 저장소든 통과하는 케이스 포함
```

결과 집합 크기를 여덟 단어 미만에서 `top_k=1`로 고정하던 규칙을 없애고, 질의에
등장한 이름의 수로 정한다. 이름은 해당 케이스의 scope에서 보이는 belief만 센다.
이름 대조는 부분 문자열에서 단어 경계로 바뀌어 `pipe`가 `pipeline` 안에서
걸리지 않는다. 코퍼스에 없는 영어 단어를 한 편집 거리로 대조하는 퍼지 경로도
추가했다(`typscript` → `typescript`, `mongod` → `mongodb`).

벤치마크 자체 사이트(`tenureai.dev/benchmark`)의 공식 표:

```
System                  Active    Total
----------------------  ------    -----
tenure (the authors)    43/43     77/77
this store              42/43     72/77
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

두 열 모두 서드파티 최고. 벤치마크 자체의 참조 구현은 43/43과 77/77.

### LoCoMo — 검색

272개 세션, 5882턴에 대한 질문 1540개.

```
hit@1   0.6310
hit@5   0.8870
hit@10  0.9484
MRR     0.7402
```

`python -m bench.locomo.run`——모델 없음, 데이터는 커밋됨. 이는 세션 검색
적중률이며 LLM 심사자 아래의 답 정확도가 아니다. 따라서 공개된 LoCoMo 수치와
비교할 수 없다.

세션 대신 턴을 인덱싱하면 hit@1이 2.5포인트 떨어진다(0.6022 → 0.6310).
토크나이저와 채점은 같다.

### 중국어 벤치마크

질문 59개(답 가능 40, 답 불가 19), 세션 44개.

| 지표 | bigram/bigram | 현재 |
|---|---|---|
| hit@1 | 0.825 | **0.950** |
| hit@5 | 0.950 | **1.000** |
| hit@10 | 1.000 | 1.000 |
| MRR | 0.891 | **0.961** |

`현재` = 인덱스 `바이그램 + 단어`, 질의 `단어 − 의문사`, 세션 상위 3턴을 접은
점수(D-037). hit@10은 포화 상태이고, hit@1과 MRR만 변별력을 가진다.

### 그 수치가 버티지 못하는 지점

**0.950은 세션 44개를 설명한다. 밀집한 근사 중복에서 무너진다.**

| 코퍼스 | hit@1 | hit@5 |
|---|---|---|
| 세션 44개 | 0.950 | 1.000 |
| 동일 주제 세션 +300개 | 0.000 | 0.000 |

"근본 원인이 어느 파일 몇 행인가"를 물으면, "some file, some line"이라고 말한
모호한 세션이 `keywords.py line 443`이라고 말한 세션보다 위로 올라온다. 골드
세션의 파일명·행 번호·커밋 해시를 더해도 15%만 회복된다:

| 질의 | hit@1 |
|---|---|
| 질문만 | 0.000 |
| 오라클 엔티티 3개 추가 | 0.150 |

상위 다섯 개를 읽고, 원점수보다 `corpus_missing`에 무게를 두고, 기억에 라벨을
붙여라——범위, kind, 무엇이 무엇을 대체하는지. 해법은 쓰기 시점에 있다.
질의 측에서 시도한 네 가지는
[D-020](docs/decisions.md#d-020-bag-of-words-retrieval-degrades-on-dense-same-topic-corpora)과
[D-021](docs/decisions.md#d-021-oracle-entities-recover-15-the-failure-is-not-missing-literals)에
있다.

### 기권

`corpus_missing`의 자연스러운 동작점에서:

| 접근 | 유지 | 기권 | 지연 시간 | 상주 메모리 |
|---|---|---|---|---|
| BM25만 | 1.000 | 0.000 | 0.22 ms | 0 |
| **corpus_missing** | **0.875** | **0.632** | ~0 | 0 |
| 로컬 2B | 0.750 | 1.000 | 0.77 s | ~1 GB |
| 로컬 9B | 0.950 | 0.895 | 1.5–2 s | 15.2 GB |
| 호스팅 `deepseek-flash` | 0.925 | 1.000 | 0.81 s | 0 |

0.875의 유지율은 답할 수 있는 질문의 12.5%가 자동으로 거절된다는 뜻이다.
그래서 이 신호는 보고만 하고 자동으로 작용하지 않는다. 호스팅 심사자는 조회당
0.81초를 더한다——한 작업에서 30회면 24초.

## 다른 메모리 서버와 함께 실행하기

**같은 에이전트에 메모리 MCP 서버 두 개를 켜면 안 된다.** `neural-memory`와
`b4a`를 모두 켜면 에이전트가 다른 서버를 조회하고 "메모리에 이 기록이 없다"고
보고했다. `b4a`에는 관련 기억 다섯 개가 있었는데도. 에이전트당 메모리 서버
하나, 또는 선택이 모호하지 않게 만드는 이름을 쓸 것.

## 최전선과 비교

| 기능 | 이 저장소의 상태 |
|---|---|
| 구조화된 기억 (타입이 있는 사실, 선호, 엔티티) | **부분적.** `kind`는 자유 형식 라벨, `tags`는 범위, `refs`는 링크. 스키마도 검증도 없다. |
| 시간·그래프 기억 (유효 구간, 관계 진화) | **미구현.** `created_at`과 `superseded_by`뿐. 유효 구간도 질의 시점 그래프 순회도 없다. 한 홉 `refs`는 PrecisionMemBench에서 4/4. |
| 충돌 해결 | **구현됨.** `supersede(old, new)`. 대체된 기억은 더 이상 나타나지 않는다. PrecisionMemBench 대체 카테고리: 3/3. |
| 쓰기 경로 중복 제거 / 패턴 추출 | **미구현.** 추출(`remember()`)은 존재하지만 **기본값은 꺼짐**(D-023). |
| 다중 검색기 융합 | **미구현.** 검색기는 하나. BM25. |
| 망각 / 통합 | **조악함.** `prune(keep_sessions=N)`은 최신성 기반 크기 제어. `decay`와 `prune_by_weight`가 가중치 신호를 더한다. 잡음 많은 코퍼스를 되살리려면 95%를 삭제해야 한다(D-022). |
| 세션 간 집계 | **미구현.** |
| 능동적 리콜 | **부분적.** 클라이언트 측 훅용 `recall-context`. 스케줄러는 없다. |
| 멀티모달 | **미구현.** |
| 환각 방어 | **부분적.** 증거/판정 분리, `corpus_missing`, 세 상태 심사. 올바른 검색에서 나온 틀린 답을 탐지하는 것은 없다. |
| 공개 수치의 재현성 | **강함.** 벤치마크 셋, 그중 둘은 루프에 모델이 전혀 없다. |

## 현황

라이브러리, CLI, MCP 서버는 완성되어 있다.

**추가됨:** 근사 중복 탐지(`duplicates_of`, SimHash. 보고만 하고 병합하지 않으며
검색 경로에 올라가지 않는다)와 weight(`reinforce`, `decay`, `prune_by_weight`).
검색은 질의에서 의문사를 제거하고, 세션의 상위 3턴을 하나의 점수로 접으며,
코퍼스에 없는 영어 단어를 한 편집 거리로 대조한다(D-037, D-038).

**미구현:** 세션 간 합성, 멀티모달, `kind`의 스키마, 다중 검색기 융합,
리콜 스케줄러, 시간 감쇠 망각.

**측정 후 기각:** 질의 시점 그래프 순회(D-024), 쓰기 시점 추출(D-023),
LoCoMo 세션 청킹(D-037), LongMemEval을 스토어 자체 토크나이저로 실행하는 것
(D-037).

릴리스 이력은 [CHANGELOG.md](CHANGELOG.md)에 있다.
[결정 로그](docs/decisions.md)는 39개 항목을 기록한다.
[연구 노트](docs/research.md)와 [지형 분석](docs/landscape.md)은 주변 분야를
다룬다.

## 기여

```bash
python -m pytest tests/ -q
python -m ruff check src bench tests
python -m ruff format --check src bench tests
python -m bench.zh.e2e
python -m bench.pmb.run
python -m bench.locomo.run
```

`bench/zh/e2e.py`는 `hit@1 0.9500`, `bench.pmb.run`은 `pass 72/77`,
`bench.locomo.run`은 `hit@1 0.6310`을 보고해야 한다. 낮아지면 회귀다. 마지막
둘은 API 키도 네트워크도 필요 없다. 설계 결정을 바꾸려면 근거가 되는 측정과
함께 `docs/decisions.md`에 새 항목을 추가할 것.

## 특별 감사

[nhadaututtheky](https://github.com/nhadaututtheky)의
[neural-memory](https://github.com/nhadaututtheky/neural-memory)에 감사한다.
이 프로젝트는 그곳의 버그 수정에서
시작됐다: 중국어 질의가 키워드 정규식에 한자 범위가 없어 아무것도 반환하지
않았다. 그 brain은 `bench/nmem/`의 코퍼스이기도 하다.

## 라이선스

Apache-2.0. [LICENSE](LICENSE) 참고.
