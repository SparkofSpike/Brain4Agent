"""Evidence for the caller to judge, instead of a verdict computed here.

Every cheap abstention signal was measured and all of them failed (see
docs/decisions.md D-008, D-012). At 95% of answerable queries kept, refusal
tops out around 10% for BM25 score, IDF coverage or unseen-word ratio, and 21%
for a logistic regression over thirteen features. A model reaches 89-100%, but
the caller is an agent that already has one, and paying for a second inference
to answer "is this relevant" is duplicate spend on the critical path.

So this module computes no verdict. It retrieves and reports what it can check
for free, including the one signal that correlates with an unanswerable
question: **a term the question treats as central that the corpus never
contains.** The caller decides what to do with that.

``missing_terms`` is the useful field, and it is a list of words rather than a
score on purpose: a caller can show it, log it, or ignore it, and a human can
read it without knowing what a BM25 score of 13.4 means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from brain4agent.tokenize import load_jieba, word_tokens

# Question terms that are structural rather than about the answer. Reporting
# "什么" as missing would be true of every question and useless.
_INTERROGATIVE = frozenset(
    {
        "什么",
        "怎么",
        "怎样",
        "哪个",
        "哪些",
        "哪里",
        "多少",
        "几次",
        "多久",
        "为什么",
        "是否",
        "有没有",
        "如何",
    }
)

_FILLER = frozenset("的了是在和与就都而及或吧呢吗啊呀哦嗯我你他她它这那一个们")

# Part-of-speech tags worth reporting as missing.
#
# Without this filter the top-1 keep rate was 0.325 -- 27 false refusals out of
# 40 answerable questions -- because "直接", "应该", "不是", "哪几个" and "最终"
# are in plenty of *questions* and in no *corpus*. They are the grammar of
# asking, not the content of an answer. Only nominals and things that look like
# names can plausibly be the thing a question is missing.
_CONTENT_POS = frozenset(
    {
        "n",  # 名词
        "nr",  # 人名
        "ns",  # 地名
        "nt",  # 机构团体
        "nz",  # 其他专名
        "ng",
        "nl",
        "eng",  # 英文
        "vn",  # 动词性名词：测试、查询
        "an",  # 形容词性名词
    }
)


class _MemoryLike(Protocol):
    id: str
    session_id: str
    content: str


@dataclass(frozen=True, slots=True)
class Evidence:
    """One retrieved session, with the checks a caller can act on.

    ``coverage`` is the share of the question's IDF-weighted terms present in
    this session. ``missing_terms`` is what *this session* does not contain --
    which explains why it is ranked where it is, but is not by itself evidence
    that the memory is absent. A session missing 用户 is normal; the corpus
    knowing 用户 somewhere else means the concept is not unknown.
    """

    memory: _MemoryLike
    score: float
    coverage: float
    missing_terms: list[str] = field(default_factory=list)

    @property
    def session_id(self) -> str:
        return self.memory.session_id


@dataclass(frozen=True, slots=True)
class SearchResult:
    """What the memory system hands back.

    ``corpus_missing`` is the strong signal: terms the question treats as
    central that appear nowhere in the entire stored corpus. A session
    missing them is irrelevant; the brain missing them is an answer.

    Keeping this separate from ``Evidence.missing_terms`` matters. Conflating
    the two produced 16 false refusals out of 40, because 用户 and 记忆 are
    absent from most individual sessions and present in the corpus.
    """

    query: str
    evidence: list[Evidence]
    corpus_terms: int
    sessions_searched: int
    corpus_missing: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.evidence

    def best(self) -> Evidence | None:
        return self.evidence[0] if self.evidence else None


@dataclass(frozen=True, slots=True)
class JudgedResult:
    """A search plus a hosted judge's opinion, with the distinction kept.

    ``verdict`` is one of:

    - ``found``     -- the judge accepted a session; ``found`` returns it
    - ``not_found`` -- every checked session was rejected
    - ``uncertain`` -- the judge failed for at least one; the answer is unknown

    The third state exists because a judge that cannot be reached is not the
    same as a memory that does not exist. A caller that treats them the same
    will tell a user "I never recorded this" during an outage.
    """

    result: SearchResult
    kept: list[Evidence]
    verdict: str
    checks: int

    @property
    def found(self) -> Evidence | None:
        return self.kept[0] if self.kept else None

    @property
    def certainty(self) -> str:
        return self.verdict


def _is_entity_like(term: str) -> bool:
    """Fallback heuristic when jieba is absent.

    No dictionary means no POS tags, so length and filler content are all there
    is to go on. Noisier than the jieba path, and documented as such rather than
    quietly degraded.
    """
    return len(term) > 1 and term not in _INTERROGATIVE and not all(ch in _FILLER for ch in term)


def _content_words(text: str) -> list[str]:
    """Nominals and names from the question, in order.

    Uses jieba's POS tagger when available. A long digit-bearing token is kept
    even though it tags as ``m``: an IP address or a version number is exactly
    the kind of thing whose absence is worth reporting.
    """
    jieba = load_jieba()
    if jieba is None:
        return [t for t in word_tokens(text) if _is_entity_like(t)]

    import jieba.posseg as pseg

    out: list[str] = []
    for word, flag in pseg.cut(text):
        w = word.strip().lower()
        if not w or w.isspace():
            continue
        long_number = flag == "m" and len(w) >= 3 and any(ch.isdigit() for ch in w)
        if flag in _CONTENT_POS or long_number:
            out.append(w)
    return out


def question_terms(query: str) -> list[str]:
    """Content terms of the question, longest first.

    Longest first because the more specific term is the more informative
    absence: for 数据库 what matters is that no session contains it, not that
    数据 appears somewhere.
    """
    terms = [t for t in _content_words(query) if t not in _INTERROGATIVE and t not in _FILLER]
    seen: set[str] = set()
    unique = [t for t in terms if not (t in seen or seen.add(t))]
    return sorted(unique, key=len, reverse=True)


def build_evidence(
    *,
    query: str,
    hits: list[tuple[str, float]],
    by_session: dict[str, list[_MemoryLike]],
    session_text: dict[str, str],
    id_to_session: dict[str, str],
    sessions_searched: int,
    top_k: int,
) -> SearchResult:
    """Turn ranked memory ids into ranked session evidence.

    Presence is checked by substring, not by token set. For the question this
    answers -- does the corpus contain the word 数据库 -- substring matching is
    both exact and cheap, and it avoids tokenising the whole corpus on every
    search. Tokenising for presence would also be wrong in the other direction:
    a session containing 数据 and 存量 separately is not a session about
    数据库, but a bag of bigrams says it is.
    """
    q_terms = question_terms(query)

    # Computed before the empty-hits early return. "Nothing matched" and "the
    # corpus has never contained 数据库" are different answers, and the second is
    # what explains the first -- dropping it exactly when nothing matched threw
    # away the only useful part of an empty result.
    #
    # Checked once against the joined corpus, not per session: 用户 missing from
    # one session says nothing, 用户 missing from every session says everything.
    texts = list(session_text.values())
    joined = "\n".join(texts)
    corpus_missing = [t for t in q_terms if t not in joined]

    if not hits:
        return SearchResult(
            query=query,
            evidence=[],
            corpus_terms=0,
            sessions_searched=sessions_searched,
            corpus_missing=corpus_missing,
        )

    n_sessions = len(by_session) or 1

    def idf(term: str) -> float:
        d = sum(1 for t in texts if term in t)
        return math.log(1.0 + (n_sessions - d + 0.5) / (d + 0.5))

    total_weight = sum(idf(t) for t in q_terms) or 1.0

    evidence: list[Evidence] = []
    seen: set[str] = set()
    for memory_id, score in hits:
        sid = id_to_session.get(memory_id)
        if sid is None or sid in seen:
            continue
        seen.add(sid)

        text = session_text.get(sid, "")
        present = [t for t in q_terms if t in text]
        missing = [t for t in q_terms if t not in text]
        covered = sum(idf(t) for t in present)

        first = by_session[sid][0]
        evidence.append(
            Evidence(
                memory=first,
                score=score,
                coverage=round(covered / total_weight, 4),
                missing_terms=missing,
            )
        )
        if len(evidence) >= top_k:
            break

    corpus_terms = sum(1 for t in q_terms if any(t in x for x in texts))
    return SearchResult(
        query=query,
        evidence=evidence,
        corpus_terms=corpus_terms,
        sessions_searched=sessions_searched,
        corpus_missing=corpus_missing,
    )


def describe(result: SearchResult) -> str:
    """Human-readable summary of a search."""
    if result.empty:
        line = f"no evidence for {result.query!r}"
        if result.corpus_missing:
            line += f"\n  corpus has no: {', '.join(result.corpus_missing)}"
        return line
    lines = [f"{len(result.evidence)} sessions for {result.query!r}"]
    if result.corpus_missing:
        lines.append(f"  corpus has no: {', '.join(result.corpus_missing)}")
    for e in result.evidence:
        missing = f"  session lacks: {', '.join(e.missing_terms)}" if e.missing_terms else ""
        lines.append(f"  [{e.score:5.2f} cov={e.coverage:.2f}] {e.session_id}{missing}")
    return "\n".join(lines)
