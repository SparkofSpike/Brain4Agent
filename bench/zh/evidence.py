"""Evidence for the caller to judge, instead of a verdict computed here.

Stage 0 measured every cheap abstention signal and all of them failed:

    BM25 top-1 score        best balanced 0.688, costs 46% of answers
    IDF coverage            0.697
    unseen-word ratio       0.706 (bigram) / 0.730 (jieba)
    high-IDF term missing   at keep>=0.95: refuse only 0.105

A local model asked directly does work -- keep 0.950 / refuse 0.895 on the
Chinese set -- but it holds 15.2 GB while running, which froze this machine
mid-run. Paying that inside a memory system is the wrong shape: the caller is
an agent, and the agent already has a model. Asking it to run a second one to
ask "is this memory relevant" is duplicate spend, not a feature.

So this module computes no verdict. It retrieves, and it reports the checks it
can do for free, including the one that correlates with an unanswerable
question: **a term the question treats as central that the corpus never
contains.** The caller -- which knows the user, the task, and has a model
already loaded -- decides what to do with that.

Two things follow from that split, and both are deliberate:

- this module never returns "not found" on its own. It returns evidence plus
  the checks; only the caller can say "I have nothing".
- every number here is explainable without a model. ``missing_terms`` is a list
  of strings, not a score: a caller can show it, log it, or ignore it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from bench.loader import MemoryItem
from bench.retrieval import BM25, tokenize

# Function characters. Used only to decide which question terms are worth
# reporting as missing -- never to split a run.
_CJK_STOP = frozenset("的了是在和与就都而及或吧呢吗啊呀哦嗯我你他她这那一个们")

# Question terms that are structural rather than about the answer.
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
    }
)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One retrieved memory, with the checks a caller can act on.

    ``coverage`` is the share of the question's weighted terms that appear in
    this session. ``missing_terms`` is the more useful half: the specific terms
    that are absent, so a caller can say "no memory mentions 数据库" rather than
    "confidence 0.31".
    """

    memory: MemoryItem
    score: float
    coverage: float
    missing_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """What the memory system hands back.

    ``evidence`` is ranked. ``corpus_terms`` lets the caller distinguish "this
    session does not mention it" from "nothing in the whole brain mentions it",
    which are different answers to a user.
    """

    query: str
    evidence: list[Evidence]
    corpus_terms: int
    sessions_searched: int

    @property
    def empty(self) -> bool:
        return not self.evidence


def _word_tokens(text: str) -> list[str]:
    """Word-level tokens for reporting, not for indexing.

        Bigrams are right for matching -- they never miss an unknown word -- but
        wrong for *explaining*: the missing terms for "what database is on the test
        machine" come out as 州测, 么数, 据库, which no caller can act on.

        jieba is used here only when present, and only for this. It is a 20 MB
    dictionary, not a model: it costs no inference, no accelerator, no memory
    beyond its lexicon. If it is absent the bigram path is used and the report is
    noisier but still correct.
    """
    try:
        import jieba

        out = [w.strip().lower() for w in jieba.lcut(text)]
        return [w for w in out if w and not w.isspace()]
    except ImportError:
        return tokenize(text)


def _question_terms(query: str) -> list[str]:
    """Content terms of the question, longest first.

    Longest first because the more specific term is the more informative
    absence: for 数据库 the useful signal is that no session contains it, not
    that 数据 appears somewhere.
    """
    terms = [
        t
        for t in _word_tokens(query)
        if len(t) > 1 and not all(ch in _CJK_STOP for ch in t) and t not in _INTERROGATIVE
    ]
    seen: set[str] = set()
    unique = [t for t in terms if not (t in seen or seen.add(t))]
    return sorted(unique, key=len, reverse=True)


def _idf(corpus_df: Counter[str], term: str, n_docs: int) -> float:
    df = corpus_df.get(term, 0)
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def search(
    memories: list[MemoryItem],
    query: str,
    *,
    top_k: int = 5,
) -> SearchResult:
    """Retrieve sessions and report the checks that need no model.

    Sessions, not turns: an answer is often spread across an exchange, and a
    single turn is frequently not enough to judge.
    """
    index = BM25()
    index.index([(m.id, m.content) for m in memories])
    hits = index.search(query, top_k=top_k * 3)

    if not hits:
        return SearchResult(
            query=query,
            evidence=[],
            corpus_terms=0,
            sessions_searched=len({m.session_id for m in memories}),
        )

    # Corpus statistics, for the IDF weighting and for "does anything mention it".
    corpus_df: Counter[str] = Counter()
    for mem in memories:
        for t in set(tokenize(mem.content)):
            corpus_df[t] += 1
    n_docs = len(memories)

    by_id = {m.id: m for m in memories}
    by_session: dict[str, list[MemoryItem]] = {}
    for mem in memories:
        by_session.setdefault(mem.session_id, []).append(mem)

    q_terms = _question_terms(query)
    total_weight = sum(_idf(corpus_df, t, n_docs) for t in q_terms) or 1.0

    evidence: list[Evidence] = []
    seen: set[str] = set()
    for memory_id, score in hits:
        mem = by_id.get(memory_id)
        if mem is None or mem.session_id in seen:
            continue
        seen.add(mem.session_id)

        turns = sorted(by_session[mem.session_id], key=lambda m: m.turn_index)
        session_tokens: set[str] = set()
        for turn in turns:
            session_tokens.update(tokenize(turn.content))
            # Word-level too, so a whole-word term like 数据库 is recognised as
            # present rather than only as its bigrams.
            session_tokens.update(_word_tokens(turn.content))

        present = [t for t in q_terms if t in session_tokens]
        missing = [t for t in q_terms if t not in session_tokens]
        covered = sum(_idf(corpus_df, t, n_docs) for t in present)

        evidence.append(
            Evidence(
                memory=mem,
                score=score,
                coverage=round(covered / total_weight, 4),
                missing_terms=missing,
            )
        )
        if len(evidence) >= top_k:
            break

    return SearchResult(
        query=query,
        evidence=evidence,
        corpus_terms=len(corpus_df),
        sessions_searched=len(by_session),
    )


def describe(result: SearchResult) -> str:
    """Human-readable summary. Used by the bench runner, not by callers."""
    if result.empty:
        return f"no evidence for {result.query!r}"
    lines = [f"{len(result.evidence)} sessions for {result.query!r}"]
    for e in result.evidence:
        missing = f"  missing: {', '.join(e.missing_terms)}" if e.missing_terms else ""
        lines.append(f"  [{e.score:5.2f} cov={e.coverage:.2f}] {e.memory.session_id}{missing}")
    return "\n".join(lines)


def term_in_corpus(memories: list[MemoryItem], term: str) -> bool:
    """Whether any memory contains *term*.

    Lets a caller tell "not in this session" from "not in the brain at all".
    """
    return any(term in tokenize(m.content) for m in memories)
