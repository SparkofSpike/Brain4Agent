"""Tokenisation and BM25.

Zero dependencies is a constraint, not an accident. Every library added before a
number exists is a variable that can mask a result.

The Chinese path is where this diverges from an English-first design. English
splits on whitespace; Chinese does not, and every "split on non-word characters"
default silently reduces a Chinese query to one token. The fallback here is
character bigrams: no dictionary, no model, but it never returns zero tokens for
Han text, which is exactly the failure this project exists to avoid.

Two tokenisations, for two different jobs:

- ``tokenize`` -- bigrams, used for *matching*. They never miss an unknown word:
  ``覆巢演出`` becomes 覆巢/巢演/演出 and any of them can hit.
- ``word_tokens`` -- whole words, used for *explaining* what is missing.
  ``数据库`` sliced into 数据 + 据库 loses the signal, because 数据 appears
  elsewhere in the corpus.

They are deliberately not merged. Using bigrams to explain produces reports full
of 州测 / 址和 / 么数, which no caller can act on; using words to match loses
every proper noun a dictionary has not seen.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_WORD = re.compile(r"[a-z0-9_]+")

# Function characters. Used only to decide whether a bigram carries signal; never
# used to split a run. 会 belongs to 社会/会议/会计, 用 to 用户, 问 to 问题 --
# cutting at these destroys the word instead of separating filler from it.
_CJK_STOP = frozenset("的了是在和与就都而及或吧呢吗啊呀哦嗯我你他她它这那一个们")

_CJK_RUN = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+"
    r"|[^\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+"
)


def tokenize(text: str) -> list[str]:
    """Index terms: Latin words plus CJK character bigrams."""
    text = text.lower()
    tokens: list[str] = []

    for chunk in re.split(r"(\s+)", text):
        if not chunk or chunk.isspace():
            continue
        if not _CJK.search(chunk):
            tokens.extend(_WORD.findall(chunk))
            continue
        for part in _CJK_RUN.findall(chunk):
            if not part:
                continue
            if _CJK.search(part):
                tokens.extend(_cjk_bigrams(part))
            else:
                tokens.extend(_WORD.findall(part))
    return tokens


def _cjk_bigrams(run: str) -> list[str]:
    """Bigram a Han run, dropping grams made entirely of filler.

    The fallback is narrow on purpose: it exists so a run containing real content
    is never reduced to nothing, not so that 的了 becomes a search term.
    """
    if len(run) == 1:
        return [] if run in _CJK_STOP else [run]

    grams = [run[i : i + 2] for i in range(len(run) - 1)]
    kept = [g for g in grams if not all(ch in _CJK_STOP for ch in g)]
    if kept:
        return kept
    if any(ch not in _CJK_STOP for ch in run):
        return grams
    return []


def load_jieba():
    """Import jieba once, with its startup banner silenced.

    The banner goes to stderr and is a full sentence of noise for a CLI user who
    asked one question. Cached so the import is attempted once per process and
    the log level is set once.
    """
    try:
        import jieba
    except ImportError:
        return None
    import logging

    jieba.setLogLevel(logging.ERROR)
    return jieba


def word_tokens(text: str) -> list[str]:
    """Whole words, for reporting rather than matching.

    Uses jieba when it happens to be installed -- a ~20 MB dictionary, no
    inference, no accelerator. Falls back to bigrams, which is noisier in a
    report but never wrong about presence.
    """
    jieba = load_jieba()
    if jieba is None:
        return tokenize(text)

    out = [w.strip().lower() for w in jieba.lcut(text)]
    return [w for w in out if w and not w.isspace()]


def jieba_tokens(text: str) -> list[str]:
    """Dictionary segmentation. Falls back to bigrams without jieba."""
    jieba = load_jieba()
    if jieba is None:
        return tokenize(text)
    return [w.strip().lower() for w in jieba.lcut(text) if w.strip()]


# jieba degrades super-linearly on one long unpunctuated run. Measured: 10k
# identical characters 585 ms, 20k 2.4 s, 100k 85 s, while 50k characters of
# natural text takes 600 ms.
#
# The first attempt at this capped the length and fell back to bigrams past the
# cap. That was worse than the slowness: the index then held no words for long
# memories while queries still used words, so a 12k-character memory containing
# 数据库 a thousand times could not be found by searching 数据库 -- and
# corpus_missing was empty, because the raw text does contain it. A stored
# memory that cannot be recalled and reports no reason is the exact failure this
# project exists to prevent.
#
# Chunking keeps the words and fixes the cost: 100k characters in 2k chunks
# indexes in ~0.4 s (repetitive) to ~1.1 s (natural), against 85 s unchunked.
JIEBA_CHUNK_CHARS = 2_000


def both_tokens(text: str) -> list[str]:
    """Union of bigrams and dictionary words -- the index-side tokeniser.

    Measured, not assumed (bench/zh/indexing_probe.py): indexing with both and
    querying with words moved hit@1 from 0.825 to 0.925 and MRR from 0.891 to
    0.940, while every symmetric pairing stayed at or below 0.875. The wide side
    is the writing side; the precise side is the asking side.

    Duplicate terms inflate tf slightly where a bigram and a word coincide. That
    is a bias in favour of terms both tokenisers agree on, which is not the same
    as a bug, and the measured result already includes it.
    """
    out = tokenize(text)
    if len(text) <= JIEBA_CHUNK_CHARS:
        out.extend(jieba_tokens(text))
        return out
    # Segment in chunks: words survive, so a query for a 3+ character term still
    # matches, and the super-linear cost stays on short runs where it belongs.
    for i in range(0, len(text), JIEBA_CHUNK_CHARS):
        out.extend(jieba_tokens(text[i : i + JIEBA_CHUNK_CHARS]))
    return out


class BM25:
    """Okapi BM25 over a fixed corpus.

    Deliberately the classic formulation: the point of a baseline is that nobody
    has to trust it.

    ``index_fn`` and ``query_fn`` are separate because they are genuinely
    different jobs. Written memories want every clue kept; a question wants its
    noise dropped. Sharing one tokeniser forces a choice that loses on one side.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        index_fn: Callable[[str], list[str]] | None = None,
        query_fn: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.index_fn = index_fn or tokenize
        self.query_fn = query_fn or tokenize
        self._ids: list[str] = []
        self._docs: list[Counter[str]] = []
        self._lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._n: int = 0
        self._order: dict[str, int] = {}
        # term -> [(doc index, term frequency)]. The point of an inverted index
        # here is not elegance: without it, every query walks all documents and
        # the cost grows linearly with the corpus. Measured before adding it:
        # 20k documents took 144 ms per query and rising.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        # Precomputed because idf depends only on df and n, and both are fixed
        # once indexing finishes. Recomputing it per (query, document, term) was
        # three logarithms per pair.
        self._idf: dict[str, float] = {}

    def __len__(self) -> int:
        return self._n

    def index(self, items: list[tuple[str, str]]) -> None:
        """Index raw text, tokenising with ``index_fn``."""
        self.index_tokens([(i, self.index_fn(t)) for i, t in items])

    def index_tokens(self, items: list[tuple[str, list[str]]]) -> None:
        """Index pre-tokenised documents.

        Lets a store persist its tokens and skip re-segmenting the corpus on
        every rebuild, which is the difference between a 3 ms rebuild and a
        30 ms one at a hundred turns.
        """
        self._ids = [i for i, _ in items]
        # Insertion order, used to break score ties. Ties are not a corner case
        # here: an entity-swapped hard negative scores within a hair of its
        # original, and breaking the tie by identifier made the winner depend on
        # whether the id happened to be a UUID. Same query, different answer,
        # run to run.
        self._order = {i: pos for pos, (i, _) in enumerate(items)}
        self._docs = []
        self._lens = []
        self._df = Counter()
        self._postings = {}

        for idx, (_, toks) in enumerate(items):
            counts = Counter(toks)
            self._docs.append(counts)
            self._lens.append(sum(counts.values()))
            for term, tf in counts.items():
                self._df[term] += 1
                self._postings.setdefault(term, []).append((idx, tf))

        self._n = len(items)
        self._avgdl = (sum(self._lens) / self._n) if self._n else 0.0
        self._idf = {
            term: math.log(1.0 + (self._n - df + 0.5) / (df + 0.5)) for term, df in self._df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._n:
            return []
        q_terms = self.query_fn(query)
        if not q_terms:
            return []

        # Only documents containing at least one query term can score above zero,
        # so walk the postings instead of the corpus. A query with one rare term
        # touches a handful of documents out of tens of thousands.
        candidates: set[int] = set()
        for term in q_terms:
            posting = self._postings.get(term)
            if posting:
                candidates.update(idx for idx, _ in posting)
        if not candidates:
            return []

        avgdl = self._avgdl or 1.0
        scores: list[tuple[str, float]] = []
        for idx in candidates:
            counts = self._docs[idx]
            dl = self._lens[idx]
            norm = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                score += self._idf[term] * (tf * (self.k1 + 1.0)) / (tf + norm)
            if score > 0.0:
                scores.append((self._ids[idx], score))

        scores.sort(key=lambda kv: (-kv[1], self._order.get(kv[0], 0)))
        return scores[:top_k]
