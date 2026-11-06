"""Tokenisation and a BM25 baseline.

Zero dependencies is a constraint, not an accident. Stage 0 is about measuring,
and every library added before the first number is a variable that can mask a
result. When a benchmark shows that a dependency would lift a specific number,
it gets added then -- see docs/decisions.md.

The tokeniser is where a Chinese-first system diverges from an English-first
one. English splits on whitespace; Chinese does not, and every "split on
non-word characters" default silently reduces a Chinese query to one token.
The fallback here is character bigrams: no dictionary, no model, but it never
returns zero tokens for Han text, which is exactly the failure this project
exists to avoid.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Han, Hiragana, Katakana, Hangul. Treated as "no word boundaries inside".
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_WORD = re.compile(r"[a-z0-9_]+")

# Function characters. A Han bigram made mostly of these carries no signal, and
# unlike a per-character filter this cannot destroy a content word: 社会 and 会议
# survive because 社/议 are not in the set, while 的了 and 是在 are dropped.
_CJK_STOP = frozenset("的了是在和与就都而及或吧呢吗啊呀哦嗯我你他她它这那一个们")


def tokenize(text: str) -> list[str]:
    """Split into index terms.

    Latin runs are lowercased words; CJK runs become overlapping bigrams. A
    single Han character becomes a unigram, because dropping it would make a
    one-character query unsearchable.
    """
    text = text.lower()
    tokens: list[str] = []

    for chunk in re.split(r"(\s+)", text):
        if not chunk or chunk.isspace():
            continue
        if _CJK.search(chunk):
            # Split mixed runs (e.g. "v2rayN代理") at the script boundary so the
            # Latin part stays a word and the Han part stays bigrams.
            parts = re.findall(
                r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+|[^\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+",
                chunk,
            )
            for part in parts:
                if not part:
                    continue
                if _CJK.search(part):
                    tokens.extend(_cjk_tokens(part))
                else:
                    tokens.extend(_WORD.findall(part.lower()))
        else:
            tokens.extend(_WORD.findall(chunk))

    return tokens


def _cjk_tokens(run: str) -> list[str]:
    """Bigram a Han run, dropping grams that are filler on both sides.

    The fallback is narrow on purpose. It exists so that a run containing real
    content is never reduced to nothing -- not so that pure filler becomes a
    search term. 的了 should return nothing; 记忆 should never return nothing.
    """
    if len(run) == 1:
        return [] if run in _CJK_STOP else [run]

    grams = [run[i : i + 2] for i in range(len(run) - 1)]
    kept = [g for g in grams if not all(ch in _CJK_STOP for ch in g)]
    if kept:
        return kept

    # Every gram was filler. Only fall back if the run holds a content
    # character somewhere -- otherwise the input really is all filler.
    if any(ch not in _CJK_STOP for ch in run):
        return grams
    return []


class BM25:
    """Okapi BM25 over a fixed corpus of (id, text) pairs.

    Deliberately the classic formulation: the point of a baseline is that
    nobody has to trust it.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._docs: list[Counter[str]] = []
        self._lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._n: int = 0

    def index(self, items: list[tuple[str, str]]) -> None:
        self._ids = [i for i, _ in items]
        self._docs = []
        self._lens = []
        self._df = Counter()

        for _, text in items:
            toks = tokenize(text)
            counts = Counter(toks)
            self._docs.append(counts)
            self._lens.append(len(toks))
            for term in counts:
                self._df[term] += 1

        self._n = len(items)
        self._avgdl = (sum(self._lens) / self._n) if self._n else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._n:
            return []

        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores: list[tuple[str, float]] = []
        for idx, counts in enumerate(self._docs):
            dl = self._lens[idx]
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scores.append((self._ids[idx], score))

        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return scores[:top_k]
