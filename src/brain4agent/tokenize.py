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
import unicodedata
from collections import Counter
from collections.abc import Callable

# Scripts that are written without spaces between words, and therefore need
# character bigrams rather than whitespace splitting. The list is longer than
# "Han" on purpose; each range below was silently dropping text before it was
# added (see tests/test_cjk.py).
#
#   \u3040-\u30ff    Hiragana + Katakana
#   \u3400-\u4dbf    CJK Unified Ideographs Extension A
#   \u4e00-\u9fff    CJK Unified Ideographs
#   \uf900-\ufaff    CJK Compatibility Ideographs
#   \uac00-\ud7af    Hangul Syllables
#   \uff66-\uff9f    Halfwidth Katakana (normalised to fullwidth by NFKC, listed
#                    so a lone one never falls into the Latin word class)
#   \U00020000-\U0002ffff  CJK Extensions B-F, supplementary plane
#   \U00030000-\U0003134f  CJK Extension G
_CJK_CLASS = (
    r"\u3040-\u30ff"
    r"\u3400-\u4dbf"
    r"\u4e00-\u9fff"
    r"\uf900-\ufaff"
    r"\uac00-\ud7af"
    r"\uff66-\uff9f"
    r"\U00020000-\U0002ffff"
    r"\U00030000-\U0003134f"
)
_CJK = re.compile(f"[{_CJK_CLASS}]")
_WORD = re.compile(r"[a-z0-9_]+")

# Function characters. Used only to decide whether a bigram carries signal; never
# used to split a run. 会 belongs to 社会/会议/会计, 用 to 用户, 问 to 问题 --
# cutting at these destroys the word instead of separating filler from it.
_CJK_STOP = frozenset("的了是在和与就都而及或吧呢吗啊呀哦嗯我你他她它这那一个们")

# Interrogatives. A question asks with them; an answer rarely answers with them.
# They occur in many questions and in the corpus by coincidence, so they match a
# lot of documents without separating any of them. Measured on the Chinese
# benchmark with everything else held constant: removing them from the query
# side moved hit@1 0.9250 -> 0.9500, hit@5 0.9750 -> 1.0000 and MRR
# 0.9404 -> 0.9600 (bench/zh/tuning_probe.py).
#
# This is narrower than dropping every single-character word, which measured
# worse (hit@1 0.9000, D-019): 和/先/做 carry signal on this corpus, 哪个 does not.
INTERROGATIVES = frozenset(
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

_CJK_RUN = re.compile(f"[{_CJK_CLASS}]+|[^{_CJK_CLASS}]+")

# English query-side stopwords. The same argument as INTERROGATIVES, one language
# over: a question is built out of words an answer rarely repeats. Measured on
# LoCoMo (272 sessions, 1540 questions, session-level index): dropping them moved
# hit@1 0.6310 -> 0.6532 and MRR 0.7402 -> 0.7552, with hit@10 down 0.006 -- see
# docs/decisions.md D-041. They stay in the *index*: a memory that happens to
# contain "the" is not improved by hiding it, and dropping them from both sides
# would only shorten documents.
#
# Chinese has its own, much smaller list above. Applying this one to Chinese text
# is a no-op: jieba never emits these tokens for Han input.
EN_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "would",
        "could",
        "should",
        "can",
        "will",
        "shall",
        "may",
        "might",
        "must",
        # Interrogatives, which the Chinese side filters separately above.
        "what",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "which",
        "why",
        "how",
        # Pronouns.
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        # Demonstratives and connectives that ask rather than answer.
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "then",
        "than",
        "so",
        "such",
        "not",
        "no",
        "nor",
        "too",
        "very",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "only",
        "own",
        "same",
        "up",
        "down",
        "out",
        "off",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
    ]
)


def stem_term(term: str) -> str:
    """Fold English inflection so a question and a memory share one stem.

    Not a linguist's stemmer: a handful of suffix rules that make the shapes a
    question actually takes -- ``camped``/``camping``/``camps``, ``stories``/
    ``story``, ``class``/``classes`` -- collapse onto one token. Correction is
    not the goal; consistency is. Both sides run this, so a crude stem that is
    wrong in the same way on both sides still matches.

    Guards that matter: ASCII letters only (Han, kana and Hangul pass through
    untouched), and four letters or more, so 'is', 'was', 'his' and 'bus' are
    not rewritten. Measured on LoCoMo as part of the query-side change in D-041:
    hit@1 0.6310 -> 0.6525 alone, 0.6702 combined with stopword filtering
    (bench/locomo/tuning_probe.py).

    Idempotent by construction, which the query path relies on: a shipped query
    is already stemmed, and the index applies this to the stored tokens a second
    time. ``camp`` -> ``camp``, ``class`` -> ``class``, ``story`` -> ``story``.
    """
    if len(term) < 4 or not term.isascii() or not term.isalpha():
        return term
    if term.endswith("ies") and len(term) > 5:
        return term[:-3] + "y"
    if term.endswith(("sses", "shes", "ches")):
        return term[:-2]
    if term.endswith("ing") and len(term) > 6:
        stem = term[:-3]
        if len(stem) > 3 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if term.endswith("ed") and len(term) > 5:
        return term[:-2]
    if term.endswith("s") and not term.endswith("ss") and len(term) > 4:
        return term[:-1]
    return term


# Latin letters that decompose to a base plus combining marks, built once at
# import. `str.translate` walks this in C; the equivalent per-character loop was
# 2.3x slower on Chinese text.
#
# The ranges are the three blocks that actually carry precomposed Latin letters:
# Latin-1 Supplement (à-ÿ), Latin Extended-A and -B (Ā-ɏ), and Latin Extended
# Additional (Ạ-ỹ, which is where Vietnamese lives -- U+1EA5 ấ is well past
# U+0250 and was missed on the first attempt).
_LATIN_RANGES = ((0x00C0, 0x0250), (0x1E00, 0x1F00))


def _latin_fold_table() -> dict[int, str]:
    table: dict[int, str] = {}
    for lo, hi in _LATIN_RANGES:
        for cp in range(lo, hi):
            base = unicodedata.normalize("NFD", chr(cp))
            if len(base) > 1 and all(unicodedata.combining(c) for c in base[1:]):
                table[cp] = base[0]
    return table


_HIRAGANA = re.compile(r"[\u3041-\u3096]")
_NEEDS_LATIN_FOLD = re.compile(r"[\u00c0-\u024f\u1e00-\u1eff]")
_LATIN_FOLD = _latin_fold_table()


def _fold(text: str) -> str:
    """Normalise compatibility forms and drop diacritics from Latin letters.

    Three different problems, one pass, and the order matters:

    - NFKC folds halfwidth katakana (ｷｰﾜｰﾄﾞ) to fullwidth (キーワード). Without
      it those characters matched neither `_CJK` nor `_WORD` and vanished from
      the index entirely.
    - NFKC leaves a halfwidth voiced mark as `ト` + U+309B rather than the single
      `ド`, so `_merge_kana_marks` puts the two back together afterwards.
    - Accented Latin letters are folded to their unaccented base, so `vấn`
      becomes `van` instead of `v` + `n`. This is a lookup table rather than a
      per-character NFD loop: the loop form was 2.3x slower on Chinese text
      (0.21 -> 0.47 ms per tokenize), and this is on the query path.

    Folding only Latin is deliberate. Applying NFD to a voiced kana decomposes
    ド into ト + U+3099, which is a combining mark too, so a blanket
    "drop the marks" pass turned every ガ行 into its unvoiced pair silently.
    Hangul has the same shape of hazard: 키 decomposes to two Jamo letters.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _merge_kana_marks(text)
    # Most text -- anything written in Han, kana or Hangul -- has no precomposed
    # Latin letter at all, and translate() on a 200-character string is ~39 us.
    if not _NEEDS_LATIN_FOLD.search(text):
        return text
    return text.translate(_LATIN_FOLD)


# Halfwidth katakana followed by the (already NFKC-folded) voiced or semi-voiced
# mark. NFKC does not compose these, so ﾄﾞ arrives as ト ゛ instead of ド.
_KANA_VOICED = {
    "ウ": "ヴ",
    "カ": "ガ",
    "キ": "ギ",
    "ク": "グ",
    "ケ": "ゲ",
    "コ": "ゴ",
    "サ": "ザ",
    "シ": "ジ",
    "ス": "ズ",
    "セ": "ゼ",
    "ソ": "ゾ",
    "タ": "ダ",
    "チ": "ヂ",
    "ツ": "ヅ",
    "テ": "デ",
    "ト": "ド",
    "ハ": "バ",
    "ヒ": "ビ",
    "フ": "ブ",
    "ヘ": "ベ",
    "ホ": "ボ",
}
_KANA_SEMI = {
    "ハ": "パ",
    "ヒ": "ピ",
    "フ": "プ",
    "ヘ": "ペ",
    "ホ": "ポ",
}
_HIRAGANA_TO_KATAKANA = str.maketrans(
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞ"
    "ただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめも"
    "ゃやゅゆょよらりるれろゎわゐゑをんゔ",
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾ"
    "タダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモ"
    "ャヤュユョヨラリルレロヮワヰヱヲンヴ",
)


def _merge_kana_marks(text: str) -> str:
    """Recompose voice marks NFKC left detached, then fold hiragana to katakana.

    Katakana and hiragana are the same sounds written two ways, and a memory
    written in one should be found by a query in the other. Folding them means
    キーワード and きーわーど share bigrams instead of being disjoint vocabularies.

    Fast path first: Chinese and Korean contain neither a stray voice mark nor
    hiragana, and the character loop below was the single largest cost in
    tokenisation (111 of 422 microseconds) when it ran unconditionally.
    """
    if "゛" not in text and "゜" not in text:
        if not _HIRAGANA.search(text):
            return text
        return text.translate(_HIRAGANA_TO_KATAKANA)

    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if i + 1 < len(text) and text[i + 1] in ("゛", "゜"):
            table = _KANA_VOICED if text[i + 1] == "゛" else _KANA_SEMI
            if ch in table:
                out.append(table[ch])
                i += 2
                continue
            i += 1  # drop the orphaned mark
            continue
        out.append(ch)
        i += 1
    return "".join(out).translate(_HIRAGANA_TO_KATAKANA)


def tokenize(text: str) -> list[str]:
    """Index terms: Latin words plus CJK character bigrams."""
    text = _fold(text).lower()
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


def query_tokens(text: str) -> list[str]:
    """The query-side tokeniser: dictionary words, noise dropped, stems folded.

    The read side is where noise costs ranking; the write side is where dropping
    a clue costs recall. Question words are pure query-side noise: 哪个 matches
    S29's "什么时候用哪个" and every other question, and separates none of them.
    English stopwords are the same shape of noise and are measured in D-041.

    Terms are stemmed here as well as in the index. Doing it on the query side
    only would leave ``camped`` unmatched against ``camp``; the index applies
    ``stem_term`` to its stored tokens at load time, so the two sides agree
    without rewriting anything already in the database.
    """
    return [
        stem_term(t)
        for t in jieba_tokens(text)
        if t not in INTERROGATIVES and t not in EN_STOPWORDS
    ]


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


def _within_one_edit(a: str, b: str) -> bool:
    """True when a and b differ by at most one edit.

    One substitution, one insertion/deletion, or one adjacent transposition --
    the shapes a typed word actually takes. ``typsecript`` is a transposition of
    ``typescript``; ``typscript`` has one missing letter; ``mongod`` is one
    letter short of ``mongodb``.
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        return (
            len(diffs) == 2
            and diffs[1] == diffs[0] + 1
            and a[diffs[0]] == b[diffs[1]]
            and a[diffs[1]] == b[diffs[0]]
        )
    short, long = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(short) and short[i] == long[i]:
        i += 1
    return short[i:] == long[i + 1 :]


def _phrasable(term: str) -> bool:
    """Whether a term can take part in an ASCII phrase.

    The phrase bonus exists for English, where a two-word name ("pottery
    class", "mental health") is one idea and BM25 throws the adjacency away.
    Chinese does not need it: character bigrams already index every adjacent
    pair, so 中文查询 is partly in the index as 中文/文查/查询. Restricting the
    path to ASCII alphabetic tokens keeps the pair index small and keeps the
    Chinese benchmark untouched -- measured, not assumed (D-041).
    """
    return len(term) >= 2 and term.isascii() and term.isalpha()


class BM25:
    """Okapi BM25 over a fixed corpus.

    Deliberately the classic formulation: the point of a baseline is that nobody
    has to trust it.

    ``index_fn`` and ``query_fn`` are separate because they are genuinely
    different jobs. Written memories want every clue kept; a question wants its
    noise dropped. Sharing one tokeniser forces a choice that loses on one side.

    ``fuzzy`` adds one optional matching path: an English word the corpus has
    never contained is also matched against a corpus word one edit away (typo,
    transposition, a missing or extra letter). It is off by default because it
    trades exactness for recall and every published benchmark number was measured
    without it.

    ``normalize`` folds a term on both sides before it is counted or looked up.
    The store passes ``stem_term`` here so an inflection in a question matches
    the same inflection in a memory. Applied at index time rather than baked into
    the persisted token column, so changing the rule never leaves a stored index
    speaking a vocabulary the query side no longer uses.

    ``phrase_bonus`` rewards adjacency: a document containing two adjacent query
    terms in the same order earns ``phrase_bonus * (idf(a) + idf(b))``. Off by
    default, because it costs an extra pair index and only earns its keep where
    names are multi-word. Measured on LoCoMo (D-041): hit@1 0.6702 -> 0.6826 and
    MRR 0.7712 -> 0.7784 at 0.3, with hit@10 one question lower.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        index_fn: Callable[[str], list[str]] | None = None,
        query_fn: Callable[[str], list[str]] | None = None,
        fuzzy: bool = False,
        normalize: Callable[[str], str] | None = None,
        phrase_bonus: float = 0.0,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.index_fn = index_fn or tokenize
        self.query_fn = query_fn or tokenize
        self.fuzzy = fuzzy
        self.normalize = normalize or (lambda term: term)
        self.phrase_bonus = phrase_bonus
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
        # Adjacent ASCII word pairs -> documents containing that bigram, in
        # order. One entry per document per pair: a session repeating a phrase
        # ten times is not ten times more about it, and counting repeats would
        # make the bonus a length signal in disguise.
        self._pair_postings: dict[str, list[int]] = {}
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
        self._pair_postings = {}

        for idx, (_, toks) in enumerate(items):
            seq = [self.normalize(t) for t in toks]
            counts = Counter(seq)
            self._docs.append(counts)
            self._lens.append(sum(counts.values()))
            for term, tf in counts.items():
                self._df[term] += 1
                self._postings.setdefault(term, []).append((idx, tf))
            if self.phrase_bonus > 0.0:
                pairs = dict.fromkeys(
                    f"{a} {b}"
                    for a, b in zip(seq, seq[1:], strict=False)
                    if _phrasable(a) and _phrasable(b)
                )
                for pair in pairs:
                    self._pair_postings.setdefault(pair, []).append(idx)

        self._n = len(items)
        self._avgdl = (sum(self._lens) / self._n) if self._n else 0.0
        self._idf = {
            term: math.log(1.0 + (self._n - df + 0.5) / (df + 0.5)) for term, df in self._df.items()
        }
        # Candidates for the fuzzy path. Built once per index, ASCII letters only.
        # Five is deliberate: English inflection (errors -> error) is one edit and
        # is a legitimate match -- raising this to six cost LoCoMo one case, and
        # skipping plural forms cost five. The reviewer flagged the inflection as
        # a defect; the measurement says it helps.
        self._fuzzy_vocab: dict[str, int] = {}
        if self.fuzzy:
            self._fuzzy_vocab = {
                term: df
                for term, df in self._df.items()
                if len(term) >= 5 and term.isascii() and term.isalpha()
            }

    def _expand(self, terms: list[str]) -> list[str]:
        """Add the corpus word an unknown English query word was probably meant to be.

        Only words the corpus has never contained are expanded, only ASCII words,
        and only from six letters up. Those three guards matter: ``test`` ->
        ``tests`` is a legitimate edit but not a misspelling, and expanding every
        short word would rewrite half of an English query.

        The closest word by document frequency wins, so between two candidates
        the more common one is chosen.
        """
        if not self._fuzzy_vocab:
            return terms
        out = list(terms)
        for term in set(terms):
            if term in self._df or len(term) < 6 or not term.isascii() or not term.isalpha():
                continue
            best: str | None = None
            best_df = -1
            for candidate, df in self._fuzzy_vocab.items():
                if abs(len(candidate) - len(term)) > 1:
                    continue
                if _within_one_edit(term, candidate) and df > best_df:
                    best, best_df = candidate, df
            if best is not None and best not in out:
                out.append(best)
        return out

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._n:
            return []
        q_terms = [self.normalize(t) for t in self.query_fn(query)]
        # Pairs are taken before fuzzy expansion: an appended correction is not
        # adjacent to anything, and treating it as such would invent a phrase.
        q_pairs = [
            (a, b)
            for a, b in zip(q_terms, q_terms[1:], strict=False)
            if self.phrase_bonus > 0.0 and _phrasable(a) and _phrasable(b)
        ]
        if self.fuzzy:
            q_terms = self._expand(q_terms)
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
        pair_bonus_by_doc: dict[int, float] = {}
        for a, b in q_pairs:
            postings = self._pair_postings.get(f"{a} {b}")
            if not postings:
                continue
            weight = self._idf.get(a, 0.0) + self._idf.get(b, 0.0)
            for idx in postings:
                pair_bonus_by_doc[idx] = pair_bonus_by_doc.get(idx, 0.0) + weight
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
            if pair_bonus_by_doc:
                score += self.phrase_bonus * pair_bonus_by_doc.get(idx, 0.0)
            if score > 0.0:
                scores.append((self._ids[idx], score))

        scores.sort(key=lambda kv: (-kv[1], self._order.get(kv[0], 0)))
        return scores[:top_k]
