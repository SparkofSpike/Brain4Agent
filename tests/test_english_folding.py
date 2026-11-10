"""The English side of the query path: stopwords, inflection folding, phrase bonus.

Three changes measured together in D-041. Each gets a test that states the
property rather than the tuning constant:

- a question's tense and number must not decide whether a memory matches
- an English question word must not steer the ranking
- two adjacent query words must outrank the same words scattered apart
"""

from __future__ import annotations

from brain4agent.tokenize import BM25, query_tokens, stem_term


def test_stem_term_merges_inflections() -> None:
    assert stem_term("camped") == stem_term("camping") == stem_term("camps") == "camp"
    assert stem_term("stories") == "story"
    assert stem_term("classes") == "class"
    assert stem_term("watches") == "watch"
    assert stem_term("running") == "run"


def test_stem_term_is_idempotent() -> None:
    """The query path stems, then the index stems again. A second pass must not move."""
    for word in (
        "camped",
        "camping",
        "camps",
        "camp",
        "classes",
        "stories",
        "redis",
        "analysis",
        "bus",
        "is",
        "friends",
        "pursuing",
    ):
        once = stem_term(word)
        assert stem_term(once) == once, word


def test_stem_term_leaves_chinese_and_short_words_alone() -> None:
    assert stem_term("中文") == "中文"
    assert stem_term("数据库") == "数据库"
    assert stem_term("is") == "is"
    assert stem_term("bus") == "bus"


def test_query_tokens_drops_english_stopwords() -> None:
    terms = query_tokens("When did Melanie sign up for a pottery class?")
    assert "when" not in terms
    assert "did" not in terms
    assert "for" not in terms
    # The words that ask about something are what is left.
    assert "pottery" in terms
    assert "class" in terms


def test_query_tokens_keeps_chinese_content_words() -> None:
    terms = query_tokens("根因在哪个文件哪一行？")
    assert "根因" in terms
    assert "哪个" not in terms


def test_bm25_normalize_matches_across_inflection() -> None:
    index = BM25(normalize=stem_term)
    index.index(
        [
            ("a", "We went camping last weekend"),
            ("b", "Someone camped in the office"),
            ("c", "The deployment lives in Frankfurt"),
        ]
    )
    ids = [mid for mid, _ in index.search("camped", top_k=3)]
    assert set(ids) == {"a", "b"}, ids


def test_phrase_bonus_prefers_adjacent_words() -> None:
    index = BM25(normalize=stem_term, phrase_bonus=0.5)
    index.index(
        [
            ("adjacent", "pottery class alpha beta gamma"),
            ("apart", "pottery alpha beta gamma class"),
        ]
    )
    scores = dict(index.search("pottery class", top_k=2))
    assert scores["adjacent"] > scores["apart"], scores


def test_phrase_bonus_off_by_default() -> None:
    index = BM25()
    assert index.phrase_bonus == 0.0


def test_phrase_bonus_does_not_touch_chinese_ranking() -> None:
    """Chinese bigrams already carry adjacency; the pair index must stay English."""
    docs = [
        ("a", "中文查询搜不到记忆"),
        ("b", "中文查询在过滤阶段变成空列表"),
    ]
    plain = BM25()
    plain.index(docs)
    phrased = BM25(phrase_bonus=0.5)
    phrased.index(docs)
    assert plain.search("中文查询") == phrased.search("中文查询")
