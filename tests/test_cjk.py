"""CJK and near-CJK scripts each fail differently. These are the four that did.

Found by probing, not by reading the ranges: the first three were invisible
because the text still *looked* indexed. A run of halfwidth katakana produced
zero terms, a supplementary-plane character was dropped from the middle of a
word, and Vietnamese lost every accented syllable to the ASCII word class. None
of them raised an error.
"""

from __future__ import annotations

import pytest

from brain4agent.tokenize import tokenize


def test_halfwidth_katakana_is_folded_not_dropped() -> None:
    """ｷｰﾜｰﾄﾞ is the same word as キーワード; NFKC says so, the old regex did not."""
    terms = tokenize("ｷｰﾜｰﾄﾞ抽出")
    assert terms, "halfwidth katakana produced no terms at all"
    assert "キー" in terms, terms
    assert "ード" in terms, terms


def test_halfwidth_and_fullwidth_agree() -> None:
    assert tokenize("ｷｰﾜｰﾄﾞ") == tokenize("キーワード")


def test_supplementary_plane_han_survives() -> None:
    """𠮷 is U+20BB7, outside the BMP. It used to be dropped from the middle of 𠮷野家."""
    terms = tokenize("𠮷野家の記憶")
    assert "𠮷野" in terms, terms
    assert "記憶" in terms, terms


def test_supplementary_plane_does_not_split_the_run() -> None:
    """The character must join the run, not act as a separator inside it."""
    terms = tokenize("康熙𤋮字典")
    assert "熙𤋮" in terms, terms


def test_vietnamese_diacritics_fold_to_their_base_letters() -> None:
    """vấn must become van, not v + n."""
    terms = tokenize("truy vấn tiếng Trung")
    assert "van" in terms, terms
    assert "tieng" in terms, terms
    assert "trung" in terms, terms


def test_vietnamese_query_matches_unaccented_text() -> None:
    """Both sides fold the same way, so a query with or without accents matches."""
    assert "van" in tokenize("vấn")
    assert "van" in tokenize("van")


@pytest.mark.parametrize(
    "text",
    [
        "根因在 keywords.py 第 443 行",
        "検索結果が空になった",
        "키워드 추출 정규식",
        "ｷｰﾜｰﾄﾞ抽出",
        "𠮷野家の記憶",
        "truy vấn tiếng Trung",
    ],
)
def test_every_script_yields_terms(text: str) -> None:
    """The invariant the project exists for: never zero tokens for real text."""
    assert tokenize(text), f"no terms for {text!r}"
