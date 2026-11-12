"""The proximity re-rank: where a query's words sit, not just whether they appear.

Five properties, each of which would be a silent behavioural bug if it broke:

- concentrated beats scattered, and that is the only thing the signal measures
- a document too short to have a passage is left alone
- a keyword query is left alone -- two words are not enough for the signal to be
  anything but noise (measured on the real store, D-042)
- an equal boosted score is resolved by insertion order, not by document order
- `top_k` truncates the ranking instead of reordering it

Every fixture here clears the three-window length floor and the five-term query
floor unless the test is *about* one of those guards. That rule is not tidiness:
an earlier version of this file asserted properties over 35-69 token documents,
where proximity never ran, so deleting the guard it named left the test green
(D-042). Each guard test below is arranged so that removing its guard flips the
result -- the ordering with and without the bonus is deliberately inverted.
"""

from __future__ import annotations

from brain4agent.tokenize import BM25, window_proximity


def _doc(word: str, n: int = 30) -> str:
    return " ".join([word] * n)


def test_proximity_prefers_a_concentrated_passage() -> None:
    # Same words, same count, different layout: `near` keeps them together.
    near = "alpha beta gamma delta " + _doc("filler")
    far = "alpha " + _doc("filler", 15) + " beta " + _doc("filler", 15) + " gamma delta"
    tokens_near = near.split()
    tokens_far = far.split()
    terms = ["alpha", "beta", "gamma", "delta"]
    assert window_proximity(terms, tokens_near, 10) > window_proximity(terms, tokens_far, 10)


def test_proximity_is_a_fraction_of_the_query() -> None:
    tokens = ["alpha", "filler", "beta", "filler", "gamma"]
    assert window_proximity(["alpha"], tokens, 10) == 1.0
    assert window_proximity(["alpha", "missing"], tokens, 10) == 0.5
    assert window_proximity(["nope", "missing"], tokens, 10) == 0.0


def test_proximity_handles_an_empty_query() -> None:
    assert window_proximity([], ["anything", "at", "all"], 10) == 0.0


def test_proximity_handles_a_degenerate_window() -> None:
    """`window` is a public argument; 0 or negative must not raise.

    With `while pos - positions[left] >= span` and a non-positive span, every
    occurrence is consumed on the first iteration and `positions[left]` walks off
    the end. Returning 0.0 keeps a misconfiguration from crashing a query.
    """
    tokens = ["a", "x", "b"]
    assert window_proximity(["a", "b"], tokens, 0) == 0.0
    assert window_proximity(["a", "b"], tokens, -5) == 0.0
    # A window of one token can only ever hold a single term.
    assert window_proximity(["a", "b"], tokens, 1) == 0.5


def test_length_floor_is_configurable_so_its_cost_can_be_measured() -> None:
    """`proximity_min_doc_tokens` exists so a guard's cost is a number, not a story.

    0 keeps the built-in three-window rule; a smaller value lets short documents
    earn the bonus. `bench.guard_attribution_probe` relies on this to measure
    what the guard is worth.
    """
    query = "alpha beta gamma delta epsilon"
    filler = " ".join(["filler"] * 20)
    # 4 gaps of 20 plus the 5 terms = 85 tokens: below the 120 floor, so the
    # default rule skips it entirely.
    short = f" {filler} ".join(query.split())
    assert len(short.split()) < 40 * 3
    docs = [("short", short), ("other", query + " " + " ".join(["filler"] * 60))]

    guarded = BM25(proximity_bonus=3.0, proximity_min_terms=5)
    guarded.index(docs)
    reduced = BM25(proximity_bonus=3.0, proximity_min_terms=5, proximity_min_doc_tokens=1)
    reduced.index(docs)
    # Both documents are scored with the floor lowered, and their boosting differs.
    assert guarded.search(query, top_k=2) != reduced.search(query, top_k=2)


def test_bonus_moves_a_concentrated_document_up() -> None:
    index = BM25(proximity_bonus=2.0, proximity_min_terms=2)
    index.index(
        [
            # Both documents contain every query term once and have the same
            # length -- above the three-window floor, or neither is scored.
            ("far", "alpha " + _doc("filler", 200) + " beta"),
            ("near", "alpha beta " + _doc("filler", 200)),
        ]
    )
    ranked = [mid for mid, _ in index.search("alpha beta", top_k=2)]
    assert ranked[0] == "near", ranked


def test_bonus_is_off_by_default() -> None:
    assert BM25().proximity_bonus == 0.0


def test_bonus_skips_documents_too_short_to_have_a_passage() -> None:
    """Under three windows the signal degenerates into coverage (D-042)."""
    short = " ".join(["alpha", "beta"] + ["x"] * 10)
    assert len(short.split()) < 40 * 3
    index = BM25(proximity_bonus=5.0, proximity_min_terms=2)
    index.index([("short", short), ("other", "nothing to see here")])
    with_bonus = dict(index.search("alpha beta", top_k=2))
    plain = BM25(proximity_min_terms=2)
    plain.index([("short", short), ("other", "nothing to see here")])
    assert with_bonus == dict(plain.search("alpha beta", top_k=2))


def test_bonus_skips_keyword_queries() -> None:
    """Two to four words is a lookup, not a question; measured harmful on the real store.

    Both documents clear the length floor, and the one whose terms are *apart* is
    inserted first. Without the term guard, proximity lifts the adjacent one over
    it; with the guard both keep BM25's order, which is the insertion order. That
    inversion is what makes this test able to fail -- an earlier version used a
    62-token body, so the length floor skipped both documents and deleting the
    term guard changed nothing.
    """
    body = _doc("filler", 200)
    # `apart` is inserted first, so plain BM25 ranks it first (same terms, same
    # length). Proximity would put `together` first, which is the point.
    docs = [
        ("apart", f"alpha {body} beta"),
        ("together", f"alpha beta {body}"),
    ]
    assert len(docs[0][1].split()) >= 40 * 3

    plain = BM25(proximity_min_terms=5)
    plain.index(docs)
    base = [mid for mid, _ in plain.search("alpha beta", top_k=2)]
    assert base[0] == "apart", base

    index = BM25(proximity_bonus=2.0)
    assert index.proximity_min_terms == 5
    index.index(docs)
    # Two terms is below the floor, so the guard keeps BM25's order exactly.
    assert index.search("alpha beta", top_k=2) == plain.search("alpha beta", top_k=2)


def test_top_k_returns_a_prefix_of_the_full_ranking() -> None:
    """`top_k` truncates the ranking; it must not reorder it.

    Documents clear the three-window floor or proximity never runs and the three
    alphas produce one identical ranking -- an earlier version of this test built
    35-69 token documents and proved nothing. The prefix property survives the
    removal of the pruning bounds: it is what any future pruning would have to
    preserve, so it stays as a guard on the shape of the result.
    """
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    pad = 200
    docs = [
        # Every query term, adjacent: proximity 1.0.
        ("d0", " ".join(words) + " " + _doc("filler", pad)),
        # Terms spread across the whole document: proximity well below 1.0.
        (
            "d1",
            "alpha "
            + _doc("filler", pad)
            + " beta "
            + _doc("filler", pad)
            + " gamma delta epsilon",
        ),
        # Half the terms, twice: lower BM25 could still be lifted by proximity.
        ("d2", "alpha beta " + _doc("filler", pad) + " alpha beta"),
        # One term only: proximity cannot exceed 1/6 however it is arranged.
        ("d3", "filler " + _doc("filler", pad) + " zeta"),
        ("d4", _doc("filler", pad) + " alpha beta gamma delta epsilon zeta"),
        ("d5", "gamma delta " + _doc("filler", pad) + " epsilon zeta"),
    ]

    for alpha in (0.5, 1.5, 3.0):
        deep = BM25(proximity_bonus=alpha, proximity_min_terms=2)
        deep.index(docs)
        full = [(mid, score) for mid, score in deep.search(" ".join(words), top_k=len(docs))]
        assert len(full) == len(docs), full
        for k in (1, 2, 3):
            shallow = BM25(proximity_bonus=alpha, proximity_min_terms=2)
            shallow.index(docs)
            got = [(mid, score) for mid, score in shallow.search(" ".join(words), top_k=k)]
            assert got == full[:k], (alpha, k, got, full[:k])


def test_a_tie_after_boosting_is_broken_by_insertion_order() -> None:
    """Equal boosted scores resolve by insertion order, not by document order.

    Two documents made to tie exactly after the boost: a higher pre-bonus score
    with weaker proximity, and a lower one with perfect proximity. The inputs to
    `_apply_proximity` are deliberately ordered so that **input order and
    insertion order disagree** -- scattered arrives first but was inserted
    second. Drop the insertion-order key from the sort and this returns the other
    document, so the assertion below is what pins the rule.
    """
    terms = ["alpha", "beta", "gamma", "delta", "epsilon"]
    alpha = 1.5
    pad = " ".join(["filler"] * 140)
    docs = [
        ("concentrated", "alpha beta gamma delta epsilon " + pad),
        ("scattered", "alpha " + pad + " beta gamma delta epsilon"),
    ]
    index = BM25(proximity_bonus=alpha, proximity_window=40, proximity_min_terms=5)
    index.index(docs)

    prox = [window_proximity(terms, seq, 40) for seq in index._seqs]  # noqa: SLF001
    assert prox[0] > prox[1], prox
    factor = [1.0 + alpha * p for p in prox]
    # Scattered keeps the higher pre-bonus score; concentrated ties it after the
    # boost. Scattered is listed first here, but was inserted second, so the tie
    # must go to concentrated -- insertion order, not input order.
    scat = factor[0]
    conc = scat * factor[1] / factor[0]
    ranked = [(1, scat), (0, conc)]

    got = [index._ids[idx] for idx, _ in index._apply_proximity(list(ranked), terms, 2)]  # noqa: SLF001
    assert got == ["concentrated", "scattered"], got
    assert [index._ids[idx] for idx, _ in index._apply_proximity(list(ranked), terms, 1)] == [  # noqa: SLF001
        "concentrated"
    ]


def test_concentration_can_invert_the_bm25_order() -> None:
    """A document that loses on BM25 must still win when it wins on concentration.

    `scattered` is shorter, so BM25 ranks it first (1.02 against 0.82). Its five
    terms are 30 filler tokens apart, so proximity caps at 0.40, while
    `concentrated` keeps them adjacent and scores 1.00. At alpha 3 the order
    inverts. This is the signal doing its whole job; without the inversion the
    rest of the file measures nothing.

    Fixture chosen by `bench/proximity_fixture_probe.py`, which prints the
    scores -- a document containing the query contiguously has proximity 1.00
    however long it is elsewhere, so an earlier version of this test had no
    inversion to measure.
    """
    query = "alpha beta gamma delta epsilon"
    filler = " ".join(["filler"] * 30)
    scattered = f" {filler} ".join(query.split())
    docs = [
        ("scattered", scattered),
        ("concentrated", query + " " + " ".join(["filler"] * 200)),
    ]

    plain = BM25(proximity_min_terms=5)
    plain.index(docs)
    assert [mid for mid, _ in plain.search(query, top_k=1)] == ["scattered"]

    index = BM25(proximity_bonus=3.0, proximity_min_terms=5)
    index.index(docs)
    full = [mid for mid, _ in index.search(query, top_k=len(docs))]
    assert full[0] == "concentrated", full
    for k in (1, 2):
        shallow = BM25(proximity_bonus=3.0, proximity_min_terms=5)
        shallow.index(docs)
        got = [mid for mid, _ in shallow.search(query, top_k=k)]
        assert got == full[:k], (k, got, full[:k])


def test_bonus_raised_after_indexing_degrades_instead_of_raising() -> None:
    """`_seqs` is only built when the bonus was on at index time.

    A caller that flips the attribute afterwards used to reach `self._seqs[idx]`
    with nothing there and get IndexError out of a read. Falling back to plain
    BM25 is the short degradation path; a crash in `search` is not.
    """
    query = "alpha beta gamma delta epsilon"
    pad = " ".join(["filler"] * 200)
    docs = [
        ("concentrated", f"{query} {pad}"),
        ("scattered", "alpha " + pad + " beta gamma delta epsilon"),
    ]
    index = BM25(proximity_min_terms=5)
    index.index(docs)
    assert index._seqs == []  # noqa: SLF001
    plain = index.search(query, top_k=2)

    index.proximity_bonus = 1.5
    assert index.search(query, top_k=2) == plain


def test_bonus_on_at_index_time_produces_the_bonus() -> None:
    """The mirror of the guard above: the honest path must still apply it."""
    query = "alpha beta gamma delta epsilon"
    filler = " ".join(["filler"] * 30)
    docs = [
        ("scattered", f" {filler} ".join(query.split())),
        ("concentrated", f"{query} " + " ".join(["filler"] * 200)),
    ]
    index = BM25(proximity_bonus=3.0, proximity_min_terms=5)
    index.index(docs)
    assert len(index._seqs) == len(docs)  # noqa: SLF001
    assert index.search(query, top_k=1)[0][0] == "concentrated"


def test_stable_across_repeated_calls() -> None:
    docs = [(f"d{i}", f"alpha beta {' '.join(['filler'] * (200 + i))}") for i in range(8)]
    first = None
    for _ in range(3):
        index = BM25(proximity_bonus=1.5, proximity_min_terms=2)
        index.index(docs)
        got = [mid for mid, _ in index.search("alpha beta", top_k=8)]
        if first is None:
            first = got
        assert got == first


def test_chinese_has_no_proximity_terms() -> None:
    """A Chinese query has no ASCII words left, so the signal never fires (D-042).

    Both documents are padded past the length floor, so the only thing keeping
    the ranking identical is the ASCII filter on the query. Remove that filter
    and the Chinese terms join the window, the bonus applies, and the two
    rankings diverge -- an earlier version used ~17-token documents, where the
    floor did the skipping and the filter was untested.
    """
    padding = "补充说明" * 40  # ~160 CJK characters, ~320 index bigrams
    docs = [
        ("a", f"中文查询搜不到记忆，因为正则不含汉字。{padding}"),
        ("b", f"中文查询失败。过滤阶段把整句话变成了一个空列表。{padding}"),
    ]
    query = "中文查询为什么搜不到"

    plain = BM25(proximity_min_terms=2)
    plain.index(docs)
    base = plain.search(query, top_k=2)

    # The documents must clear the floor, or removing the filter proves nothing.
    with_bonus = BM25(proximity_bonus=2.0, proximity_min_terms=2)
    with_bonus.index(docs)
    assert all(len(seq) >= 40 * 3 for seq in with_bonus._seqs), [len(s) for s in with_bonus._seqs]  # noqa: SLF001
    # No ASCII term survives the filter, so the signal cannot fire.
    assert with_bonus.search(query, top_k=2) == base
