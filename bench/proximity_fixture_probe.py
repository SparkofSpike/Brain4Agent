"""Find a fixture where proximity actually changes the winner.

The point of `test_concentration_can_invert_the_bm25_order` is that a document
which loses on BM25 wins on concentration. Every earlier attempt failed because a
document containing "alpha beta gamma delta epsilon" contiguously has proximity
1.0 no matter how long it is elsewhere -- so `scattered` was never scattered in
the window's terms.

This prints the scores for candidate fixtures so the test asserts a real
inversion rather than a tie-break.

    python -m bench.proximity_fixture_probe
"""

from __future__ import annotations

from brain4agent.tokenize import BM25, window_proximity

QUERY = "alpha beta gamma delta epsilon"
TERMS = QUERY.split()


def doc_with_gap(gap: int) -> str:
    """The query's terms separated by `gap` filler tokens each."""
    filler = " ".join(["filler"] * gap)
    return f" {filler} ".join(TERMS)


def show(label: str, docs: list[tuple[str, str]], alpha: float) -> None:
    index = BM25(proximity_bonus=alpha, proximity_min_terms=5)
    index.index(docs)
    plain = BM25(proximity_min_terms=5)
    plain.index(docs)
    prox = {
        mid: window_proximity(TERMS, seq, 40)
        for mid, seq in zip(index._ids, index._seqs, strict=False)  # noqa: SLF001
    }
    base = dict(plain.search(QUERY, top_k=len(docs)))
    boosted = dict(index.search(QUERY, top_k=len(docs)))
    order_plain = [m for m, _ in plain.search(QUERY, top_k=len(docs))]
    order_boost = [m for m, _ in index.search(QUERY, top_k=len(docs))]
    print(f"{label} (alpha={alpha})")
    print(f"  {'doc':14s} {'len':>5} {'prox':>6} {'bm25':>8} {'boosted':>9}")
    for mid, text in docs:
        print(
            f"  {mid:14s} {len(text.split()):>5} {prox[mid]:>6.2f} "
            f"{base.get(mid, 0.0):>8.2f} {boosted.get(mid, 0.0):>9.2f}"
        )
    print(f"  order without bonus: {order_plain}")
    print(f"  order with bonus   : {order_boost}")
    print(f"  inversion: {'YES' if order_plain[0] != order_boost[0] else 'no'}")
    print()


def main() -> int:
    # `scattered` is short (so BM25 likes it) but its terms are spread out.
    for gap in (30, 32, 36):
        docs = [
            ("scattered", doc_with_gap(gap)),
            ("concentrated", QUERY + " " + " ".join(["filler"] * 200)),
        ]
        if len(doc_with_gap(gap).split()) < 120:
            size = len(doc_with_gap(gap).split())
            print(f"gap={gap}: skipped, document is {size} tokens (< floor 120)\n")
            continue
        show(f"gap={gap}", docs, alpha=3.0)

    # And the version used by the tie-break test, to show it is real.
    show(
        "multi-document",
        [
            ("d0", QUERY + " " + " ".join(["filler"] * 200)),
            (
                "d1",
                "alpha "
                + " ".join(["filler"] * 200)
                + " beta "
                + " ".join(["filler"] * 200)
                + " gamma delta epsilon",
            ),
            ("d3", "filler " + " ".join(["filler"] * 200) + " zeta"),
        ],
        alpha=1.5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
