"""Near-duplicate detection by SimHash.

Why this exists: two sessions can say the same thing in slightly different
words, and nothing in the store noticed. A user asks which database the service
uses, gets two results, and has to decide himself which one is current. The
project's answer to that (D-027) handles the case where a caller *declared* a
replacement; this handles the case where nobody did.

Scope, deliberately: this module **detects and reports**. It does not merge. A
merge is irreversible and this store has no standing to decide that two memories
are the same — the caller is an agent with a model, and it can read both and
decide. What it could not do was notice in the first place.

The algorithm is SimHash, unchanged from the 2002 paper: hash each shingle, weight
each hash by its frequency, and take the sign of the per-bit sum. Similar texts
produce fingerprints a small Hamming distance apart. Everything here is stdlib —
no numpy, no hashing library.

Chinese is the reason for character shingles rather than word shingles. A word
tokenizer would make two memories with a single character difference look
identical, and this corpus is mostly Han text where one character carries real
meaning.
"""

from __future__ import annotations

import hashlib

BITS = 64
_MASK = (1 << BITS) - 1

# Three characters is the usual shingle size for CJK: one character is too
# discriminating (every common character collides), five is too coarse to tell
# two statements apart.
SHINGLE_SIZE = 3

# 10 of 64 bits is the standard operating point for ~85% textual similarity.
# Lower catches more paraphrases and starts matching unrelated text about the
# same topic; higher misses rewording.
DEFAULT_THRESHOLD = 10


def shingles(text: str, size: int = SHINGLE_SIZE) -> list[str]:
    """Character n-grams over whitespace-normalised, lowercased text."""
    normalized = " ".join(text.lower().split())
    if not normalized:
        return []
    if len(normalized) < size:
        return [normalized]
    return [normalized[i : i + size] for i in range(len(normalized) - size + 1)]


def fingerprint(text: str) -> int:
    """64-bit SimHash of ``text``. Returns 0 for empty input."""
    grams = shingles(text)
    if not grams:
        return 0

    weights = [0] * BITS
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(BITS):
            if value >> bit & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1

    out = 0
    for bit in range(BITS):
        if weights[bit] > 0:
            out |= 1 << bit
    return out & _MASK


def hamming(a: int, b: int) -> int:
    """Bit difference between two fingerprints."""
    return (a ^ b).bit_count()


def near_duplicate(a: int, b: int, threshold: int = DEFAULT_THRESHOLD) -> bool:
    """Whether two fingerprints are close enough to be the same statement.

    Both being 0 means both texts were empty. That is not a duplicate; it is
    absence, and treating it as a match would pair every empty row with every
    other.
    """
    if a == 0 or b == 0:
        return False
    return hamming(a, b) <= threshold
