"""Positional inverted index (L1 Boolean retrieval, L2 positional index).

For every term, the index stores a postings list: (doc, [positions]) pairs
sorted by doc number, with each position list sorted too.

    term -> [(0, [3, 9]), (4, [1]), (7, [0, 5]), ...]

Docs are numbered 0..N-1 in the order they are added; `doc_ids[d]` gives
the original id (the tweet id). The terms are whatever the normalisation
mode produced (raw tokens, Soundex codes, H-Keys or variant classes), so
one index type serves every mode.
"""

import pickle

from hinglishkey.normalize import normalize_tokens, tokenize


class PositionalIndex:
    """term -> postings list of (doc, sorted positions), docs sorted."""

    def __init__(self, mode: str = "raw"):
        self.mode = mode
        self.postings: dict[str, list[tuple[int, list[int]]]] = {}
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []

    # -- building ------------------------------------------------------------

    def add(self, doc_id: str, terms: list[str]) -> None:
        """Append one document (its normalised terms, in order)."""
        d = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(terms))
        positions: dict[str, list[int]] = {}
        for pos, t in enumerate(terms):
            positions.setdefault(t, []).append(pos)
        # Docs arrive in increasing order, so appending keeps lists sorted.
        for t, plist in positions.items():
            self.postings.setdefault(t, []).append((d, plist))

    def terms_of(self, text: str) -> list[str]:
        """Tokenise and normalise text with this index's mode."""
        return normalize_tokens(tokenize(text), self.mode)

    # -- statistics ----------------------------------------------------------

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def df(self, term: str) -> int:
        """Document frequency = length of the term's postings list."""
        return len(self.postings.get(term, ()))

    def n_postings(self) -> int:
        return sum(len(p) for p in self.postings.values())

    def n_positions(self) -> int:
        return sum(len(pl) for p in self.postings.values() for _, pl in p)

    # -- Boolean AND (L1) ----------------------------------------------------

    def and_order(self, terms: list[str]) -> list[str]:
        """Query terms in the order AND processes them: increasing df (L1).

        Starting with the rarest term keeps every intermediate result no
        larger than the smallest postings list seen so far.
        """
        return sorted(set(terms), key=lambda t: (self.df(t), t))

    def boolean_and(self, terms: list[str]) -> list[int]:
        """Docs containing every term, merging postings in increasing-df order."""
        order = self.and_order(terms)
        if not order:
            return []
        result = [d for d, _ in self.postings.get(order[0], [])]
        for t in order[1:]:
            if not result:
                break                          # nothing can survive an empty AND
            result = intersect(result, [d for d, _ in self.postings.get(t, [])])
        return result

    # -- Phrase queries (L2) -------------------------------------------------

    def phrase(self, terms: list[str]) -> list[tuple[int, list[int]]]:
        """Docs where the terms occur consecutively, with phrase start positions.

        Starts from the first term's postings; for the i-th later term keeps
        the starts p where the term occurs at p + i (positional intersection).
        """
        if not terms:
            return []
        matches = list(self.postings.get(terms[0], []))
        for i, t in enumerate(terms[1:], start=1):
            if not matches:
                break
            matches = positional_intersect(matches, self.postings.get(t, []), i)
        return matches

    # -- persistence ---------------------------------------------------------

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "PositionalIndex":
        with open(path, "rb") as f:
            state = pickle.load(f)
        index = cls(state["mode"])
        index.__dict__.update(state)
        return index


def build_index(docs, mode: str = "raw") -> PositionalIndex:
    """Index (doc_id, text) pairs under one normalisation mode.

    For mode "hkey_clusters", normalize.fit_variants() must have been called
    (on TRAIN texts) first.
    """
    index = PositionalIndex(mode)
    for doc_id, text in docs:
        index.add(doc_id, index.terms_of(text))
    return index


def intersect(p1: list[int], p2: list[int]) -> list[int]:
    """Merge two sorted doc lists, keeping docs in both (L1, IIR Fig. 1.6).

    One pass with two pointers: O(len(p1) + len(p2)).
    """
    answer = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return answer


def positional_intersect(p1, p2, offset: int):
    """Positional intersection (L2, after IIR Fig. 2.12).

    p1, p2 : postings lists [(doc, sorted positions), ...], docs sorted.
    Keeps, for each doc in both lists, the positions s of p1 such that
    s + offset is a position in p2. offset = 1 means "the next word".

    Docs are merged with two pointers as in intersect(); inside a doc the two
    position lists are merged the same way. Total cost is linear in the
    lengths of both postings lists, including their positions.
    """
    answer = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        d1, pos1 = p1[i]
        d2, pos2 = p2[j]
        if d1 == d2:
            kept = []
            a = b = 0
            while a < len(pos1) and b < len(pos2):
                target = pos1[a] + offset
                if pos2[b] == target:
                    kept.append(pos1[a])
                    a += 1
                    b += 1
                elif pos2[b] < target:
                    b += 1
                else:
                    a += 1
            if kept:
                answer.append((d1, kept))
            i += 1
            j += 1
        elif d1 < d2:
            i += 1
        else:
            j += 1
    return answer
