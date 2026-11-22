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


# ---------------------------------------------------------------------------
# Precomputed artifacts for the demo (python -m hinglishkey.index)
# ---------------------------------------------------------------------------

ARTIFACT_MODES = ("raw", "soundex", "hkey", "hkey_clusters", "hkey_vdrop")


def build_artifacts(out_dir) -> dict:
    """Index all 20,000 tweets under every mode and save them with the variant model.

    Writes index_<mode>.pkl for each mode in ARTIFACT_MODES and
    variants_hkey_clusters.json (fitted on TRAIN only) into out_dir.
    """
    import time
    from pathlib import Path

    from hinglishkey import data, normalize

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {s: data.load_split(s) for s in ("train", "dev", "test")}
    normalize.fit_variants([text for _, text, _ in splits["train"]])
    normalize._variants.save(out / "variants_hkey_clusters.json")
    docs = [(tid, text) for s in ("train", "dev", "test") for tid, text, _ in splits[s]]
    sizes = {}
    for mode in ARTIFACT_MODES:
        t0 = time.perf_counter()
        idx = build_index(docs, mode)
        idx.save(out / f"index_{mode}.pkl")
        sizes[mode] = (idx.n_docs, len(idx.postings), round(time.perf_counter() - t0, 1))
    return sizes


if __name__ == "__main__":
    from pathlib import Path

    target = Path(__file__).resolve().parents[2] / "artifacts"
    for mode, (n, v, s) in build_artifacts(target).items():
        print(f"index_{mode}.pkl: {n} docs, {v} terms ({s} s)")
    print(f"wrote {target}")


class DemoData:
    """Everything the demo tabs need, loaded once per process (see load_demo)."""

    def __init__(self, indexes, rankers, variants, texts, labels, splits, train_freq):
        self.indexes = indexes          # mode -> PositionalIndex
        self.rankers = rankers          # mode -> Ranker
        self.variants = variants        # fitted VariantClusters (train only)
        self.texts = texts              # tweet_id -> text
        self.labels = labels            # tweet_id -> sentiment label
        self.splits = splits            # tweet_id -> "train" / "dev" / "test"
        self.train_freq = train_freq    # token -> frequency in the train split


_demo_cache: dict = {}


def load_demo(artifacts_dir) -> DemoData:
    """Load the saved indexes and variant model (built by python -m hinglishkey.index).

    Cached per directory, so every tab and every rerun shares one copy.
    Raises FileNotFoundError if an artifact is missing.
    """
    from pathlib import Path

    from hinglishkey import data, normalize
    from hinglishkey.rank import Ranker
    from hinglishkey.variants import VariantClusters

    key = str(Path(artifacts_dir).resolve())
    if key in _demo_cache:
        return _demo_cache[key]
    folder = Path(artifacts_dir)
    variants = VariantClusters.load(folder / "variants_hkey_clusters.json")
    normalize._variants = variants          # mode "hkey_clusters" maps query words with it
    indexes = {m: PositionalIndex.load(folder / f"index_{m}.pkl") for m in ARTIFACT_MODES}
    rankers = {m: Ranker(idx) for m, idx in indexes.items()}
    texts, labels, splits = {}, {}, {}
    for s in ("train", "dev", "test"):
        for tid, text, label in data.load_split(s):
            texts[tid], labels[tid], splits[tid] = text, label, s
    demo = DemoData(indexes, rankers, variants, texts, labels, splits, variants.freq)
    _demo_cache[key] = demo
    return demo
