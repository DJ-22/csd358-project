"""Ranked retrieval with lnc.ltc cosine scores and heap top-K (L3, L7).

SMART notation ddd.qqq (document weighting . query weighting):

    document  lnc :  w(t,d) = (1 + log10 tf(t,d)) / |d|          no idf
    query     ltc :  w(t,q) = (1 + log10 tf(t,q)) * log10(N / df(t)) / |q|

where |d| and |q| are the Euclidean lengths of the unnormalised vectors.
score(q, d) = sum over query terms t of w(t,q) * w(t,d) = cos(q, d).

Scores are accumulated term-at-a-time over the postings lists (L3
CosineScore). The top K of the J docs with a non-zero score come from a
binary max-heap: building it takes at most 2J comparisons and each of the
K pops takes about 2 log J, so 2J + 2K log J in total (L7).
"""

import math
from collections import Counter

from hinglishkey.index import PositionalIndex


def log_tf(tf: int) -> float:
    """1 + log10 tf for tf > 0, else 0 (L3)."""
    return 1 + math.log10(tf) if tf > 0 else 0.0


def idf(n_docs: int, df: int) -> float:
    """log10(N / df) (L3)."""
    return math.log10(n_docs / df)


def ltc_query_weights(query_terms: list[str], df: dict[str, int], n_docs: int) -> dict[str, float]:
    """ltc weights of the query terms; terms with df = 0 get no weight."""
    tf = Counter(query_terms)
    w = {t: log_tf(c) * idf(n_docs, df[t]) for t, c in tf.items() if df.get(t, 0) > 0}
    norm = math.sqrt(sum(x * x for x in w.values()))
    return {t: x / norm for t, x in w.items()} if norm > 0 else {}


class Ranker:
    """lnc.ltc cosine ranking over a PositionalIndex."""

    def __init__(self, index: PositionalIndex):
        self.index = index
        # |d| for lnc: sqrt of the sum of (1 + log10 tf)^2 over the doc's terms.
        sq = [0.0] * index.n_docs
        for plist in index.postings.values():
            for d, positions in plist:
                sq[d] += log_tf(len(positions)) ** 2
        self.doc_norm = [math.sqrt(x) for x in sq]

    def scores(self, query_terms: list[str]) -> dict[int, float]:
        """Cosine score of every doc sharing a term with the query (term-at-a-time)."""
        idx = self.index
        df = {t: idx.df(t) for t in set(query_terms)}
        wq = ltc_query_weights(query_terms, df, idx.n_docs)
        acc: dict[int, float] = {}
        for t, w in wq.items():
            for d, positions in idx.postings[t]:
                acc[d] = acc.get(d, 0.0) + w * log_tf(len(positions)) / self.doc_norm[d]
        return acc

    def search(self, query_terms: list[str], k: int = 10) -> list[tuple[int, float]]:
        """Top-k (doc, score) pairs, best first; ties go to the lower doc number."""
        return top_k(self.scores(query_terms), k)


# ---------------------------------------------------------------------------
# Binary max-heap for top-K (L7)
# ---------------------------------------------------------------------------

def _better(a: tuple[float, int], b: tuple[float, int]) -> bool:
    """a ranks above b: higher score, or equal score and lower doc number."""
    return a[0] > b[0] or (a[0] == b[0] and a[1] < b[1])


def _sift_down(heap: list, i: int, n: int) -> None:
    """Move heap[i] down until both children rank below it."""
    while True:
        left, right, best = 2 * i + 1, 2 * i + 2, i
        if left < n and _better(heap[left], heap[best]):
            best = left
        if right < n and _better(heap[right], heap[best]):
            best = right
        if best == i:
            return
        heap[i], heap[best] = heap[best], heap[i]
        i = best


def top_k(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
    """The k best (doc, score) pairs via a max-heap built bottom-up.

    heapify: sift down every internal node, last to first (<= 2J comparisons);
    then pop the root k times, each pop costing about 2 log J comparisons.
    """
    heap = [(s, d) for d, s in scores.items()]
    n = len(heap)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(heap, i, n)
    out = []
    while n > 0 and len(out) < k:
        s, d = heap[0]
        out.append((d, s))
        n -= 1
        heap[0] = heap[n]
        _sift_down(heap, 0, n)
    return out
