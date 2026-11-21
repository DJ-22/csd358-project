"""From-scratch text classifiers and the tf-idf document vectors they share.

    nb.py       L12 multinomial Naive Bayes (uses raw term counts)
    rocchio.py  L13 Rocchio / nearest centroid (uses the tf-idf vectors below)
    knn.py      L13 k nearest neighbours      (uses the tf-idf vectors below)

Every classifier takes documents that are already token lists, i.e. the output of
normalize.normalize_tokens(tokenize(text), mode). No normalisation happens here.
"""

from __future__ import annotations

import math
from collections import Counter


def log_tf(tf: int) -> float:
    """L3 log-frequency weight: 1 + log10(tf) if tf > 0, else 0."""
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


class TfidfVectorizer:
    """L3 "ltc" document vectors: w_{t,d} = (1 + log10 tf_{t,d}) * log10(N / df_t), then
    cosine (length) normalisation so that |v(d)| = 1.

    idf is learned from the training documents only. At prediction time, a term never
    seen in training has no idf and is ignored. A term that occurs in every training
    document has idf = log10(1) = 0 and therefore weight 0.

    Vectors are sparse: a dict {term: weight} holding only the non-zero weights.
    """

    def fit(self, docs: list[list[str]]) -> "TfidfVectorizer":
        n_docs = len(docs)
        df = Counter()
        for tokens in docs:
            df.update(set(tokens))            # document frequency: count each term once per doc
        self.idf = {t: math.log10(n_docs / df_t) for t, df_t in df.items()}
        return self

    def transform_one(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(t for t in tokens if t in self.idf)
        weights = {t: log_tf(count) * self.idf[t] for t, count in tf.items()}
        weights = {t: w for t, w in weights.items() if w > 0}
        length = math.sqrt(sum(w * w for w in weights.values()))
        if length == 0:
            return {}                          # no informative known term: the zero vector
        return {t: w / length for t, w in weights.items()}

    def transform(self, docs: list[list[str]]) -> list[dict[str, float]]:
        return [self.transform_one(tokens) for tokens in docs]


def dot(u: dict[str, float], v: dict[str, float]) -> float:
    """Dot product of two sparse vectors (loop over the shorter one)."""
    if len(u) > len(v):
        u, v = v, u
    return sum(w * v[t] for t, w in u.items() if t in v)
