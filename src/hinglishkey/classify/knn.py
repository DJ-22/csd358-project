"""L13 k nearest neighbour classification on ltc tf-idf vectors, cosine similarity.

    1. Represent every training doc as a unit tf-idf vector v(d).
    2. For a test doc q, compute cos(q, d) = q.d for every training doc d
       (both are unit vectors, so the dot product IS the cosine).
    3. Take the k training docs with the highest cosine; predict the majority class among them.

Ties in the vote are broken in favour of the tied class that owns the single nearest
neighbour. A test doc with no known informative term (zero vector) is similar to nothing,
so it gets the most frequent training class.

Efficiency: kNN has no training cost but each prediction compares q with all N training
docs. The cosines are computed term-at-a-time through an inverted index (as in L3
cosine scoring): only training docs sharing at least one term with q get a score, and the
work per query is the sum of the posting-list lengths of q's terms.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from hinglishkey.classify import TfidfVectorizer

K_VALUES = (1, 3, 9, 15)      # the k values from the L13 slide exercise


class KNN:

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def fit(self, docs: list[list[str]], labels: list[str]) -> "KNN":
        if len(docs) != len(labels):
            raise ValueError("docs and labels must have the same length")
        self.vectorizer = TfidfVectorizer().fit(docs)
        self.labels = list(labels)
        self.n_docs = len(docs)
        self.majority = Counter(labels).most_common(1)[0][0]

        # Inverted index: term -> (array of doc ids, array of weights w_{t,d})
        doc_ids, weights = defaultdict(list), defaultdict(list)
        for i, v in enumerate(self.vectorizer.transform(docs)):
            for t, w in v.items():
                doc_ids[t].append(i)
                weights[t].append(w)
        self.postings = {t: (np.array(doc_ids[t]), np.array(weights[t])) for t in doc_ids}
        return self

    def cosines(self, tokens: list[str]) -> np.ndarray:
        """cos(q, d) for every training doc d, accumulated term by term over the postings."""
        scores = np.zeros(self.n_docs)
        for t, q_w in self.vectorizer.transform_one(tokens).items():
            ids, d_w = self.postings[t]
            scores[ids] += q_w * d_w        # each doc id occurs once per posting list
        return scores

    def neighbours(self, tokens: list[str], k: int | None = None) -> list[tuple[int, float]]:
        """The k most similar training docs as (doc id, cosine), most similar first."""
        k = min(k or self.k, self.n_docs)
        scores = self.cosines(tokens)
        top = np.argpartition(-scores, k - 1)[:k]                  # the k largest, unordered
        top = sorted(top, key=lambda i: (-scores[i], i))            # order by cosine, then doc id
        return [(int(i), float(scores[i])) for i in top]

    def predict_one(self, tokens: list[str], k: int | None = None) -> str:
        nearest = self.neighbours(tokens, k)
        if nearest[0][1] == 0.0:
            return self.majority                                     # zero vector / no overlap
        votes = Counter(self.labels[i] for i, _ in nearest)
        best = max(votes.values())
        for i, _ in nearest:                                         # nearest first
            if votes[self.labels[i]] == best:
                return self.labels[i]

    def predict(self, docs: list[list[str]], k: int | None = None) -> list[str]:
        return [self.predict_one(tokens, k) for tokens in docs]
