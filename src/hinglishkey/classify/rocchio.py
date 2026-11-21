"""L13 Rocchio classification (nearest centroid) on ltc tf-idf vectors.

Training: one centroid per class, the mean of its length-normalised document vectors
    mu(c) = (1 / |D_c|) * sum_{d in D_c} v(d)

Classification: assign d to the class whose centroid is nearest (Euclidean distance)
    c* = argmin_c |mu(c) - v(d)|

Because |v(d)| = 1,  |mu(c) - v(d)|^2 = |mu(c)|^2 - 2 mu(c).v(d) + 1,  so only the
dot product with each centroid has to be computed per document.

Weakness (L13): one centroid per class assumes each class is a single compact blob.
A multimodal class (e.g. "negative" = angry politics tweets + sad cricket tweets)
gets a centroid between its modes that may be closer to another class.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from hinglishkey.classify import TfidfVectorizer, dot


class Rocchio:

    def fit(self, docs: list[list[str]], labels: list[str]) -> "Rocchio":
        if len(docs) != len(labels):
            raise ValueError("docs and labels must have the same length")
        self.vectorizer = TfidfVectorizer().fit(docs)
        self.classes = sorted(set(labels))
        n_c = Counter(labels)

        # Sum the unit vectors of each class, then divide by |D_c|.
        sums = {c: defaultdict(float) for c in self.classes}
        for v, c in zip(self.vectorizer.transform(docs), labels):
            for t, w in v.items():
                sums[c][t] += w
        self.centroids = {c: {t: w / n_c[c] for t, w in sums[c].items()} for c in self.classes}
        self.sq_norms = {c: dot(mu, mu) for c, mu in self.centroids.items()}
        return self

    def distances(self, tokens: list[str]) -> dict[str, float]:
        """Squared Euclidean distance |mu(c) - v(d)|^2 from the document to every centroid."""
        v = self.vectorizer.transform_one(tokens)
        v_sq = 1.0 if v else 0.0                     # a zero vector has length 0, not 1
        return {c: self.sq_norms[c] - 2 * dot(self.centroids[c], v) + v_sq for c in self.classes}

    def predict_one(self, tokens: list[str]) -> str:
        d = self.distances(tokens)
        return min(self.classes, key=lambda c: d[c])   # ties -> first class in sorted order

    def predict(self, docs: list[list[str]]) -> list[str]:
        return [self.predict_one(tokens) for tokens in docs]
