"""L12 multinomial Naive Bayes with add-one smoothing, scored in log space.

    c_map = argmax_c [ log P(c) + sum_{1 <= k <= n_d} log P(t_k | c) ]

    P(c)     = N_c / N                                   (fraction of training docs in class c)
    P(t | c) = (T_ct + 1) / (sum_{t'} T_ct' + B)          (add-one / Laplace smoothing)

T_ct = number of occurrences of term t in the training docs of class c, B = |V| = size of
the training vocabulary.

- Bayes rule: P(c|d) is proportional to P(c) P(d|c); P(d) is the same for every class, so it is dropped.
- Independence assumption: P(d|c) = product of P(t_k|c) over the token positions of d.
- Log space: a product of many small probabilities underflows to 0.0 in floating point;
  log is monotonic, so the argmax of the sum of logs is the same class.
- Add-one: without it, one term never seen with class c (T_ct = 0) makes P(d|c) = 0,
  whatever the other evidence says.
- Tokens not in the training vocabulary are skipped (they carry no training evidence).
"""

from __future__ import annotations

import math
from collections import Counter


class MultinomialNB:

    def fit(self, docs: list[list[str]], labels: list[str]) -> "MultinomialNB":
        if len(docs) != len(labels):
            raise ValueError("docs and labels must have the same length")
        self.classes = sorted(set(labels))
        n_docs = len(docs)

        # Prior: log P(c) = log(N_c / N)
        n_c = Counter(labels)
        self.log_prior = {c: math.log(n_c[c] / n_docs) for c in self.classes}

        # T_ct: term counts per class (every occurrence counts, it is a multinomial model)
        counts = {c: Counter() for c in self.classes}
        for tokens, c in zip(docs, labels):
            counts[c].update(tokens)
        self.vocab = set().union(*counts.values())
        B = len(self.vocab)

        # Smoothed conditional probabilities log P(t|c) for every t in V
        self.log_cond = {}
        for c in self.classes:
            denominator = sum(counts[c].values()) + B
            self.log_cond[c] = {t: math.log((counts[c][t] + 1) / denominator) for t in self.vocab}
        return self

    def log_scores(self, tokens: list[str]) -> dict[str, float]:
        """log P(c) + sum over the document's known tokens of log P(t|c), for every class."""
        known = [t for t in tokens if t in self.vocab]
        return {c: self.log_prior[c] + sum(self.log_cond[c][t] for t in known) for c in self.classes}

    def predict_one(self, tokens: list[str]) -> str:
        scores = self.log_scores(tokens)
        return max(self.classes, key=lambda c: scores[c])   # ties -> first class in sorted order

    def predict(self, docs: list[list[str]]) -> list[str]:
        return [self.predict_one(tokens) for tokens in docs]

    def predict_proba(self, tokens: list[str]) -> dict[str, float]:
        """Posterior P(c|d) by normalising exp(log score) over the classes.

        Subtracting the largest score first keeps exp() from underflowing.
        """
        scores = self.log_scores(tokens)
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        total = sum(exp.values())
        return {c: e / total for c, e in exp.items()}
