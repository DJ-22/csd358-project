"""Tests for src/hinglishkey/rank.py."""

import math
import random
from collections import Counter

import numpy as np
import pytest

from hinglishkey.index import PositionalIndex
from hinglishkey.rank import Ranker, idf, log_tf, ltc_query_weights, top_k


def make_index(docs):
    idx = PositionalIndex("raw")
    for i, text in enumerate(docs):
        idx.add(f"t{i}", text.split())
    return idx


# --- Weights (L3) -----------------------------------------------------------------

def test_log_tf_and_idf():
    assert log_tf(0) == 0.0
    assert log_tf(1) == 1.0
    assert log_tf(10) == pytest.approx(2.0)
    assert idf(1_000_000, 1000) == pytest.approx(3.0)


def test_lnc_ltc_textbook_example():
    # IIR Example 6.4: query "best car insurance", document "car insurance
    # auto insurance", N = 1,000,000; df auto 5000, best 50000, car 10000,
    # insurance 1000. The book's score is 0.8.
    df = {"auto": 5000, "best": 50000, "car": 10000, "insurance": 1000}
    wq = ltc_query_weights(["best", "car", "insurance"], df, 1_000_000)
    assert wq["best"] == pytest.approx(0.34, abs=0.005)
    assert wq["car"] == pytest.approx(0.52, abs=0.005)
    assert wq["insurance"] == pytest.approx(0.78, abs=0.005)
    doc_tf = {"auto": 1, "car": 1, "insurance": 2}
    raw = {t: log_tf(c) for t, c in doc_tf.items()}
    norm = math.sqrt(sum(x * x for x in raw.values()))
    wd = {t: x / norm for t, x in raw.items()}
    score = sum(w * wd.get(t, 0.0) for t, w in wq.items())
    assert score == pytest.approx(0.80, abs=0.005)


def test_query_terms_missing_from_index_get_no_weight():
    assert ltc_query_weights(["gone"], {"gone": 0}, 10) == {}
    assert set(ltc_query_weights(["a", "gone"], {"a": 2, "gone": 0}, 10)) == {"a"}


def test_term_in_every_doc_has_zero_idf():
    idx = make_index(["a b", "a c"])
    assert Ranker(idx).scores(["a"]) == {}


# --- Cosine scoring ---------------------------------------------------------------------

def brute_force_scores(docs, query):
    """Dense lnc.ltc cosine for every doc, for cross-checking."""
    vocab = sorted({t for d in docs for t in d.split()})
    col = {t: i for i, t in enumerate(vocab)}
    n = len(docs)
    df = Counter(t for d in docs for t in set(d.split()))
    D = np.zeros((n, len(vocab)))
    for i, d in enumerate(docs):
        for t, c in Counter(d.split()).items():
            D[i, col[t]] = 1 + math.log10(c)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    q = np.zeros(len(vocab))
    for t, c in Counter(query).items():
        if t in col:
            q[col[t]] = (1 + math.log10(c)) * math.log10(n / df[t])
    if np.linalg.norm(q) == 0:
        return np.zeros(n)
    return D @ (q / np.linalg.norm(q))


def test_scores_match_dense_cosine():
    rng = random.Random(42)
    vocab = [f"w{i}" for i in range(30)]
    docs = [" ".join(rng.choices(vocab, k=rng.randint(1, 15))) for _ in range(300)]
    ranker = Ranker(make_index(docs))
    for _ in range(50):
        q = rng.choices(vocab + ["unseen"], k=rng.randint(1, 5))
        dense = brute_force_scores(docs, q)
        got = ranker.scores(q)
        for d in range(len(docs)):
            assert got.get(d, 0.0) == pytest.approx(dense[d], abs=1e-12)


def test_search_ranks_closer_doc_first():
    docs = ["bahut accha movie", "accha", "bahut bura movie", "kal milte"]
    ranker = Ranker(make_index(docs))
    results = ranker.search(["bahut", "accha"], k=10)
    assert [d for d, _ in results][:1] == [0]
    assert 3 not in [d for d, _ in results]
    assert all(results[i][1] >= results[i + 1][1] for i in range(len(results) - 1))


# --- Heap top-K (L7) -----------------------------------------------------------------------

def test_top_k_matches_full_sort_with_ties():
    rng = random.Random(42)
    for _ in range(200):
        scores = {d: rng.choice([0.1, 0.2, 0.3, rng.random()]) for d in rng.sample(range(500), rng.randint(0, 60))}
        k = rng.randint(0, 20)
        expected = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        assert top_k(scores, k) == expected


def test_top_k_edge_cases():
    assert top_k({}, 10) == []
    assert top_k({3: 0.5}, 0) == []
    assert top_k({3: 0.5, 1: 0.5}, 5) == [(1, 0.5), (3, 0.5)]
