"""Tests for the shared tf-idf vectors and classify/rocchio.py, plus agreement with scikit-learn."""

import math

import pytest

from hinglishkey.classify import TfidfVectorizer, dot, log_tf
from hinglishkey.classify.rocchio import Rocchio


# --- L3 tf-idf (ltc) --------------------------------------------------------------

def test_log_tf():
    assert log_tf(0) == 0.0
    assert log_tf(1) == 1.0
    assert log_tf(10) == pytest.approx(2.0)
    assert log_tf(1000) == pytest.approx(4.0)


def test_tfidf_by_hand():
    docs = [["bahut", "accha"], ["bahut", "bura", "bura"], ["movie"]]
    vec = TfidfVectorizer().fit(docs)
    # N = 3; df: bahut 2, accha 1, bura 1, movie 1
    assert vec.idf["bahut"] == pytest.approx(math.log10(3 / 2))
    assert vec.idf["bura"] == pytest.approx(math.log10(3))
    v = vec.transform_one(["bahut", "bura", "bura"])
    w_bahut = 1 * math.log10(3 / 2)
    w_bura = (1 + math.log10(2)) * math.log10(3)
    length = math.sqrt(w_bahut ** 2 + w_bura ** 2)
    assert v["bahut"] == pytest.approx(w_bahut / length)
    assert v["bura"] == pytest.approx(w_bura / length)
    assert dot(v, v) == pytest.approx(1.0)                      # unit length


def test_tfidf_edge_cases():
    vec = TfidfVectorizer().fit([["hai", "accha"], ["hai", "bura"]])
    assert "hai" not in vec.transform_one(["hai", "accha"])     # in every doc -> idf 0 -> dropped
    assert vec.transform_one(["unseen"]) == {}                  # unknown terms ignored
    assert vec.transform_one(["hai"]) == {}                     # nothing informative -> zero vector


# --- L13 Rocchio --------------------------------------------------------------------

TRAIN = [(["accha", "movie"], "pos"), (["accha", "khush"], "pos"),
         (["bura", "movie"], "neg"), (["bura", "gussa"], "neg")]


@pytest.fixture
def rocchio():
    return Rocchio().fit([d for d, _ in TRAIN], [c for _, c in TRAIN])


def test_centroid_is_mean_of_unit_vectors(rocchio):
    vecs = rocchio.vectorizer.transform([d for d, c in TRAIN if c == "pos"])
    for t in ("accha", "movie", "khush"):
        expected = sum(v.get(t, 0.0) for v in vecs) / len(vecs)
        assert rocchio.centroids["pos"].get(t, 0.0) == pytest.approx(expected)


def test_nearest_centroid_prediction(rocchio):
    assert rocchio.predict_one(["accha"]) == "pos"
    assert rocchio.predict_one(["gussa", "bura"]) == "neg"
    d = rocchio.distances(["accha"])
    assert d["pos"] < d["neg"]
    # distance formula |mu|^2 - 2 mu.v + 1 equals the explicit squared Euclidean distance
    v = rocchio.vectorizer.transform_one(["accha"])
    mu = rocchio.centroids["neg"]
    terms = set(v) | set(mu)
    explicit = sum((mu.get(t, 0.0) - v.get(t, 0.0)) ** 2 for t in terms)
    assert d["neg"] == pytest.approx(explicit)


def test_agrees_with_sklearn_on_sentimix_dev(sentimix_raw, sentimix_raw_tfidf):
    """Config: mode "raw", trained on train, predicting dev; sklearn's NearestCentroid
    (Euclidean) gets exactly our tf-idf vectors. Expect identical predictions."""
    from sklearn.neighbors import NearestCentroid

    X_train, y_train, X_dev, _ = sentimix_raw
    V_train, V_dev = sentimix_raw_tfidf
    model = Rocchio().fit(X_train, y_train)
    ours = model.predict(X_dev)
    theirs = NearestCentroid().fit(V_train, y_train).predict(V_dev).tolist()

    agreement = sum(a == b for a, b in zip(ours, theirs)) / len(ours)
    # On a failure, list each disagreeing dev doc with our centroid distances, so that a
    # float-level near-tie can be told apart from a real bug (smallest gap on dev is ~6e-6).
    diffs = {j: (ours[j], theirs[j], model.distances(X_dev[j]))
             for j in range(len(ours)) if ours[j] != theirs[j]}
    assert agreement >= 0.99, diffs
    assert agreement == 1.0, diffs
