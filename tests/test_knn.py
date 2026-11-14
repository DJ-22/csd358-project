"""Tests for classify/knn.py: hand-made cases + agreement with scikit-learn."""

from collections import Counter

import pytest

from hinglishkey.classify.knn import K_VALUES, KNN

TRAIN = [(["accha", "movie"], "pos"), (["accha", "khush"], "pos"), (["accha", "gana"], "pos"),
         (["bura", "movie"], "neg"), (["bura", "gussa"], "neg")]


@pytest.fixture
def knn():
    return KNN(k=3).fit([d for d, _ in TRAIN], [c for _, c in TRAIN])


def test_k_values_from_lecture():
    assert K_VALUES == (1, 3, 9, 15)


def test_cosines_match_dot_products(knn):
    from hinglishkey.classify import dot

    q = ["accha", "movie"]
    qv = knn.vectorizer.transform_one(q)
    expected = [dot(qv, dv) for dv in knn.vectorizer.transform([d for d, _ in TRAIN])]
    assert knn.cosines(q).tolist() == pytest.approx(expected)


def test_majority_vote(knn):
    assert knn.predict_one(["accha", "khush"]) == "pos"
    assert knn.predict_one(["bura", "gussa"], k=1) == "neg"
    nearest = knn.neighbours(["bura", "gussa"], k=2)
    assert nearest[0][0] == 4 and nearest[0][1] == pytest.approx(1.0)   # identical doc first


def test_vote_tie_goes_to_class_of_nearest_neighbour():
    knn = KNN(k=2).fit([["bura", "gussa"], ["accha", "movie"]], ["neg", "pos"])
    # one neighbour of each class -> tie; "movie accha" is nearer to the pos doc
    assert knn.predict_one(["movie", "accha"]) == "pos"
    assert knn.predict_one(["gussa"]) == "neg"


def test_zero_vector_gets_majority_class(knn):
    assert knn.predict_one(["never_seen_word"]) == "pos"
    assert knn.predict_one([]) == "pos"


def test_k_larger_than_training_set(knn):
    assert knn.predict_one(["bura"], k=100) == "pos"       # all 5 docs vote: 3 pos vs 2 neg


@pytest.mark.parametrize("k", [1, 3, 15])
def test_agrees_with_sklearn_on_sentimix_dev(k, sentimix_raw, sentimix_raw_tfidf):
    """Config: mode "raw", trained on train, predicting dev; sklearn's KNeighborsClassifier
    (brute force, cosine) gets exactly our tf-idf vectors.

    Explaining the gap: the k nearest neighbours themselves must be identical for every dev doc
    that has a non-zero vector. Predictions can differ only in two cases where the two
    implementations deliberately use different rules:
      1. vote ties: we pick the tied class of the nearest neighbour; sklearn picks the first
         tied label in sorted order (always "negative" if it is tied), which biases one class;
      2. zero vectors (no known informative term): every training doc is equally (dis)similar,
         we answer the majority training class, sklearn's choice is arbitrary.
    Every disagreement must fall into one of these cases; for k = 1 there are no vote ties.
    """
    from sklearn.neighbors import KNeighborsClassifier

    X_train, y_train, X_dev, _ = sentimix_raw
    V_train, V_dev = sentimix_raw_tfidf
    ours_model = KNN(k=k).fit(X_train, y_train)
    sk = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute").fit(V_train, y_train)
    theirs = sk.predict(V_dev).tolist()
    _, sk_neighbours = sk.kneighbors(V_dev)

    n_tie = n_zero = 0
    for j, tokens in enumerate(X_dev):
        nearest = ours_model.neighbours(tokens)
        zero = nearest[0][1] == 0.0
        if not zero:
            assert {i for i, _ in nearest} == set(sk_neighbours[j].tolist())
        ours = ours_model.predict_one(tokens)
        if ours != theirs[j]:
            votes = Counter(y_train[i] for i, _ in nearest)
            tie = list(votes.values()).count(max(votes.values())) > 1
            assert zero or tie, f"unexplained disagreement on dev doc {j}"
            n_zero += zero
            n_tie += tie and not zero

    agreement = 1 - (n_tie + n_zero) / len(X_dev)
    print(f"k={k}: agreement {agreement:.4f} (vote-tie disagreements {n_tie}, zero-vector {n_zero})")
    assert n_zero <= 0.01 * len(X_dev)
    if k == 1:
        assert n_tie == 0 and agreement >= 0.99
