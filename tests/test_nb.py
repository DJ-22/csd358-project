"""Tests for classify/nb.py: the L12 worked example + agreement with scikit-learn."""

import math

import pytest

from hinglishkey.classify.nb import MultinomialNB

# L12 worked example (IIR Example 13.1): is the test doc about China?
TRAIN = [
    ("Chinese Beijing Chinese", "china"),
    ("Chinese Chinese Shanghai", "china"),
    ("Chinese Macao", "china"),
    ("Tokyo Japan Chinese", "not_china"),
]
TEST_DOC = "Chinese Chinese Chinese Tokyo Japan".split()


@pytest.fixture
def lecture_nb():
    return MultinomialNB().fit([d.split() for d, _ in TRAIN], [c for _, c in TRAIN])


def test_lecture12_priors_and_smoothed_conditionals(lecture_nb):
    nb = lecture_nb
    assert len(nb.vocab) == 6                                    # B = 6
    assert math.exp(nb.log_prior["china"]) == pytest.approx(3 / 4)
    assert math.exp(nb.log_prior["not_china"]) == pytest.approx(1 / 4)
    # china: 8 tokens in total -> denominator 8 + 6 = 14
    assert math.exp(nb.log_cond["china"]["Chinese"]) == pytest.approx((5 + 1) / 14)    # 3/7
    assert math.exp(nb.log_cond["china"]["Tokyo"]) == pytest.approx((0 + 1) / 14)
    assert math.exp(nb.log_cond["china"]["Japan"]) == pytest.approx((0 + 1) / 14)
    # not_china: 3 tokens -> denominator 3 + 6 = 9
    assert math.exp(nb.log_cond["not_china"]["Chinese"]) == pytest.approx((1 + 1) / 9)  # 2/9
    assert math.exp(nb.log_cond["not_china"]["Tokyo"]) == pytest.approx((1 + 1) / 9)


def test_lecture12_prediction(lecture_nb):
    scores = lecture_nb.log_scores(TEST_DOC)
    # P(china|d5)     ∝ 3/4 * (3/7)^3 * 1/14 * 1/14 ≈ 0.0003
    # P(not_china|d5) ∝ 1/4 * (2/9)^3 * 2/9  * 2/9  ≈ 0.0001
    assert math.exp(scores["china"]) == pytest.approx(3 / 4 * (3 / 7) ** 3 * (1 / 14) ** 2)
    assert math.exp(scores["not_china"]) == pytest.approx(1 / 4 * (2 / 9) ** 3 * (2 / 9) ** 2)
    assert lecture_nb.predict_one(TEST_DOC) == "china"
    proba = lecture_nb.predict_proba(TEST_DOC)
    assert sum(proba.values()) == pytest.approx(1.0)
    assert proba["china"] > proba["not_china"]


def test_unknown_tokens_are_ignored(lecture_nb):
    assert lecture_nb.log_scores(TEST_DOC + ["Paris", "xyz"]) == pytest.approx(lecture_nb.log_scores(TEST_DOC))
    # an empty / all-unknown doc falls back to the prior
    assert lecture_nb.predict_one(["Paris"]) == "china"


def test_log_space_avoids_underflow(lecture_nb):
    long_doc = TEST_DOC * 1000                                   # product of 5000 probabilities
    scores = lecture_nb.log_scores(long_doc)
    assert all(math.isfinite(s) for s in scores.values())
    assert math.prod([3 / 7] * 3000) == 0.0                     # the naive product underflows


def test_agrees_with_sklearn_on_sentimix_dev(sentimix_raw):
    """Config: mode "raw", trained on train, predicting dev. Expect identical predictions."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB

    X_train, y_train, X_dev, _ = sentimix_raw
    ours = MultinomialNB().fit(X_train, y_train).predict(X_dev)

    counts = CountVectorizer(analyzer=lambda tokens: tokens)    # documents are already token lists
    sk = SkMultinomialNB(alpha=1.0).fit(counts.fit_transform(X_train), y_train)
    theirs = sk.predict(counts.transform(X_dev)).tolist()

    agreement = sum(a == b for a, b in zip(ours, theirs)) / len(ours)
    assert agreement >= 0.99
    assert agreement == 1.0
