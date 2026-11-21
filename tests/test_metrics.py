"""Tests for metrics.py: hand-computed lecture examples + scikit-learn cross-checks."""

import random

import pytest

from hinglishkey.metrics import (
    accuracy,
    average_precision,
    confusion_matrix,
    pairwise_prf,
    precision_at_k,
    precision_recall_f1,
    recall_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# L6 worked example (contingency table)
#                      relevant   nonrelevant
#   retrieved              20            40
#   not retrieved          60     1,000,000
# P = 20/60 = 1/3, R = 20/80 = 1/4, F1 = 2/7, accuracy ≈ 0.99992 (misleading).
# ---------------------------------------------------------------------------

TP, FP, FN, TN = 20, 40, 60, 1_000_000


def _lecture_lists():
    y_true = ["rel"] * TP + ["nonrel"] * FP + ["rel"] * FN + ["nonrel"] * TN
    y_pred = ["rel"] * TP + ["rel"] * FP + ["nonrel"] * FN + ["nonrel"] * TN
    return y_true, y_pred


def test_lecture6_precision_recall_f1():
    y_true, y_pred = _lecture_lists()
    rel = precision_recall_f1(y_true, y_pred)["per_class"]["rel"]
    assert rel["precision"] == pytest.approx(1 / 3)
    assert rel["recall"] == pytest.approx(1 / 4)
    assert rel["f1"] == pytest.approx(2 / 7)
    assert rel["support"] == TP + FN


def test_lecture6_accuracy_is_misleading():
    y_true, y_pred = _lecture_lists()
    acc = accuracy(y_true, y_pred)
    assert acc == pytest.approx((TP + TN) / (TP + FP + FN + TN))
    assert acc > 0.9999          # looks excellent ...
    # ... yet always answering "nonrel" is even better on accuracy, with recall 0 for "rel".
    trivial = ["nonrel"] * len(y_true)
    assert accuracy(y_true, trivial) > acc
    assert precision_recall_f1(y_true, trivial)["per_class"]["rel"]["recall"] == 0.0


def test_lecture6_confusion_matrix():
    y_true, y_pred = _lecture_lists()
    assert confusion_matrix(y_true, y_pred, ["rel", "nonrel"]) == [[TP, FN], [FP, TN]]


def test_macro_weighted_micro_small_example():
    #            pred: neg  neu  pos
    # true neg        2    1    0
    # true neu        0    1    1
    # true pos        0    0    1
    y_true = ["neg", "neg", "neg", "neu", "neu", "pos"]
    y_pred = ["neg", "neg", "neu", "neu", "pos", "pos"]
    # per class P: neg 2/2, neu 1/2, pos 1/2 ; R: neg 2/3, neu 1/2, pos 1/1
    macro = precision_recall_f1(y_true, y_pred, "macro")
    assert macro["precision"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert macro["recall"] == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    f1s = [2 * 1 * (2 / 3) / (1 + 2 / 3), 0.5, 2 * 0.5 * 1 / 1.5]
    assert macro["f1"] == pytest.approx(sum(f1s) / 3)
    weighted = precision_recall_f1(y_true, y_pred, "weighted")
    assert weighted["f1"] == pytest.approx((3 * f1s[0] + 2 * f1s[1] + 1 * f1s[2]) / 6)
    micro = precision_recall_f1(y_true, y_pred, "micro")
    assert micro["f1"] == pytest.approx(accuracy(y_true, y_pred)) == pytest.approx(4 / 6)


def test_bad_average_and_length_mismatch():
    with pytest.raises(ValueError):
        precision_recall_f1(["a"], ["a"], average="samples")
    with pytest.raises(ValueError):
        accuracy(["a", "b"], ["a"])


# ---------------------------------------------------------------------------
# Ranked retrieval, hand-computed
# ---------------------------------------------------------------------------

RANKED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d1", "d3", "d6"}          # d6 is never retrieved


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, "d3") == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, "d1") == 1.0
    assert reciprocal_rank(RANKED, "zzz") == 0.0


def test_precision_and_recall_at_k():
    assert precision_at_k(RANKED, RELEVANT, 1) == 1.0
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, RELEVANT, 10) == pytest.approx(2 / 10)   # divides by k
    assert recall_at_k(RANKED, RELEVANT, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_average_precision():
    # hits at rank 1 (P=1/1) and rank 3 (P=2/3); d6 missing contributes 0 -> (1 + 2/3 + 0) / 3
    assert average_precision(RANKED, RELEVANT) == pytest.approx(5 / 9)
    assert average_precision(RANKED, set()) == 0.0


def test_pairwise_prf():
    gold = [{"bahut", "bohot", "bht"}, {"acha", "accha"}]
    pred = [{"bahut", "bohot"}, {"bht", "acha", "accha"}, {"xyz", "abc"}]
    # gold pairs: bahut-bohot, bahut-bht, bohot-bht, acha-accha (4)
    # pred pairs (gold items only): bahut-bohot, bht-acha, bht-accha, acha-accha (4); xyz/abc ignored
    out = pairwise_prf(pred, gold)
    assert out["tp"] == 2
    assert out["n_gold_pairs"] == 4 and out["n_pred_pairs"] == 4
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.5)


def test_pairwise_prf_singletons_give_zero():
    out = pairwise_prf([{"a"}, {"b"}], [{"a", "b"}])
    assert out == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0,
                   "n_pred_pairs": 0, "n_gold_pairs": 1}


# ---------------------------------------------------------------------------
# Cross-checks against scikit-learn on random inputs (sklearn used ONLY here)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_classification_metrics_match_sklearn(seed):
    skm = pytest.importorskip("sklearn.metrics")
    rng = random.Random(42 + seed)
    labels = ["negative", "neutral", "positive"]
    n = rng.randint(1, 300)
    y_true = [rng.choice(labels) for _ in range(n)]
    # sometimes drop a class from the predictions to exercise zero-division paths
    pred_labels = labels if seed % 3 else labels[:2]
    y_pred = [rng.choice(pred_labels) for _ in range(n)]

    for avg in ("macro", "weighted", "micro"):
        ours = precision_recall_f1(y_true, y_pred, avg)
        p, r, f, _ = skm.precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        assert ours["precision"] == pytest.approx(p)
        assert ours["recall"] == pytest.approx(r)
        assert ours["f1"] == pytest.approx(f)

    assert accuracy(y_true, y_pred) == pytest.approx(skm.accuracy_score(y_true, y_pred))
    assert confusion_matrix(y_true, y_pred, labels) == skm.confusion_matrix(y_true, y_pred, labels=labels).tolist()


@pytest.mark.parametrize("seed", range(20))
def test_average_precision_matches_sklearn_on_full_ranking(seed):
    # When every document is ranked (no ties), sklearn's AP equals the L6 definition.
    skm = pytest.importorskip("sklearn.metrics")
    rng = random.Random(42 + seed)
    n = rng.randint(2, 100)
    ranked = [f"d{i}" for i in range(n)]
    rng.shuffle(ranked)
    relevant = set(rng.sample(ranked, rng.randint(1, n)))
    y_true = [1 if d in relevant else 0 for d in ranked]
    scores = [n - i for i in range(n)]         # higher score = earlier rank
    assert average_precision(ranked, relevant) == pytest.approx(skm.average_precision_score(y_true, scores))
