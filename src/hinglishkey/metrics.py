"""Evaluation metrics, implemented from scratch.

Lecture 6 (Evaluation): precision, recall, F1 as the harmonic mean, accuracy,
the confusion (contingency) matrix, and ranked-retrieval measures P@k, R@k,
reciprocal rank and average precision (MAP = mean of AP over queries).

Shared by all experiments (E1-E3), so function signatures must stay fixed.
"""

from __future__ import annotations

from itertools import combinations


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    """num / den, or 0.0 when den == 0 (same convention as scikit-learn's zero_division=0)."""
    return num / den if den else 0.0


def _f1(p: float, r: float) -> float:
    """L6: F1 = 2PR / (P + R), the harmonic mean of P and R.

    The harmonic mean is close to the smaller of P and R, so a system cannot
    score well by maximising only one of them (e.g. return everything -> R = 1).
    """
    return _safe_div(2 * p * r, p + r)


def _check_lengths(y_true, y_pred) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} items but y_pred has {len(y_pred)}")


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def precision_recall_f1(y_true, y_pred, average="macro") -> dict:
    """Precision, recall and F1 for a multi-class prediction (L6).

    For each class c, treating c as "relevant" and every other class as "not":
        P_c = tp_c / (tp_c + fp_c)      R_c = tp_c / (tp_c + fn_c)      F1_c = 2 P_c R_c / (P_c + R_c)

    average:
        "macro"    -> unweighted mean of the per-class P, R and F1 (every class counts equally).
                      Note macro-F1 is the mean of the per-class F1s, not F1(macro-P, macro-R).
        "weighted" -> mean of the per-class values weighted by class support (number of true items).
        "micro"    -> pool tp/fp/fn over all classes first; for single-label data this equals accuracy.

    Classes are the sorted union of labels seen in y_true and y_pred.
    Returns {"precision", "recall", "f1", "per_class": {c: {"precision","recall","f1","support"}}}.
    """
    _check_lengths(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))

    # Count tp, fp, fn per class in one pass.
    tp = {c: 0 for c in labels}
    fp = {c: 0 for c in labels}
    fn = {c: 0 for c in labels}
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1   # predicted p, but it was not p
            fn[t] += 1   # it was t, but we missed it

    per_class = {}
    for c in labels:
        p_c = _safe_div(tp[c], tp[c] + fp[c])
        r_c = _safe_div(tp[c], tp[c] + fn[c])
        per_class[c] = {"precision": p_c, "recall": r_c, "f1": _f1(p_c, r_c),
                        "support": tp[c] + fn[c]}

    if average == "macro":
        n = len(labels)
        precision = _safe_div(sum(per_class[c]["precision"] for c in labels), n)
        recall = _safe_div(sum(per_class[c]["recall"] for c in labels), n)
        f1 = _safe_div(sum(per_class[c]["f1"] for c in labels), n)
    elif average == "weighted":
        total = sum(per_class[c]["support"] for c in labels)
        precision = _safe_div(sum(per_class[c]["precision"] * per_class[c]["support"] for c in labels), total)
        recall = _safe_div(sum(per_class[c]["recall"] * per_class[c]["support"] for c in labels), total)
        f1 = _safe_div(sum(per_class[c]["f1"] * per_class[c]["support"] for c in labels), total)
    elif average == "micro":
        TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
        precision = _safe_div(TP, TP + FP)
        recall = _safe_div(TP, TP + FN)
        f1 = _f1(precision, recall)
    else:
        raise ValueError(f"average must be 'macro', 'weighted' or 'micro', got {average!r}")

    return {"precision": precision, "recall": recall, "f1": f1, "per_class": per_class}


def accuracy(y_true, y_pred) -> float:
    """L6: accuracy = fraction of items classified correctly = (tp + tn) / N.

    Misleading on imbalanced data: if 99.9% of documents are non-relevant,
    a system that always says "non-relevant" gets 99.9% accuracy and is useless.
    """
    _check_lengths(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return _safe_div(correct, len(y_true))


def confusion_matrix(y_true, y_pred, labels) -> list[list[int]]:
    """L6 contingency table generalised to many classes.

    matrix[i][j] = number of items whose TRUE class is labels[i] and whose
    PREDICTED class is labels[j]. The diagonal holds the correct predictions.
    (Same orientation as scikit-learn: rows = truth, columns = prediction.)
    """
    _check_lengths(y_true, y_pred)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix


# ---------------------------------------------------------------------------
# Ranked-retrieval metrics (E3)
# ---------------------------------------------------------------------------

def reciprocal_rank(ranked_ids, relevant_id) -> float:
    """1 / (rank of the relevant item), with ranks starting at 1; 0 if it was not retrieved.

    Averaged over queries this gives MRR, the standard measure for known-item search.
    """
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == relevant_id:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_ids, relevant_ids, k) -> float:
    """L6: P@k = (# relevant items in the top k) / k.

    Divides by k even if fewer than k items were returned (a short list is penalised).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    hits = sum(1 for doc_id in list(ranked_ids)[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(ranked_ids, relevant_ids, k) -> float:
    """L6: R@k = (# relevant items in the top k) / (total # relevant items)."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    hits = sum(1 for doc_id in list(ranked_ids)[:k] if doc_id in relevant)
    return _safe_div(hits, len(relevant))


def average_precision(ranked_ids, relevant_ids) -> float:
    """L6: AP = mean, over ALL relevant items, of the precision at the rank where each is found.

    A relevant item that is never retrieved contributes precision 0.
    MAP = mean of AP over the query set (each query counts equally).
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank      # precision at this rank
    return precision_sum / len(relevant)


# ---------------------------------------------------------------------------
# Clustering / grouping quality (E1)
# ---------------------------------------------------------------------------

def _pairs(groups, keep) -> set[frozenset]:
    """All unordered pairs {a, b} of items that share a group, restricted to items in `keep`."""
    pairs = set()
    for group in groups:
        members = sorted(set(group) & keep)
        for a, b in combinations(members, 2):
            pairs.add(frozenset((a, b)))
    return pairs


def pairwise_prf(pred_groups, gold_groups) -> dict:
    """Pairwise precision / recall / F1 of a grouping against a gold grouping (E1).

    Each grouping is an iterable of groups; each group is an iterable of items (e.g. spellings).
    A "pair" is two items placed in the same group. Only items that appear in the gold
    groups are scored, so a system may group the whole vocabulary while gold covers ~300 groups.

        P = |pred pairs ∩ gold pairs| / |pred pairs|
        R = |pred pairs ∩ gold pairs| / |gold pairs|
        F1 = 2PR / (P + R)

    Returns {"precision", "recall", "f1", "tp", "n_pred_pairs", "n_gold_pairs"}.
    """
    gold_groups = [list(g) for g in gold_groups]
    gold_items = set(item for g in gold_groups for item in g)

    gold_pairs = _pairs(gold_groups, gold_items)
    pred_pairs = _pairs(pred_groups, gold_items)
    tp = len(pred_pairs & gold_pairs)

    p = _safe_div(tp, len(pred_pairs))
    r = _safe_div(tp, len(gold_pairs))
    return {"precision": p, "recall": r, "f1": _f1(p, r), "tp": tp,
            "n_pred_pairs": len(pred_pairs), "n_gold_pairs": len(gold_pairs)}
