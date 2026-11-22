"""Tests for classify/demo.py: the explanations shown in the Sentiment tab add up to the
classifiers' own scores, and the model bundle loads as the app expects."""

import pickle

import pytest

from hinglishkey import normalize
from hinglishkey.classify import dot
from hinglishkey.classify.demo import MODELS_FILE, knn_votes, load_models, nb_evidence, rocchio_contributions
from hinglishkey.classify.knn import KNN
from hinglishkey.classify.nb import MultinomialNB
from hinglishkey.classify.rocchio import Rocchio

DOCS = [["accha", "movie"], ["accha", "khush"], ["accha", "gana"], ["bura", "movie"], ["bura", "gussa"]]
LABELS = ["pos", "pos", "pos", "neg", "neg"]


def test_nb_evidence_adds_up_to_the_class_score():
    nb = MultinomialNB().fit(DOCS, LABELS)
    doc = ["accha", "unseen", "movie", "accha"]
    evidence = nb_evidence(nb, doc)
    assert [t for t, _ in evidence] == ["accha", "movie", "accha"]          # unknown token skipped, order kept
    scores = nb.log_scores(doc)
    for c in nb.classes:
        assert nb.log_prior[c] + sum(vals[c] for _, vals in evidence) == pytest.approx(scores[c])


def test_nb_evidence_empty_for_unknown_words():
    assert nb_evidence(MultinomialNB().fit(DOCS, LABELS), ["xyz"]) == []


def test_rocchio_contributions_add_up_to_the_distance():
    model = Rocchio().fit(DOCS, LABELS)
    doc = ["accha", "movie"]
    contrib = rocchio_contributions(model, doc)
    v = model.vectorizer.transform_one(doc)
    distances = model.distances(doc)
    for c in model.classes:
        mu_dot_v = sum(vals[c] for _, vals in contrib)
        assert mu_dot_v == pytest.approx(dot(model.centroids[c], v))
        assert model.sq_norms[c] - 2 * mu_dot_v + 1 == pytest.approx(distances[c])


def test_knn_votes_match_the_prediction():
    model = KNN(k=3).fit(DOCS, LABELS)
    votes, nearest = knn_votes(model, ["accha", "khush"])
    assert sum(votes.values()) == 3 and len(nearest) == 3
    assert max(votes, key=votes.get) == model.predict_one(["accha", "khush"])
    assert knn_votes(model, ["never_seen"]) == ({}, [])                       # zero vector: no evidence


def test_load_models_round_trip_and_variants(tmp_path, monkeypatch):
    bundle = {"models": {("nb", "raw"): MultinomialNB().fit(DOCS, LABELS)}, "k": {"raw": 3},
              "train_ids": ["1"], "variants": "fitted-clusters", "modes": ["raw"]}
    with open(tmp_path / MODELS_FILE, "wb") as f:
        pickle.dump(bundle, f)
    monkeypatch.setattr(normalize, "_variants", None)
    loaded = load_models(tmp_path)
    assert loaded["models"][("nb", "raw")].predict_one(["accha"]) == "pos"
    assert normalize._variants == "fitted-clusters"                            # set only when missing
    assert load_models(tmp_path) is loaded                                     # cached per directory


def test_load_models_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models(tmp_path / "nothing_here")
