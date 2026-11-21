"""Tests for src/hinglishkey/variants.py."""

import math

import pytest

from hinglishkey.hkey import hkey
from hinglishkey.variants import (
    UnionFind,
    VariantClusters,
    char_trigrams,
    cosine,
    similarity_form,
    tfidf_vector,
    trigram_idf,
)


# --- Char-trigram tf-idf ----------------------------------------------------

def test_char_trigrams_padded():
    assert char_trigrams("acha") == {"$ac": 1, "ach": 1, "cha": 1, "ha$": 1}


def test_char_trigrams_counts_repeats():
    assert char_trigrams("hahaha")["hah"] == 2


def test_trigram_idf_is_log10_n_over_df():
    idf, unseen = trigram_idf(["acha", "accha", "bahut", "bohot"])
    assert idf["$ac"] == pytest.approx(math.log10(4 / 2))
    assert idf["hut"] == pytest.approx(math.log10(4 / 1))
    assert unseen == pytest.approx(math.log10(4))


def test_tfidf_vector_is_unit_length():
    idf, unseen = trigram_idf(["acha", "accha", "bahut"])
    v = tfidf_vector("accha", idf, unseen)
    assert math.sqrt(sum(x * x for x in v.values())) == pytest.approx(1.0)


def test_cosine_identical_and_disjoint():
    idf, unseen = trigram_idf(["acha", "bahut"])
    a = tfidf_vector("acha", idf, unseen)
    b = tfidf_vector("bahut", idf, unseen)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == 0.0


# --- Union-find --------------------------------------------------------------

def test_union_find_is_transitive():
    uf = UnionFind(["bahut", "bahot", "bohot", "acha"])
    uf.union("bahut", "bahot")
    uf.union("bahot", "bohot")
    assert uf.find("bahut") == uf.find("bohot")
    assert uf.find("acha") != uf.find("bahut")


# --- Merge rule: key(a) == key(b) AND cos >= theta ---------------------------

TEXTS = ["bahut bahut bahut bohot bahot", "acha accha", "match matches"]


def test_theta_zero_merges_whole_key_block():
    m = VariantClusters(theta=0.0).fit(TEXTS)
    assert m.canonical("bohot") == m.canonical("bahot") == "bahut"


def test_never_merges_across_key_blocks():
    # match / matches share most trigrams but have different H-Keys.
    assert hkey("match") != hkey("matches")
    m = VariantClusters(theta=0.0).fit(TEXTS)
    assert m.canonical("match") != m.canonical("matches")


def test_theta_above_one_merges_nothing():
    m = VariantClusters(theta=1.01).fit(TEXTS)
    assert all(m.canonical(w) == w for w in ["bahut", "bohot", "bahot", "acha", "accha"])


def test_class_id_is_most_frequent_member():
    m = VariantClusters(theta=0.0).fit(TEXTS)
    assert m.classes()["bahut"] == ["bahut", "bahot", "bohot"]   # 3, then 1-1 tie


def test_class_id_tie_goes_to_alphabetically_first():
    m = VariantClusters(theta=0.0).fit(["acha accha"])
    assert m.canonical("acha") == m.canonical("accha") == "accha"


def test_key_options_are_used_for_blocking():
    # Under V-norm, bahut and bohot have different keys, so they never merge.
    m = VariantClusters(theta=0.0, key_opts={"vowels": "norm"}).fit(TEXTS)
    assert m.canonical("bohot") != m.canonical("bahut")


# --- Words unseen in training -----------------------------------------------

def test_unseen_word_joins_nearest_same_key_class():
    m = VariantClusters(theta=0.5).fit(["bahut bahut bohot"])
    # bahuut has the same similarity form as bahut ("bahut"): cos = 1
    assert m.canonical("bahuut") == "bahut"


def test_unseen_word_below_theta_stays_itself():
    m = VariantClusters(theta=0.99).fit(["bahut bahut bohot"])
    # bahuta: same key (bht) but a different form, so cos < 0.99
    assert m.canonical("bahuta") == "bahuta"


# --- Similarity form: trigrams of the V-norm H-Key ---------------------------

def test_similarity_form_examples():
    assert similarity_form("theek") == similarity_form("thik") == "tik"
    assert similarity_form("phir") == similarity_form("fir") == "fir"
    assert similarity_form("doston", {"max_len": 3}) == "doston"   # cap not applied


def test_similarity_form_merges_what_raw_trigrams_miss():
    texts = ["phir fir bahut", "thik theek kal"]
    raw_idf, raw_unseen = trigram_idf(["phir", "fir", "bahut", "thik", "theek", "kal"])
    raw_cos = cosine(tfidf_vector("phir", raw_idf, raw_unseen),
                     tfidf_vector("fir", raw_idf, raw_unseen))
    assert raw_cos < 0.5                     # raw spellings share only "ir$"
    m = VariantClusters(theta=0.5).fit(texts)
    assert m.canonical("fir") == m.canonical("phir")
    assert m.canonical("theek") == m.canonical("thik")


def test_similarity_form_follows_rule_switches():
    # with de-aspiration off, ph is no longer f in the form either
    assert similarity_form("phir", {"deaspirate": False}) == "phir"


@pytest.mark.parametrize("tok", ["2020", "भाई"])
def test_tokens_without_key_are_their_own_class(tok):
    m = VariantClusters(theta=0.0).fit(TEXTS)
    assert m.canonical(tok) == tok


def test_fit_sees_only_given_texts():
    m = VariantClusters(theta=0.0).fit(["acha"])
    assert set(m.canon) == {"acha"}
    assert m.freq["bahut"] == 0


# --- Persistence -------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    m = VariantClusters(theta=0.3, key_opts={"vowels": "norm"}).fit(TEXTS)
    path = tmp_path / "variants.json"
    m.save(path)
    m2 = VariantClusters.load(path)
    assert m2.theta == 0.3 and m2.key_opts == {"vowels": "norm"}
    assert m2.canon == m.canon
    assert m2.canonical("bahuut") == m.canonical("bahuut")


# --- k-means ablation (L14) ------------------------------------------------------

import random as _random

import numpy as np

from hinglishkey import variants as _variants
from hinglishkey.variants import kmeans, knee


def test_kmeans_separates_two_clear_clusters():
    X = np.array([[0, 0], [0, 0.1], [0.1, 0], [5, 5], [5, 5.1], [5.1, 5]], dtype=float)
    labels, rss = kmeans(X, 2, _random.Random(42))
    assert labels[0] == labels[1] == labels[2] != labels[3] == labels[4] == labels[5]
    assert rss == pytest.approx(4 * 0.1 ** 2 * 2 / 3, rel=1e-6)


def test_kmeans_rss_zero_when_k_equals_n():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    assert kmeans(X, 3, _random.Random(42))[1] == pytest.approx(0.0)


def test_knee_picks_the_elbow():
    ks = [1, 2, 3, 4, 5, 6]
    rss = [100, 40, 12, 10, 9, 8.5]
    assert knee(ks, rss) == 3


def test_knee_with_too_few_points():
    assert knee([1, 2], [5.0, 1.0]) == 1


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        VariantClusters(method="dbscan")


def test_kmeans_method_used_only_in_large_blocks(monkeypatch):
    monkeypatch.setattr(_variants, "LARGE_BLOCK", 3)
    texts = ["bahut bahut bohot bahot bht", "acha accha"]
    m = VariantClusters(theta=0.0, method="kmeans").fit(texts)
    # block "bht" (4 words) is split by k-means; every word still gets a class
    assert set(m.canon) == {"bahut", "bohot", "bahot", "bht", "acha", "accha"}
    # block "ac" (2 words) is below LARGE_BLOCK, so the theta rule (theta=0) merges it
    assert m.canonical("acha") == m.canonical("accha")


def test_kmeans_method_survives_save_load(tmp_path):
    m = VariantClusters(theta=0.4, method="kmeans").fit(["acha accha"])
    m.save(tmp_path / "v.json")
    assert VariantClusters.load(tmp_path / "v.json").method == "kmeans"
