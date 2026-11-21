"""Tests for src/hinglishkey/index.py."""

import random

import pytest

from hinglishkey.index import PositionalIndex, build_index, intersect, positional_intersect


def make_index(docs):
    idx = PositionalIndex("raw")
    for i, text in enumerate(docs):
        idx.add(f"t{i}", text.split())
    return idx


DOCS = [
    "bahut accha laga yaar",          # 0
    "kya bahut bura hua",             # 1
    "accha bahut accha",              # 2
    "yaar bahut accha movie thi",     # 3
]


# --- Postings structure -------------------------------------------------------

def test_postings_hold_sorted_docs_and_positions():
    idx = make_index(DOCS)
    assert idx.postings["accha"] == [(0, [1]), (2, [0, 2]), (3, [2])]
    assert idx.postings["bahut"] == [(0, [0]), (1, [1]), (2, [1]), (3, [1])]
    assert idx.doc_ids == ["t0", "t1", "t2", "t3"]
    assert idx.df("accha") == 3 and idx.df("missing") == 0


def test_size_counts():
    idx = make_index(DOCS)
    assert idx.n_postings() == sum(len(set(d.split())) for d in DOCS)
    assert idx.n_positions() == sum(len(d.split()) for d in DOCS)


# --- Intersection and Boolean AND (L1) ----------------------------------------

def test_intersect_lecture_example():
    # IIR Fig. 1.5/1.6: Brutus AND Calpurnia
    brutus = [1, 2, 4, 11, 31, 45, 173, 174]
    calpurnia = [2, 31, 54, 101]
    assert intersect(brutus, calpurnia) == [2, 31]


def test_and_processes_terms_in_increasing_df():
    idx = make_index(DOCS)
    # df: movie 1, yaar 2, accha 3, bahut 4
    assert idx.and_order(["bahut", "accha", "yaar", "movie"]) == ["movie", "yaar", "accha", "bahut"]


def test_boolean_and():
    idx = make_index(DOCS)
    assert idx.boolean_and(["bahut", "accha"]) == [0, 2, 3]
    assert idx.boolean_and(["bahut", "accha", "yaar"]) == [0, 3]
    assert idx.boolean_and(["bahut", "missing"]) == []
    assert idx.boolean_and([]) == []


def test_boolean_and_matches_brute_force():
    rng = random.Random(42)
    vocab = [f"w{i}" for i in range(15)]
    docs = [" ".join(rng.choices(vocab, k=rng.randint(1, 12))) for _ in range(200)]
    idx = make_index(docs)
    for _ in range(100):
        q = rng.sample(vocab, rng.randint(1, 4))
        expected = [d for d, text in enumerate(docs) if set(q) <= set(text.split())]
        assert idx.boolean_and(q) == expected


# --- Phrase queries (L2) --------------------------------------------------------

def test_phrase_finds_adjacent_terms_only():
    idx = make_index(DOCS)
    assert idx.phrase(["bahut", "accha"]) == [(0, [0]), (2, [1]), (3, [1])]
    assert idx.phrase(["accha", "bahut"]) == [(2, [0])]
    assert idx.phrase(["bahut", "laga"]) == []          # both present, not adjacent


def test_phrase_three_terms_and_single_term():
    idx = make_index(DOCS)
    assert idx.phrase(["yaar", "bahut", "accha"]) == [(3, [0])]
    assert idx.phrase(["accha"]) == idx.postings["accha"]
    assert idx.phrase([]) == []


def test_phrase_with_repeated_term():
    idx = make_index(["haha haha haha", "haha yaar haha"])
    assert idx.phrase(["haha", "haha"]) == [(0, [0, 1])]


def test_positional_intersect_offset():
    p1 = [(1, [0, 5, 9]), (3, [2])]
    p2 = [(1, [1, 7, 11]), (2, [0]), (3, [4])]
    assert positional_intersect(p1, p2, 1) == [(1, [0])]
    assert positional_intersect(p1, p2, 2) == [(1, [5, 9]), (3, [2])]


def test_phrase_matches_brute_force():
    rng = random.Random(42)
    vocab = ["a", "b", "c", "d"]
    docs = [" ".join(rng.choices(vocab, k=rng.randint(1, 10))) for _ in range(150)]
    idx = make_index(docs)
    for _ in range(100):
        q = rng.choices(vocab, k=rng.randint(1, 3))
        expected = []
        for d, text in enumerate(docs):
            toks = text.split()
            starts = [s for s in range(len(toks) - len(q) + 1) if toks[s:s + len(q)] == q]
            if starts:
                expected.append((d, starts))
        assert idx.phrase(q) == expected


# --- Normalised index: spelling-robust phrase search ------------------------------

def test_phrase_is_spelling_robust_with_hkey():
    # mode "hkey" is V-norm: theek/thik -> tik, hai/he -> he
    docs = [("t1", "Movie thik he yaar"), ("t2", "he nahi theek")]
    raw = build_index(docs, "raw")
    hk = build_index(docs, "hkey")
    assert raw.phrase(raw.terms_of("theek hai")) == []
    assert hk.phrase(hk.terms_of("theek hai")) == [(0, [1])]


def test_hkey_vnorm_does_not_join_vowel_quality_variants():
    # Known limitation of the V-norm key chosen on E1 dev: bahut/bohot differ in
    # vowel quality (a vs o), so they get different keys. V-drop would join them.
    hk = build_index([("t1", "bohot acha")], "hkey")
    assert hk.phrase(hk.terms_of("bahut accha")) == []


def test_build_index_uses_tokenizer():
    idx = build_index([("t1", "@ rahul Bahut ACCHA https // t co / x1")], "raw")
    assert set(idx.postings) == {"bahut", "accha"}


# --- Persistence -------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    idx = build_index([("t1", "bahut accha"), ("t2", "kal milte")], "soundex")
    path = tmp_path / "index.pkl"
    idx.save(path)
    loaded = PositionalIndex.load(path)
    assert loaded.mode == "soundex"
    assert loaded.postings == idx.postings and loaded.doc_ids == idx.doc_ids
    assert loaded.phrase(loaded.terms_of("bahut accha")) == [(0, [0])]
