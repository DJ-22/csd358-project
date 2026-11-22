"""Tests for src/hinglishkey/normalize.py."""

import inspect

import pytest

from hinglishkey import normalize
from hinglishkey.hkey import hkey
from hinglishkey.variants import VariantClusters
from hinglishkey.normalize import (
    MODES,
    fit_variants,
    normalize_tokens,
    soundex,
    tokenize,
)


# --- Interface (signatures must stay fixed) --------------------------------

def test_modes_tuple_is_frozen():
    assert MODES == ("raw", "soundex", "hkey", "hkey_clusters")


def test_signatures_match_contract():
    assert list(inspect.signature(tokenize).parameters) == ["text"]
    assert list(inspect.signature(normalize_tokens).parameters) == ["tokens", "mode"]
    assert list(inspect.signature(fit_variants).parameters) == ["train_texts"]


def test_tokenize_uses_the_tokenizer_module():
    assert tokenize("@ rahul Achhaaaa https // t co / x1 #Jeet") == ["achhaa", "jeet"]


# --- Soundex: lecture 2 examples -------------------------------------------

def test_soundex_lecture_herman():
    assert soundex("HERMAN") == "H655"
    assert soundex("herman") == "H655"


def test_soundex_lecture_hermann():
    assert soundex("HERMANN") == "H655"


def test_soundex_step4_collapses_consecutive_identical_digits():
    # hermann: E R M A N N -> 0 6 5 0 5 5 -> 0 6 5 0 5 -> H655
    assert soundex("hermann") == soundex("herman")


def test_soundex_first_letter_not_coded_lecture_version():
    # Lecture: first letter retained, rest coded. F after P is NOT merged.
    assert soundex("pfister") == "P123"


def test_soundex_pads_with_zeros():
    assert soundex("a") == "A000"
    assert soundex("lee") == "L000"


def test_soundex_truncates_to_four():
    assert soundex("bhagwanpur") == "B251"


def test_soundex_groups_hinglish_variants():
    # The motivating case: bahut/bohot/bhot share a code.
    assert soundex("bahut") == soundex("bohot") == soundex("bhot") == "B300"


def test_soundex_non_alpha_token_unchanged():
    assert soundex("2020") == "2020"
    assert soundex("") == ""


# --- normalize_tokens -------------------------------------------------------

def test_raw_mode_is_identity_and_copies():
    toks = ["bahut", "accha"]
    out = normalize_tokens(toks, "raw")
    assert out == toks
    assert out is not toks


def test_soundex_mode():
    assert normalize_tokens(["herman", "hermann"], "soundex") == ["H655", "H655"]


def test_hkey_mode_uses_frozen_vnorm_policy():
    assert normalize.HKEY_OPTS == {"vowels": "norm"}
    toks = ["theek", "thik", "bahut", "2020"]
    assert normalize_tokens(toks, "hkey") == [hkey(t, vowels="norm") for t in toks] \
        == ["tik", "tik", "bahut", "2020"]


def test_ablation_mode_hkey_vdrop():
    assert normalize.ABLATION_MODES == ("hkey_vdrop",)
    assert "hkey_vdrop" not in MODES                      # the frozen tuple is unchanged
    toks = ["bahut", "bohot", "nahi", "nhi", "2020"]
    assert normalize_tokens(toks, "hkey_vdrop") == [hkey(t, vowels="drop") for t in toks] \
        == ["bht", "bht", "nh", "nh", "2020"]


def test_cluster_defaults_are_the_dev_tuned_ones():
    from hinglishkey import variants
    assert variants.DEFAULT_THETA == 0.85
    assert variants.DEFAULT_KEY_OPTS == {"vowels": "drop"}
    m = VariantClusters()
    assert m.theta == 0.85 and m.key_opts == {"vowels": "drop"}


def test_hkey_clusters_needs_fit(monkeypatch):
    monkeypatch.setattr(normalize, "_variants", None)
    with pytest.raises(RuntimeError):
        normalize_tokens(["bahut"], "hkey_clusters")


def test_hkey_clusters_after_fit(monkeypatch):
    # Merge behaviour itself is tested in test_variants.py; this checks wiring.
    monkeypatch.setattr(normalize, "_variants", None)
    texts = ["bahut accha", "bahut acha", "bahuut badiya", "kal milte"]
    fit_variants(texts)
    toks = ["bahuut", "acha", "kal", "2020"]
    expected = [VariantClusters().fit(texts).canonical(t) for t in toks]
    assert normalize_tokens(toks, "hkey_clusters") == expected


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        normalize_tokens(["x"], "stemming")
