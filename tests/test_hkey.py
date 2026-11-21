"""Tests for src/hinglishkey/hkey.py: one test per rule, each on spelling
pairs that occur in the SentiMix train split.

Where possible a test also switches its rule OFF and checks the pair then
differs, which shows that this rule (not another one) does the merging.
"""

import pytest

from hinglishkey.hkey import (
    deaspirate_digraphs,
    drop_vowels,
    harden_lone_c,
    hkey,
    merge_letters,
    normalize_vowels,
    squash_consonants,
)


# --- Rule 1: keep the first letter -----------------------------------------

def test_rule1_keep_first_separates_aam_from_mai():
    # aam (mango) / mai (I): dropping the initial vowel would merge them.
    assert hkey("aam") == "am"
    assert hkey("mai") == "m"
    assert hkey("aam", keep_first=False) == hkey("mai", keep_first=False) == "m"


def test_rule1_aap_ap_still_match():
    assert hkey("aap") == hkey("ap") == "ap"


# --- Rule 2: de-aspirate digraphs -------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("thik", "tik"),        # th -> t
    ("phir", "fir"),        # ph -> f
    ("jhooth", "jhoot"),    # th -> t at the end
    ("theek", "teek"),      # th -> t at the start
])
def test_rule2_deaspirate_pairs(a, b):
    assert hkey(a) == hkey(b)
    assert hkey(a, deaspirate=False) != hkey(b, deaspirate=False)


def test_rule3_other_letters():
    assert merge_letters("wzqx") == "vjkks"


def test_rule2_all_nine_digraphs():
    assert deaspirate_digraphs("kh gh ch jh th dh ph bh sh") == "k g c j t d f b s"


def test_rule2_repeats_until_no_digraph_left():
    # achha has a doubled aspirate: achha -> acha -> aca
    assert deaspirate_digraphs("achha") == deaspirate_digraphs("acha") == "aca"
    assert hkey("achha") == hkey("acha") == hkey("accha") == "ac"


# --- Rule 3: merge equivalent letters ---------------------------------------

@pytest.mark.parametrize("a, b", [
    ("wala", "vala"),        # w -> v
    ("zindagi", "jindagi"),  # z -> j
    ("qismat", "kismat"),    # q -> k
    ("vipax", "vipaks"),     # x -> ks
    ("thanx", "thanks"),     # x -> ks
    ("congress", "kongress"),  # lone c -> k
])
def test_rule3_merge_pairs(a, b):
    assert hkey(a) == hkey(b)
    assert hkey(a, merge=False) != hkey(b, merge=False)


def test_rule3_lone_c_only():
    assert harden_lone_c("cool") == "kool"
    assert harden_lone_c("chal") == "chal"      # c of ch is not lone
    assert harden_lone_c("accha") == "accha"    # nor the first c of cch
    assert harden_lone_c("occasion") == "okkasion"


@pytest.mark.parametrize("ch_word, k_word", [
    ("chal", "kal"),     # walk / tomorrow
    ("chai", "kai"),     # tea / several
    ("chor", "kor"),
])
def test_rule3_keeps_ch_apart_from_k(ch_word, k_word):
    assert hkey(ch_word) != hkey(k_word)


def test_rule3_ch_variants_still_match():
    assert hkey("kuch") == hkey("kuchh") == hkey("kucch")


# --- Rule 4: squash repeated consonants -------------------------------------

@pytest.mark.parametrize("a, b", [
    ("pakka", "paka"),
    ("kutta", "kuta"),
    ("accha", "acha"),
])
def test_rule4_squash_pairs(a, b):
    assert hkey(a) == hkey(b)
    assert hkey(a, squash=False) != hkey(b, squash=False)


def test_rule4_leaves_vowel_runs_for_rule5():
    assert squash_consonants("pyaar") == "pyaar"
    assert squash_consonants("kutta") == "kuta"


# --- Rule 5a: V-drop ---------------------------------------------------------

def test_rule5_vdrop_bahut_bohot_bahot():
    assert hkey("bahut") == hkey("bohot") == hkey("bahot") == "bht"
    # V-norm keeps a/o apart, so these differ under V-norm
    assert hkey("bahut", vowels="norm") != hkey("bohot", vowels="norm")


def test_rule5_vdrop_keeps_only_initial_vowel():
    assert drop_vowels("achha") == "achh"
    assert drop_vowels("achha", keep_first=False) == "chh"
    assert drop_vowels("aao", keep_first=False) == "a"   # never empty


# --- Rule 5b: V-norm ---------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("pyaar", "pyar"),     # aa -> a
    ("theek", "thik"),     # ee -> i
    ("hoon", "hun"),       # oo -> u
    ("hai", "hae"),        # ai, ae -> e
    ("aur", "our"),        # au, ou -> o
])
def test_rule5_vnorm_pairs(a, b):
    assert hkey(a, vowels="norm") == hkey(b, vowels="norm")


def test_rule5_vnorm_mapping():
    assert normalize_vowels("aa ee ii oo uu ai ae au ou") == "a i i u u e e o o"


def test_rule5_unknown_vowel_policy_rejected():
    with pytest.raises(ValueError):
        hkey("bahut", vowels="keep")


# --- Rule 6: optional length cap ---------------------------------------------

def test_rule6_length_cap_merges_inflections():
    # doston / dosto (friends, oblique / vocative plural)
    assert hkey("doston") != hkey("dosto")
    assert hkey("doston", max_len=3) == hkey("dosto", max_len=3) == "dst"


def test_rule6_no_cap_by_default():
    assert hkey("dhanyavad") == "dnyvd"


# --- Tokens without an H-Key -------------------------------------------------

@pytest.mark.parametrize("tok", ["2020", "article370", "भाई", ""])
def test_non_alpha_tokens_unchanged(tok):
    assert hkey(tok) == tok
