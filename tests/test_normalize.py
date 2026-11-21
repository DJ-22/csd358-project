"""Tests for src/hinglishkey/normalize.py (raw + soundex modes)."""

import inspect

import pytest

from hinglishkey import normalize
from hinglishkey.normalize import (
    MODES,
    fit_variants,
    normalize_tokens,
    soundex,
    squash_runs,
    tokenize,
)


# --- Interface (signatures must stay fixed) --------------------------------

def test_modes_tuple_is_frozen():
    assert MODES == ("raw", "soundex", "hkey", "hkey_clusters")


def test_signatures_match_contract():
    assert list(inspect.signature(tokenize).parameters) == ["text"]
    assert list(inspect.signature(normalize_tokens).parameters) == ["tokens", "mode"]
    assert list(inspect.signature(fit_variants).parameters) == ["train_texts"]


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


# --- Tokenisation -----------------------------------------------------------

def test_tokenize_lowercases():
    assert tokenize("Bahut ACCHA") == ["bahut", "accha"]


def test_tokenize_strips_urls_and_mentions():
    text = "@rahul dekho https://t.co/abc123 www.example.com mast"
    assert tokenize(text) == ["dekho", "mast"]


def test_tokenize_strips_pretokenised_url_from_dataset():
    # SentiMix files split URLs into space-separated pieces.
    text = "kya match tha https // t . co / hBg7zS0viy bhai"
    assert tokenize(text) == ["kya", "match", "tha", "bhai"]


@pytest.mark.parametrize("url", [
    "https // t co / hBg7zS0viy",      # no dot between t and co
    "https // tco / hBg7zS0viy",
    "https // t . c",                  # tweet truncated mid-URL
    "https // t",
    "https",
])
def test_tokenize_strips_other_dataset_url_forms(url):
    assert tokenize(f"dekho {url} yaar") == ["dekho", "yaar"]


def test_tokenize_strips_split_mention_from_dataset():
    assert tokenize("@ rahul_01 sahi bola") == ["sahi", "bola"]


def test_tokenize_strips_split_mention_with_underscores():
    assert tokenize("@ nitin _ gadkari ji sahi") == ["ji", "sahi"]
    assert tokenize("@ bts _ army _ fin dekho") == ["dekho"]


def test_tokenize_url_strip_does_not_eat_following_words():
    assert tokenize("https://t.co/abc . bahut accha") == ["bahut", "accha"]


def test_tokenize_keeps_hashtag_text():
    assert tokenize("#IndVsPak jeet gaye") == ["indvspak", "jeet", "gaye"]


def test_tokenize_drops_emojis_and_punctuation():
    assert tokenize("kya baat hai!!! \U0001F602\U0001F525") == ["kya", "baat", "hai"]


def test_tokenize_squashes_long_letter_runs():
    assert tokenize("achhaaaa yaaaar") == ["achhaa", "yaar"]


def test_tokenize_does_not_squash_digits():
    assert tokenize("1000 runs") == ["1000", "runs"]


def test_tokenize_keeps_apostrophe_words_whole():
    assert tokenize("don't worry") == ["dont", "worry"]


def test_squash_runs_is_separately_ablatable():
    assert squash_runs("achhaaaa") == "achhaa"
    assert squash_runs("acha") == "acha"
    assert normalize._tokenize("achhaaaa", squash=False) == ["achhaaaa"]


# --- normalize_tokens -------------------------------------------------------

def test_raw_mode_is_identity_and_copies():
    toks = ["bahut", "accha"]
    out = normalize_tokens(toks, "raw")
    assert out == toks
    assert out is not toks


def test_soundex_mode():
    assert normalize_tokens(["herman", "hermann"], "soundex") == ["H655", "H655"]


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        normalize_tokens(["x"], "stemming")


@pytest.mark.parametrize("mode", ["hkey", "hkey_clusters"])
def test_later_modes_not_yet_implemented(mode):
    with pytest.raises(NotImplementedError):
        normalize_tokens(["x"], mode)
