"""Tests for src/hinglishkey/tokenize.py."""

import pytest

from hinglishkey.tokenize import squash_runs, tokenize


def test_lowercases():
    assert tokenize("Bahut ACCHA") == ["bahut", "accha"]


def test_strips_urls_and_mentions():
    text = "@rahul dekho https://t.co/abc123 www.example.com mast"
    assert tokenize(text) == ["dekho", "mast"]


def test_strips_pretokenised_url_from_dataset():
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
def test_strips_other_dataset_url_forms(url):
    assert tokenize(f"dekho {url} yaar") == ["dekho", "yaar"]


def test_url_strip_does_not_eat_following_words():
    assert tokenize("https://t.co/abc . bahut accha") == ["bahut", "accha"]


def test_strips_split_mention_from_dataset():
    assert tokenize("@ rahul_01 sahi bola") == ["sahi", "bola"]


def test_strips_split_mention_with_underscores():
    assert tokenize("@ nitin _ gadkari ji sahi") == ["ji", "sahi"]
    assert tokenize("@ bts _ army _ fin dekho") == ["dekho"]


def test_keeps_hashtag_text():
    assert tokenize("#IndVsPak jeet gaye") == ["indvspak", "jeet", "gaye"]
    assert tokenize("# IndVsPak jeet") == ["indvspak", "jeet"]    # split form


def test_drops_emojis_and_punctuation():
    assert tokenize("kya baat hai!!! \U0001F602\U0001F525") == ["kya", "baat", "hai"]


def test_keeps_english_and_devanagari_tokens():
    # the vowel sign U+093E is a combining mark and must stay inside the word
    assert tokenize("match भाई great") == ["match", "भाई", "great"]


def test_emoji_variation_selector_is_a_separator():
    assert tokenize("❤️ pyaar") == ["pyaar"]


def test_squashes_long_letter_runs():
    assert tokenize("achhaaaa yaaaar") == ["achhaa", "yaar"]


def test_does_not_squash_digits():
    assert tokenize("1000 runs") == ["1000", "runs"]


def test_keeps_apostrophe_words_whole():
    assert tokenize("don't worry") == ["dont", "worry"]


def test_squash_is_separately_ablatable():
    assert squash_runs("achhaaaa") == "achhaa"
    assert squash_runs("acha") == "acha"
    assert tokenize("achhaaaa", squash=False) == ["achhaaaa"]


# --- Alignment of display words to token positions ---------------------------

from hinglishkey.tokenize import align_tokens


def test_align_tokens_maps_words_to_positions():
    text = "@ rahul Bahut accha!!! https // t co / x1 yaar"
    assert tokenize(text) == ["bahut", "accha", "yaar"]
    assert align_tokens(text) == [("@", []), ("rahul", []), ("Bahut", [0]), ("accha!!!", [1]),
                                  ("https", []), ("//", []), ("t", []), ("co", []), ("/", []),
                                  ("x1", []), ("yaar", [2])]


def test_align_tokens_word_with_two_tokens():
    assert align_tokens("match-fixing hai") == [("match-fixing", [0, 1]), ("hai", [2])]
