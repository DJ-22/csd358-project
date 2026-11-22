"""Tokenisation of Hinglish tweets (L2 term vocabulary).

Steps, in order:
    1. lowercase (case folding)
    2. strip URLs and @mentions, in both normal and space-split form
    3. drop the '#' of hashtags but keep their text
    4. split into words: a word starts with a letter or digit (any script)
       and continues with letters, digits and combining marks, so
       Devanagari vowel signs stay inside their word ("भाई" is one token).
       Emojis and punctuation are separators. Apostrophes are deleted
       first so "don't" stays one token: "dont".
    5. squash runs of 3+ identical letters to 2 ("achhaaaa" -> "achhaa")

Step 5 can be switched off (squash=False) for the ablation.
No language identification is done: English tokens are kept as they are.
"""

import re
import unicodedata

# URLs, in normal form ("https://t.co/abc") and in the pre-tokenised forms
# found in the SentiMix files: "https // t . co / abc", "https // t co / abc",
# "https // tco / abc", and tweets cut off mid-URL ("https // t . c ...").
# Parts: scheme and optional colon; then, only if the "//" follows, a host
# and any "/segment" parts. A bare "https" never swallows the next word.
_URL_RE = re.compile(
    r"https?\s*:?"
    r"(?:\s*/\s*/\s*"
    r"(?:t(?:\s*\.?\s*co?)?(?!\w)"                # Twitter's t.co shortener
    r"|[\w-]+(?:\s*\.\s*[\w-]+)*)?"               # any other host
    r"(?:\s*/\s*[\w.?=&%-]*)*)?"
    r"|www\.\S+"
)
# @mentions, joined ("@rahul") or split by the data's tokeniser ("@ rahul"),
# whose underscores are split off too ("@ nitin _ gadkari").
_MENTION_RE = re.compile(r"@\s*\w+(?:\s*_+\s*\w*)*")
# 3 or more copies of the same letter (not digit) in a row.
_LONG_RUN_RE = re.compile(r"([^\W\d_])\1{2,}")


def squash_runs(token: str) -> str:
    """Squash runs of 3+ identical letters to 2 ("achhaaaa" -> "achhaa").

    L2 normalisation. Digits are never squashed ("1000" stays "1000").
    """
    return _LONG_RUN_RE.sub(r"\1\1", token)


def split_words(text: str) -> list[str]:
    """Step 4: split text into words (letters/digits plus combining marks).

    Unicode category L = letter, N = number, M = combining mark. A mark only
    continues a word; it never starts one. Variation selectors (U+FE00-FE0F)
    are marks that follow emojis, so they are treated as separators.
    """
    words, current = [], []
    for ch in text:
        cat = unicodedata.category(ch)[0]
        is_mark = cat == "M" and not "︀" <= ch <= "️"
        if cat in "LN" or (is_mark and current):
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def tokenize(text: str, squash: bool = True) -> list[str]:
    """Turn a raw tweet into a list of lowercase tokens (L2 tokenisation)."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("'", "").replace("’", "")
    tokens = split_words(text)
    if squash:
        tokens = [squash_runs(t) for t in tokens]
    return tokens


def align_tokens(text: str, squash: bool = True) -> list[tuple[str, list[int]]]:
    """Pair each space-separated word of `text` with its token positions.

    Positions index into tokenize(text). Words removed as part of a URL or
    mention (or with no letters/digits) get no positions. Used to highlight
    matched words in the demo.
    """
    tokens = tokenize(text, squash)
    out, i = [], 0
    for word in text.split():
        sub = tokenize(word, squash)
        if sub and tokens[i:i + len(sub)] == sub:
            out.append((word, list(range(i, i + len(sub)))))
            i += len(sub)
        else:
            out.append((word, []))
    return out
