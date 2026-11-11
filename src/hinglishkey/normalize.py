"""Normalisation front door for HinglishKey.

The classifiers, the index and every experiment go through these functions
only, so their signatures must stay fixed:

    MODES = ("raw", "soundex", "hkey", "hkey_clusters")
    tokenize(text)                 -> list[str]
    normalize_tokens(tokens, mode) -> list[str]
    fit_variants(train_texts)      -> None

Implemented so far: "raw" and "soundex". "hkey" and "hkey_clusters" are
not implemented yet.
"""

import re

MODES = ("raw", "soundex", "hkey", "hkey_clusters")


# ---------------------------------------------------------------------------
# Tokenisation (L2 term vocabulary: tokenisation + case folding)
# ---------------------------------------------------------------------------

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
# A "word" is a run of letters or digits in any script; everything else
# (punctuation, emojis, symbols, whitespace) is a separator.
_WORD_RE = re.compile(r"[^\W_]+")
# 3 or more copies of the same letter (not digit) in a row.
_LONG_RUN_RE = re.compile(r"([^\W\d_])\1{2,}")


def squash_runs(token: str) -> str:
    """Squash runs of 3+ identical letters to 2 ("achhaaaa" -> "achhaa").

    L2 normalisation. Kept as its own function so it can be ablated.
    """
    return _LONG_RUN_RE.sub(r"\1\1", token)


def _tokenize(text: str, squash: bool) -> list[str]:
    """L2 tokenisation.

    1. lowercase (case folding)
    2. strip URLs and @mentions (joined or already split by spaces)
    3. drop the '#' of hashtags but keep their text (falls out of step 4)
    4. split into runs of letters/digits; emojis and punctuation vanish
       (apostrophes are deleted first so "don't" stays one token: "dont")
    5. optionally squash 3+ repeated letters to 2
    """
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("'", "").replace("’", "")
    tokens = _WORD_RE.findall(text)
    if squash:
        tokens = [squash_runs(t) for t in tokens]
    return tokens


def tokenize(text: str) -> list[str]:
    """Turn a raw tweet into a list of lowercase tokens (L2 tokenisation)."""
    return _tokenize(text, squash=True)


# ---------------------------------------------------------------------------
# Soundex, exactly as on the lecture 2 slide (IIR section 3.4)
# ---------------------------------------------------------------------------

_SOUNDEX_DIGIT = {}
for _letters, _digit in (
    ("aeiouhwy", "0"),
    ("bfpv", "1"),
    ("cgjkqsxz", "2"),
    ("dt", "3"),
    ("l", "4"),
    ("mn", "5"),
    ("r", "6"),
):
    for _ch in _letters:
        _SOUNDEX_DIGIT[_ch] = _digit


def soundex(token: str) -> str:
    """L2 Soundex, following the lecture's five steps literally.

    1. Retain the first letter.
    2. A, E, I, O, U, H, W, Y -> 0.
    3. B,F,P,V->1; C,G,J,K,Q,S,X,Z->2; D,T->3; L->4; M,N->5; R->6.
    4. Repeatedly remove one of each pair of consecutive identical digits.
    5. Remove all zeros; pad with trailing zeros; return first 4 positions.

    The first letter is kept as a letter and is not itself coded (lecture
    version), so e.g. "pfister" -> "P123" (American Soundex would give P236).
    Characters other than a-z after the first letter are ignored.
    A token that does not start with a-z (a number, Devanagari, ...) has no
    Soundex code and is returned unchanged.

    >>> soundex("herman"), soundex("hermann")
    ('H655', 'H655')
    """
    token = token.lower()
    if not token or token[0] not in _SOUNDEX_DIGIT:
        return token

    # Steps 1-3: first letter, then one digit per remaining a-z letter.
    first = token[0].upper()
    digits = [_SOUNDEX_DIGIT[ch] for ch in token[1:] if ch in _SOUNDEX_DIGIT]

    # Step 4: collapse each run of identical consecutive digits to one digit.
    collapsed = []
    for d in digits:
        if not collapsed or collapsed[-1] != d:
            collapsed.append(d)

    # Step 5: drop zeros, pad, truncate to letter + 3 digits.
    code = "".join(d for d in collapsed if d != "0")
    return (first + code + "000")[:4]


# ---------------------------------------------------------------------------
# Public normalisation API
# ---------------------------------------------------------------------------

def normalize_tokens(tokens: list[str], mode: str) -> list[str]:
    """Map each token to its index term under the chosen normalisation mode.

    L2 equivalence classing: tokens that map to the same string are treated
    as the same term by the index and the classifiers.

    raw           -> tokens unchanged
    soundex       -> lecture-2 Soundex code of each token
    hkey          -> H-Key code (not implemented yet)
    hkey_clusters -> canonical spelling of the token's variant class
                     (not implemented yet)
    """
    if mode == "raw":
        return list(tokens)
    if mode == "soundex":
        return [soundex(t) for t in tokens]
    if mode in ("hkey", "hkey_clusters"):
        raise NotImplementedError(
            f"mode={mode!r} is not implemented yet; use 'raw' or 'soundex' for now"
        )
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def fit_variants(train_texts: list[str]) -> None:
    """Learns clusters from TRAIN only; required before mode="hkey_clusters"."""
    raise NotImplementedError("fit_variants is not implemented yet")
