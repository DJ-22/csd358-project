"""Normalisation front door for HinglishKey.

The classifiers, the index and every experiment go through these functions
only, so their signatures must stay fixed:

    MODES = ("raw", "soundex", "hkey", "hkey_clusters")
    tokenize(text)                 -> list[str]
    normalize_tokens(tokens, mode) -> list[str]
    fit_variants(train_texts)      -> None

Each mode is an L2 equivalence classing: tokens that map to the same string
become one term for the index and the classifiers.
"""

from hinglishkey.hkey import hkey
from hinglishkey.tokenize import tokenize as _tokenize_text
from hinglishkey.variants import VariantClusters

MODES = ("raw", "soundex", "hkey", "hkey_clusters")

# Vowel policy of mode "hkey", chosen on the E1 dev groups by strict pairwise
# F1 (V-norm 0.373 vs V-drop 0.218; results/e1_variants.json). Frozen.
HKEY_OPTS = {"vowels": "norm"}

# Learned by fit_variants(); used by mode "hkey_clusters".
_variants: VariantClusters | None = None


def tokenize(text: str) -> list[str]:
    """Turn a raw tweet into a list of lowercase tokens (L2 tokenisation).

    See tokenize.py for the steps.
    """
    return _tokenize_text(text)


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

    raw           -> tokens unchanged
    soundex       -> lecture-2 Soundex code of each token
    hkey          -> H-Key of each token (HKEY_OPTS: V-norm, see hkey.py)
    hkey_clusters -> canonical spelling of the token's variant class
                     (needs fit_variants() first)
    """
    if mode == "raw":
        return list(tokens)
    if mode == "soundex":
        return [soundex(t) for t in tokens]
    if mode == "hkey":
        return [hkey(t, **HKEY_OPTS) for t in tokens]
    if mode == "hkey_clusters":
        if _variants is None:
            raise RuntimeError('call fit_variants(train_texts) before mode="hkey_clusters"')
        return [_variants.canonical(t) for t in tokens]
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def fit_variants(train_texts: list[str]) -> None:
    """Learns clusters from TRAIN only; required before mode="hkey_clusters".

    Uses the settings tuned on E1 dev: V-drop key, theta = variants.DEFAULT_THETA.
    """
    global _variants
    _variants = VariantClusters().fit(train_texts)
