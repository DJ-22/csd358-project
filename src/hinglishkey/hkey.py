"""H-Key: a Soundex-style phonetic key for romanised Hindi (extends L2 Soundex).

Soundex maps letters to 6 digit classes, built for English surnames. Hinglish
spelling varies along other lines: aspiration is written or not (thik/theek,
kharab), letters are swapped for the same sound (wala/vala, zindagi/jindagi),
letters are doubled or not (accha/acha), and vowels are written freely
(bahut/bohot/bht). H-Key keeps letters (not digits) and applies one named
rule per kind of variation:

    1. keep_first  : the first letter is never dropped by the vowel rule
    2. deaspirate  : kh->k gh->g ch->c jh->j th->t dh->d ph->f bh->b sh->s
    3. merge       : w->v z->j q->k x->ks, and a lone c->k
                     ("lone" = not part of ch/cch; this part runs BEFORE
                     rule 2, so ch stays c and chal/kal stay apart)
    4. squash      : collapse runs of the same consonant (kk->k)
    5. vowels      : "drop" = remove non-initial vowels (Soundex style)
                     "norm" = aa->a ee,ii->i oo,uu->u ai,ae->e au,ou->o
    6. max_len     : optional length cap (None = no cap, the default)

Every rule can be switched off with its keyword, for the ablation study.
Tokens that are not purely a-z (numbers, Devanagari, "article370") have no
H-Key and are returned unchanged.
"""

import re

VOWELS = frozenset("aeiou")

# Rule 2: consonant + 'h' -> unaspirated consonant.
_DEASPIRATE = {"k": "k", "g": "g", "c": "c", "j": "j", "t": "t",
               "d": "d", "p": "f", "b": "b", "s": "s"}

# Rule 3: letters that stand for the same sound in romanised Hindi.
_MERGE = {"w": "v", "z": "j", "q": "k", "x": "ks"}
# Rule 3, lone c: a c that is not followed by "h" or "ch".
_LONE_C_RE = re.compile(r"c(?!c?h)")

# Rule 5 (V-norm): two-letter vowel spellings -> one canonical vowel.
_VOWEL_NORM = {"aa": "a", "ee": "i", "ii": "i", "oo": "u", "uu": "u",
               "ai": "e", "ae": "e", "au": "o", "ou": "o"}

_ALPHA_RE = re.compile(r"[a-z]+")


def deaspirate_digraphs(s: str) -> str:
    """Rule 2: replace each aspirated digraph by its plain consonant.

    Repeated until no digraph is left, so the doubled aspirate in
    "achha" goes achha -> acha -> aca, the same as "acha" -> aca.
    """
    while True:
        out = []
        i = 0
        while i < len(s):
            if i + 1 < len(s) and s[i] in _DEASPIRATE and s[i + 1] == "h":
                out.append(_DEASPIRATE[s[i]])
                i += 2
            else:
                out.append(s[i])
                i += 1
        new = "".join(out)
        if new == s:
            return s
        s = new


def harden_lone_c(s: str) -> str:
    """Rule 3, first part: a lone c sounds like k ("congress" -> "kongress").

    Runs before rule 2. A c in "ch" or "cch" is left alone, so after rule 2
    it stays c (cha/chal) and never meets k (kal): chal -> cal, kal -> kal.
    """
    return _LONE_C_RE.sub("k", s)


def merge_letters(s: str) -> str:
    """Rule 3, second part: w->v, z->j, q->k, x->ks (one letter at a time)."""
    return "".join(_MERGE.get(ch, ch) for ch in s)


def squash_consonants(s: str) -> str:
    """Rule 4: collapse each run of the same consonant to one ("pakka" -> "paka").

    Vowel runs are left alone here because rule 5 (V-norm) needs to see
    "ee" and "oo", which mean different vowels from "e" and "o".
    """
    out = []
    for ch in s:
        if out and out[-1] == ch and ch not in VOWELS:
            continue
        out.append(ch)
    return "".join(out)


def drop_vowels(s: str, keep_first: bool = True) -> str:
    """Rule 5, V-drop: remove every vowel except (if keep_first) the first letter.

    As in Soundex, a word made only of vowels keeps its first letter even
    with keep_first=False, so no key is ever empty.
    """
    if not s:
        return s
    start = 1 if keep_first else 0
    rest = "".join(ch for ch in s[start:] if ch not in VOWELS)
    key = s[:start] + rest
    return key if key else s[0]


def normalize_vowels(s: str) -> str:
    """Rule 5, V-norm: map two-letter vowel spellings to one vowel, left to right."""
    out = []
    i = 0
    while i < len(s):
        pair = s[i:i + 2]
        if pair in _VOWEL_NORM:
            out.append(_VOWEL_NORM[pair])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def hkey(token: str, *, keep_first: bool = True, deaspirate: bool = True,
         merge: bool = True, squash: bool = True, vowels: str = "drop",
         max_len: int | None = None) -> str:
    """Compute the H-Key of one lowercase token (rules 1-6 above).

    Order: lone c->k, rule 2, rest of rule 3, rule 4, rule 5, rule 6.

    >>> hkey("bahut"), hkey("bohot"), hkey("theek", vowels="norm")
    ('bht', 'bht', 'tik')
    """
    if vowels not in ("drop", "norm"):
        raise ValueError(f"vowels must be 'drop' or 'norm', got {vowels!r}")
    if not _ALPHA_RE.fullmatch(token):
        return token

    s = token
    if merge:
        s = harden_lone_c(s)
    if deaspirate:
        s = deaspirate_digraphs(s)
    if merge:
        s = merge_letters(s)
    if squash:
        s = squash_consonants(s)
    if vowels == "drop":
        s = drop_vowels(s, keep_first)
    else:
        s = normalize_vowels(s)
    if max_len is not None:
        s = s[:max_len]
    return s
