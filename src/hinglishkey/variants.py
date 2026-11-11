"""Corpus-learned spelling-variant classes (L2 equivalence classes).

Merge rule for two word types a, b seen in the training corpus:

    key(a) == key(b)  AND  cos(a_3g, b_3g) >= theta

key   = H-Key (hkey.py). Grouping words by key is *blocking*: only pairs
        inside one block are compared, instead of all V^2/2 pairs.
a_3g  = char-trigram tf-idf vector (L2 k-grams, L3 tf-idf) of the word's
        *similarity form*: its H-Key with the V-norm vowel rule and no
        length cap (e.g. theek -> tik, phir -> fir). Taking trigrams of the
        form instead of the raw spelling lets the cosine see what the key
        already knows (ph = f, ee = i); on raw spellings phir and fir share
        only one trigram and never merge.

Merged pairs are closed transitively with union-find, so every word ends up
in exactly one class. The class ID is its most frequent member (the
canonical spelling); ties go to the alphabetically first member.

Fit on the TRAIN split only. theta is tuned on dev gold groups.
"""

import json
import math
from collections import Counter, defaultdict

from hinglishkey.hkey import hkey
from hinglishkey.tokenize import tokenize

DEFAULT_THETA = 0.5     # provisional; replaced by the value tuned on dev


# ---------------------------------------------------------------------------
# Char-trigram tf-idf vectors (L2 k-grams + L3 weighting)
# ---------------------------------------------------------------------------

def char_trigrams(word: str) -> Counter:
    """Trigrams of the word padded with '$' at both ends.

    >>> sorted(char_trigrams("acha"))
    ['$ac', 'ach', 'cha', 'ha$']
    """
    padded = f"${word}$"
    return Counter(padded[i:i + 3] for i in range(len(padded) - 2))


def trigram_idf(words) -> tuple[dict[str, float], float]:
    """idf_g = log10(N / df_g) over the word types (L3).

    Each word type counts as one "document". Returns the idf table and the
    idf given to a trigram never seen in training (df treated as 1).
    """
    words = list(words)
    n = len(words)
    df = Counter()
    for w in words:
        df.update(set(char_trigrams(w)))
    idf = {g: math.log10(n / d) for g, d in df.items()}
    return idf, math.log10(n) if n else 0.0


def tfidf_vector(word: str, idf: dict[str, float], unseen_idf: float) -> dict[str, float]:
    """Unit-length vector with weight (1 + log10 tf) * idf per trigram (L3)."""
    vec = {}
    for g, tf in char_trigrams(word).items():
        vec[g] = (1 + math.log10(tf)) * idf.get(g, unseen_idf)
    norm = math.sqrt(sum(x * x for x in vec.values()))
    if norm == 0:
        return {}
    return {g: x / norm for g, x in vec.items()}


def cosine(u: dict[str, float], v: dict[str, float]) -> float:
    """Cosine of two unit-length sparse vectors = their dot product (L3)."""
    if len(u) > len(v):
        u, v = v, u
    return sum(x * v.get(g, 0.0) for g, x in u.items())


# ---------------------------------------------------------------------------
# Union-find, to close merged pairs into classes
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint sets with path halving; find/union are near O(1)."""

    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# The variant model
# ---------------------------------------------------------------------------

def _has_key(token: str) -> bool:
    return token.isascii() and token.isalpha()


def similarity_form(word: str, key_opts: dict | None = None) -> str:
    """The string whose trigrams are compared: H-Key with V-norm, no cap.

    Uses the same rule switches as the blocking key, so ablations that turn
    a rule off turn it off in both places.
    """
    opts = {**(key_opts or {}), "vowels": "norm", "max_len": None}
    return hkey(word, **opts)


class VariantClusters:
    """Spelling-variant classes learned from a training corpus.

    theta    : cosine threshold of the merge rule
    key_opts : keyword options passed to hkey() (e.g. {"vowels": "norm"})
    """

    def __init__(self, theta: float = DEFAULT_THETA, key_opts: dict | None = None):
        self.theta = theta
        self.key_opts = dict(key_opts or {})
        self.freq: Counter = Counter()
        self.canon: dict[str, str] = {}

    # -- fitting -------------------------------------------------------------

    def fit(self, train_texts) -> "VariantClusters":
        """Learn classes from TRAIN texts only."""
        self.freq = Counter(t for text in train_texts for t in tokenize(text))
        self._build_index()

        words = list(self._vectors)
        uf = UnionFind(words)
        # Blocking: compare only words that share an H-Key.
        for members in self._blocks.values():
            for i in range(len(members)):
                vi = self._vectors[members[i]]
                for j in range(i + 1, len(members)):
                    if cosine(vi, self._vectors[members[j]]) >= self.theta:
                        uf.union(members[i], members[j])

        classes = defaultdict(list)
        for w in words:
            classes[uf.find(w)].append(w)
        self.canon = {}
        for members in classes.values():
            head = min(members, key=lambda w: (-self.freq[w], w))
            for w in members:
                self.canon[w] = head
        return self

    def _build_index(self):
        """Trigram idf, unit vectors and H-Key blocks for the training words.

        idf is computed over the training word types (one "document" each),
        using each word's similarity form.
        """
        words = sorted(w for w in self.freq if _has_key(w))
        forms = {w: similarity_form(w, self.key_opts) for w in words}
        self._idf, self._unseen_idf = trigram_idf(forms.values())
        self._vectors = {w: tfidf_vector(forms[w], self._idf, self._unseen_idf) for w in words}
        self._blocks = defaultdict(list)
        for w in words:
            self._blocks[hkey(w, **self.key_opts)].append(w)

    # -- lookup --------------------------------------------------------------

    def canonical(self, token: str) -> str:
        """Canonical spelling of the token's class.

        A word unseen in training joins the class of its most similar
        same-key training word if that cosine is >= theta; otherwise, and
        for tokens without an H-Key, the token is its own class.
        """
        if token in self.canon:
            return self.canon[token]
        if not _has_key(token):
            return token
        vec = tfidf_vector(similarity_form(token, self.key_opts), self._idf, self._unseen_idf)
        best, best_cos = None, self.theta
        for w in self._blocks.get(hkey(token, **self.key_opts), ()):
            c = cosine(vec, self._vectors[w])
            if c >= best_cos:
                best, best_cos = w, c
        return self.canon[best] if best is not None else token

    def classes(self) -> dict[str, list[str]]:
        """canonical -> members, most frequent first."""
        out = defaultdict(list)
        for w, head in self.canon.items():
            out[head].append(w)
        return {h: sorted(m, key=lambda w: (-self.freq[w], w)) for h, m in out.items()}

    # -- persistence ---------------------------------------------------------

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"theta": self.theta, "key_opts": self.key_opts,
                       "freq": self.freq, "canon": self.canon}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "VariantClusters":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        model = cls(d["theta"], d["key_opts"])
        model.freq = Counter(d["freq"])
        model.canon = d["canon"]
        model._build_index()
        return model


if __name__ == "__main__":
    # Fit on train and print 30 sample multi-member classes for human review.
    import random
    import sys
    import time

    from hinglishkey.data import load_split

    vowels = sys.argv[1] if len(sys.argv) > 1 else "drop"
    t0 = time.perf_counter()
    model = VariantClusters(key_opts={"vowels": vowels}).fit(
        [text for _, text, _ in load_split("train")])
    secs = time.perf_counter() - t0
    classes = model.classes()
    multi = [h for h, m in classes.items() if len(m) > 1]
    sizes = [len(model._blocks[k]) for k in model._blocks]
    print(f"vowels={vowels} theta={model.theta} (provisional)  fit {secs:.1f}s")
    print(f"word types with a key: {len(model.canon)}  key blocks: {len(sizes)}"
          f"  largest block: {max(sizes)}  pairs compared: "
          f"{sum(b * (b - 1) // 2 for b in sizes)}")
    print(f"classes: {len(classes)}  with 2+ members: {len(multi)}\n")
    rng = random.Random(42)
    for h in sorted(rng.sample(multi, min(30, len(multi)))):
        print(f"{h:>14s} [{hkey(h, **model.key_opts)}]  "
              + ", ".join(f"{w}({model.freq[w]})" for w in classes[h]))
