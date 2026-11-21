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

Ablation (method="kmeans"): inside every key block of at least LARGE_BLOCK
words, the theta rule is replaced by k-means over the same trigram vectors
(L14), with K chosen at the knee of the RSS curve. Smaller blocks still use
the theta rule.
"""

import json
import math
import random
from collections import Counter, defaultdict

import numpy as np

from hinglishkey.hkey import hkey
from hinglishkey.tokenize import tokenize

# Chosen on the E1 dev groups by strict pairwise F1 (results/e1_variants.json):
# V-drop key, theta 0.85. Frozen; used by normalize.fit_variants().
DEFAULT_THETA = 0.85
DEFAULT_KEY_OPTS = {"vowels": "drop"}
LARGE_BLOCK = 10        # k-means ablation: blocks with at least this many words
KMEANS_SEED = 42


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
# k-means and the RSS knee (L14 flat clustering), for the ablation
# ---------------------------------------------------------------------------

def kmeans(X: np.ndarray, k: int, rng: random.Random, n_init: int = 5,
           max_iter: int = 100) -> tuple[list[int], float]:
    """k-means (L14): best of n_init runs, each seeded with k random points.

    Each run alternates (1) assign every vector to its nearest centroid and
    (2) move each centroid to the mean of its vectors, until no assignment
    changes. RSS = sum of squared distances to the assigned centroids.
    Returns the labels and RSS of the run with the lowest RSS.
    """
    n = len(X)
    best_labels, best_rss = None, math.inf
    for _ in range(n_init):
        centroids = X[rng.sample(range(n), k)].copy()
        labels = None
        for _ in range(max_iter):
            # squared distances, n x k
            d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new = d2.argmin(axis=1)
            if labels is not None and (new == labels).all():
                break
            labels = new
            for c in range(k):
                members = X[labels == c]
                if len(members):                       # an empty cluster keeps its centroid
                    centroids[c] = members.mean(axis=0)
        rss = float(((X - centroids[labels]) ** 2).sum())
        if rss < best_rss:
            best_labels, best_rss = labels.tolist(), rss
    return best_labels, best_rss


def knee(ks: list[int], rss: list[float]) -> int:
    """K at the knee of the RSS curve (L14 "knee" heuristic).

    Both axes are scaled to [0, 1]; the knee is the K whose point lies
    furthest below the straight line from the first to the last point.
    """
    if len(ks) < 3:
        return ks[0]
    x = [(k - ks[0]) / (ks[-1] - ks[0]) for k in ks]
    span = (rss[0] - rss[-1]) or 1.0
    y = [(r - rss[-1]) / span for r in rss]
    # the line runs from (0, 1) to (1, 0): y = 1 - x; distance below it ~ (1 - x) - y
    gaps = [(1 - xi) - yi for xi, yi in zip(x, y)]
    return ks[max(range(len(ks)), key=lambda i: (gaps[i], -i))]


def kmeans_partition(words: list[str], vectors: dict[str, dict[str, float]],
                     rng: random.Random) -> list[list[str]]:
    """Split one key block with k-means, K = knee of RSS over K = 1..len//2."""
    dims = sorted({g for w in words for g in vectors[w]})
    col = {g: i for i, g in enumerate(dims)}
    X = np.zeros((len(words), len(dims)))
    for i, w in enumerate(words):
        for g, x in vectors[w].items():
            X[i, col[g]] = x
    ks = list(range(1, max(2, len(words) // 2) + 1))
    runs = [kmeans(X, k, rng) for k in ks]
    best_k = knee(ks, [rss for _, rss in runs])
    labels = runs[ks.index(best_k)][0]
    groups = defaultdict(list)
    for w, c in zip(words, labels):
        groups[c].append(w)
    return list(groups.values())


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

    def __init__(self, theta: float = DEFAULT_THETA, key_opts: dict | None = None,
                 method: str = "theta"):
        if method not in ("theta", "kmeans"):
            raise ValueError(f"method must be 'theta' or 'kmeans', got {method!r}")
        self.theta = theta
        self.key_opts = dict(DEFAULT_KEY_OPTS if key_opts is None else key_opts)
        self.method = method
        self.freq: Counter = Counter()
        self.canon: dict[str, str] = {}

    # -- fitting -------------------------------------------------------------

    def fit(self, train_texts) -> "VariantClusters":
        """Learn classes from TRAIN texts only."""
        self.freq = Counter(t for text in train_texts for t in tokenize(text))
        self._build_index()

        words = list(self._vectors)
        uf = UnionFind(words)
        rng = random.Random(KMEANS_SEED)
        # Blocking: compare only words that share an H-Key.
        for members in self._blocks.values():
            if self.method == "kmeans" and len(members) >= LARGE_BLOCK:
                for group in kmeans_partition(members, self._vectors, rng):
                    for w in group[1:]:
                        uf.union(group[0], w)
                continue
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
            json.dump({"theta": self.theta, "key_opts": self.key_opts, "method": self.method,
                       "freq": self.freq, "canon": self.canon}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "VariantClusters":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        model = cls(d["theta"], d["key_opts"], d.get("method", "theta"))
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
    print(f"vowels={vowels} theta={model.theta} (tuned on E1 dev)  fit {secs:.1f}s")
    print(f"word types with a key: {len(model.canon)}  key blocks: {len(sizes)}"
          f"  largest block: {max(sizes)}  pairs compared: "
          f"{sum(b * (b - 1) // 2 for b in sizes)}")
    print(f"classes: {len(classes)}  with 2+ members: {len(multi)}\n")
    rng = random.Random(42)
    for h in sorted(rng.sample(multi, min(30, len(multi)))):
        print(f"{h:>14s} [{hkey(h, **model.key_opts)}]  "
              + ", ".join(f"{w}({model.freq[w]})" for w in classes[h]))
