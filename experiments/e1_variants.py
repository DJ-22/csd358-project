"""E1: does H-Key group true spelling variants?

    python experiments/e1_variants.py candidates     # writes data/gold/candidates.tsv

Gold candidates. Each candidate group = one headword + up to 6 candidate
spellings from the TRAIN vocabulary. Humans label every candidate against
its headword as SAME / INFLECTION / DIFFERENT (see data/gold/LABELLING_GUIDE.md).

To keep the gold from being circular, candidates come from four sources,
only two of which are H-Key: edit distance <= 2 (independent of any
phonetic key), lecture Soundex, H-Key (V-drop) and the V-norm similarity
form. The file does not say which source proposed a candidate. Every word
appears in at most one group, so words in different groups can be scored
as non-variants.

candidates.tsv holds short tweet contexts, so it is not committed; this
command regenerates it byte for byte (seed 42).
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglishkey import data                          # noqa: E402
from hinglishkey.hkey import hkey                      # noqa: E402
from hinglishkey.normalize import soundex              # noqa: E402
from hinglishkey.tokenize import tokenize              # noqa: E402
from hinglishkey.variants import similarity_form       # noqa: E402

SEED = 42
CANDIDATES_TSV = ROOT / "data" / "gold" / "candidates.tsv"
COLUMNS = ["group_id", "headword", "head_freq", "head_context",
           "candidate", "cand_freq", "cand_context", "label", "note"]

# Headword sampling: (min freq, max freq, Hindi-tagged quota, English-tagged quota)
BANDS = [(50, None, 150, 50), (10, 49, 150, 50), (3, 9, 150, 50)]
PER_SOURCE = 2          # candidates taken from each source first
MAX_CANDIDATES = 6
CONTEXT = 3             # tokens of context on each side


# ---------------------------------------------------------------------------
# Edit distance (Levenshtein, dynamic programming) with a bigram prefilter
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance (insert/delete/substitute = 1), row by row.

    Stops early and returns limit + 1 once every cell in a row exceeds limit.
    """
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def padded_bigrams(word: str) -> set[str]:
    p = f"${word}$"
    return {p[i:i + 2] for i in range(len(p) - 1)}


def edit_neighbours(head, bigram_index, limit):
    """Words within `limit` edits of head.

    One edit destroys at most 2 of a word's padded bigrams, so a word within
    `limit` edits shares at least max(n_head, n_word) - 2*limit bigrams;
    only words passing that count are compared with the full DP.
    """
    shared = Counter()
    hb = padded_bigrams(head)
    for g in hb:
        shared.update(bigram_index[g])
    out = []
    for w, n in shared.items():
        if w == head or n < max(len(hb), len(w) + 1) - 2 * limit:
            continue
        d = edit_distance(head, w, limit)
        if d <= limit:
            out.append((w, d))
    return out


# ---------------------------------------------------------------------------
# Train vocabulary, language tags and contexts
# ---------------------------------------------------------------------------

def train_language_tags() -> dict[str, str]:
    """Majority language tag (Hin/Eng/O) of each token in the TRAIN file."""
    counts = defaultdict(Counter)
    with open(data.RAW_DIR / data.SPLIT_FILES["train"], encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[0] != "meta":
                for t in tokenize(parts[0]):
                    counts[t][parts[1].strip()] += 1
    return {w: c.most_common(1)[0][0] for w, c in counts.items()}


def is_word(w: str) -> bool:
    return w.isascii() and w.isalpha() and len(w) >= 2


def kwic(tokens, pos):
    """Key word in context: up to CONTEXT tokens each side, target in [ ]."""
    left = tokens[max(0, pos - CONTEXT):pos]
    right = tokens[pos + 1:pos + 1 + CONTEXT]
    return " ".join(left + [f"[{tokens[pos]}]"] + right)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def make_candidates():
    rng = random.Random(SEED)
    docs = [tokenize(text) for _, text, _ in data.load_split("train")]
    freq = Counter(t for toks in docs for t in toks)
    words = sorted(w for w in freq if is_word(w))
    lang = train_language_tags()

    bigram_index = defaultdict(list)
    by_soundex, by_hkey, by_form = defaultdict(list), defaultdict(list), defaultdict(list)
    for w in words:
        for g in padded_bigrams(w):
            bigram_index[g].append(w)
        by_soundex[soundex(w)].append(w)
        by_hkey[hkey(w)].append(w)
        by_form[similarity_form(w)].append(w)

    used = set()

    def candidates_for(head):
        limit = 1 if len(head) <= 4 else 2
        dist = dict(edit_neighbours(head, bigram_index, limit))

        def rank(ws):
            ws = [w for w in ws if w != head and w not in used]
            return sorted(ws, key=lambda w: (dist.get(w, 99), -freq[w], w))

        sources = [rank(dist), rank(by_soundex[soundex(head)]),
                   rank(by_hkey[hkey(head)]), rank(by_form[similarity_form(head)])]
        chosen = []
        for src in sources:                               # quota per source
            for w in src[:PER_SOURCE]:
                if w not in chosen:
                    chosen.append(w)
        pool = sorted({w for src in sources for w in src} - set(chosen),
                      key=lambda w: (dist.get(w, 99), -freq[w], w))
        chosen += pool[:max(0, MAX_CANDIDATES - len(chosen))]
        return chosen[:MAX_CANDIDATES]

    # Headwords: per band and language, a seeded shuffle; bands are filled
    # from most to least frequent so frequent words claim their variants first.
    groups = []
    for lo, hi, n_hin, n_eng in BANDS:
        band = [w for w in words if freq[w] >= lo and (hi is None or freq[w] <= hi)]
        for tag, quota in (("Hin", n_hin), ("Eng", n_eng)):
            pool = sorted(w for w in band if lang.get(w) == tag)
            rng.shuffle(pool)
            taken = 0
            for head in pool:
                if taken == quota:
                    break
                if head in used:
                    continue
                cands = candidates_for(head)
                if not cands:
                    continue
                used.add(head)
                used.update(cands)
                rng.shuffle(cands)                        # hide the ranking
                groups.append((head, cands))
                taken += 1

    rng.shuffle(groups)                                   # mix bands and languages

    # One seeded context per word, from the train tweets.
    needed = {w for head, cands in groups for w in [head, *cands]}
    occurrences = defaultdict(list)
    for d, toks in enumerate(docs):
        for i, t in enumerate(toks):
            if t in needed:
                occurrences[t].append((d, i))
    context = {w: kwic(docs[d], i) for w in sorted(needed)
               for d, i in [rng.choice(occurrences[w])]}

    rows = []
    for gid, (head, cands) in enumerate(groups, 1):
        for c in cands:
            rows.append({"group_id": f"g{gid:03d}", "headword": head, "head_freq": freq[head],
                         "head_context": context[head], "candidate": c, "cand_freq": freq[c],
                         "cand_context": context[c], "label": "", "note": ""})
    return rows


def write_candidates(rows, path, force=False):
    if path.exists() and not force:
        sys.exit(f"{path} exists; refusing to overwrite (use --force only if no one has labelled it)")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates", help="write data/gold/candidates.tsv")
    c.add_argument("--out", type=Path, default=CANDIDATES_TSV)
    c.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.cmd == "candidates":
        rows = make_candidates()
        write_candidates(rows, args.out, args.force)
        n_groups = len({r["group_id"] for r in rows})
        print(f"wrote {args.out}: {n_groups} groups, {len(rows)} candidate rows")


if __name__ == "__main__":
    main()
