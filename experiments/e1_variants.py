"""E1: does H-Key group true spelling variants?

    python experiments/e1_variants.py candidates     # writes data/gold/candidates.tsv
    python experiments/e1_variants.py run            # E1 -> results/e1_variants.json, figures/e1_theta.png

Gold candidates. Each candidate group = one headword + up to 6 candidate
spellings from the TRAIN vocabulary. Annotators label every candidate against
its headword as SAME / INFLECTION / DIFFERENT (see data/gold/LABELLING_GUIDE.md
and data/README.md for who labelled and how).

To keep the gold from being circular, candidates come from four sources,
only two of which are H-Key: edit distance <= 2 (independent of any
phonetic key), lecture Soundex, H-Key (V-drop) and the V-norm similarity
form. The file does not say which source proposed a candidate. Every word
appears in at most one group, so words in different groups can be scored
as non-variants.

candidates.tsv holds short tweet contexts, so it is not committed; this
command regenerates it byte for byte (seed 42).

E1 run. Gold groups (strict: headword + SAME candidates; lenient: + INFLECTION;
every other labelled word is a singleton) are split 50/50 by group, seed 42,
into dev and test. Systems group the gold words; pairwise P/R/F1 (L14
pair counting, metrics.pairwise_prf) compares them with the gold.
On dev only: the vowel policy of plain H-Key, and the (vowel policy, theta)
of H-Key + clusters, are chosen by strict F1. Every system is then scored
once on test with the chosen settings.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglishkey import data                          # noqa: E402
from hinglishkey.hkey import hkey                      # noqa: E402
from hinglishkey.metrics import pairwise_prf           # noqa: E402
from hinglishkey.normalize import soundex              # noqa: E402
from hinglishkey.tokenize import tokenize              # noqa: E402
from hinglishkey.variants import VariantClusters, similarity_form   # noqa: E402

SEED = 42
CANDIDATES_TSV = ROOT / "data" / "gold" / "candidates.tsv"
GOLD_TSV = ROOT / "data" / "gold" / "variants_gold.tsv"
RESULTS_JSON = ROOT / "results" / "e1_variants.json"
THETA_FIG = ROOT / "figures" / "e1_theta.png"
THETA_GRID = [round(0.20 + 0.05 * i, 2) for i in range(15)]      # 0.20 .. 0.90
VOWEL_POLICIES = ("drop", "norm")
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


# ---------------------------------------------------------------------------
# E1 evaluation
# ---------------------------------------------------------------------------

def load_gold(path: Path) -> dict[str, list[dict]]:
    """group_id -> its rows; every row must carry a final label."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    empty = [r for r in rows if r["label"] not in ("SAME", "INFLECTION", "DIFFERENT")]
    if empty:
        sys.exit(f"{path}: {len(empty)} rows without a final label (adjudicate them first)")
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    return dict(groups)


def split_groups(group_ids: list[str]) -> tuple[list[str], list[str]]:
    """Seeded 50/50 split of the gold groups into dev and test."""
    ids = sorted(group_ids)
    random.Random(SEED).shuffle(ids)
    half = len(ids) // 2
    return sorted(ids[:half]), sorted(ids[half:])


def gold_groups(groups: dict[str, list[dict]], ids: list[str], lenient: bool) -> list[list[str]]:
    """Headword + variant candidates as one group; every other candidate alone."""
    variant = {"SAME", "INFLECTION"} if lenient else {"SAME"}
    out = []
    for gid in ids:
        rows = groups[gid]
        out.append([rows[0]["headword"]] + [r["candidate"] for r in rows if r["label"] in variant])
        out += [[r["candidate"]] for r in rows if r["label"] not in variant]
    return out


def group_by(words, key) -> list[list[str]]:
    """Partition words by a key function (a system's equivalence classes)."""
    out = defaultdict(list)
    for w in words:
        out[key(w)].append(w)
    return list(out.values())


def score(pred_groups, groups, ids) -> dict:
    """Strict and lenient pairwise P/R/F1, plus where the false positives sit."""
    group_of = {}
    for gid in ids:
        group_of[groups[gid][0]["headword"]] = gid
        for r in groups[gid]:
            group_of[r["candidate"]] = gid
    strict = pairwise_prf(pred_groups, gold_groups(groups, ids, lenient=False))
    lenient = pairwise_prf(pred_groups, gold_groups(groups, ids, lenient=True))
    # predicted pairs that join words from two different gold groups (assumed non-variants)
    cross = 0
    for g in pred_groups:
        members = [w for w in g if w in group_of]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                cross += group_of[members[i]] != group_of[members[j]]
    strict["fp_across_groups"] = cross
    strict["fp_within_groups"] = strict["n_pred_pairs"] - strict["tp"] - cross
    return {"strict": strict, "lenient": lenient}


def words_of(groups, ids) -> list[str]:
    ws = []
    for gid in ids:
        ws.append(groups[gid][0]["headword"])
        ws += [r["candidate"] for r in groups[gid]]
    return ws


def run_e1(gold_path: Path, results_path: Path, fig_path: Path) -> dict:
    t_start = time.perf_counter()
    groups = load_gold(gold_path)
    dev_ids, test_ids = split_groups(list(groups))
    train_texts = [text for _, text, _ in data.load_split("train")]

    # Cluster models fitted on TRAIN for every (vowel policy, theta) on the grid.
    models = {(v, th): VariantClusters(theta=th, key_opts={"vowels": v}).fit(train_texts)
              for v in VOWEL_POLICIES for th in THETA_GRID}

    def systems_for(ids, cluster_choice=None, kmeans_model=None):
        ws = words_of(groups, ids)
        out = {
            "exact": group_by(ws, lambda w: w),
            "soundex": group_by(ws, soundex),
            "hkey_vdrop": group_by(ws, lambda w: hkey(w, vowels="drop")),
            "hkey_vnorm": group_by(ws, lambda w: hkey(w, vowels="norm")),
        }
        if cluster_choice is not None:
            out["hkey_clusters"] = group_by(ws, models[cluster_choice].canonical)
        if kmeans_model is not None:
            out["hkey_kmeans_ablation"] = group_by(ws, kmeans_model.canonical)
        return out

    # ---- dev: tune -------------------------------------------------------
    dev = {name: score(g, groups, dev_ids) for name, g in systems_for(dev_ids).items()}
    dev_words = words_of(groups, dev_ids)
    grid = {v: [] for v in VOWEL_POLICIES}
    for v in VOWEL_POLICIES:
        for th in THETA_GRID:
            s = score(group_by(dev_words, models[(v, th)].canonical), groups, dev_ids)
            grid[v].append({"theta": th, **{m: s["strict"][m] for m in ("precision", "recall", "f1")},
                            "lenient_f1": s["lenient"]["f1"]})
    hkey_vowels = max(VOWEL_POLICIES, key=lambda v: (dev[f"hkey_v{v}"]["strict"]["f1"], v == "drop"))
    best_v, best_th = max(((v, row["theta"]) for v in VOWEL_POLICIES for row in grid[v]),
                          key=lambda vt: (next(r["f1"] for r in grid[vt[0]] if r["theta"] == vt[1]), vt[1]))
    drop_th = max(grid["drop"], key=lambda r: (r["f1"], r["theta"]))["theta"]
    kmeans_model = VariantClusters(theta=drop_th, key_opts={"vowels": "drop"}, method="kmeans").fit(train_texts)
    dev["hkey_clusters"] = score(group_by(dev_words, models[(best_v, best_th)].canonical), groups, dev_ids)
    dev["hkey_kmeans_ablation"] = score(group_by(dev_words, kmeans_model.canonical), groups, dev_ids)

    # ---- test: scored once with the chosen settings -------------------------
    test = {name: score(g, groups, test_ids)
            for name, g in systems_for(test_ids, (best_v, best_th), kmeans_model).items()}

    def n_pairs(ids, lenient):
        return sum(len(g) * (len(g) - 1) // 2 for g in gold_groups(groups, ids, lenient))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {
        "question": "Does H-Key group true spelling variants?",
        "gold": {"file": gold_path.resolve().relative_to(ROOT).as_posix()
                 if gold_path.resolve().is_relative_to(ROOT) else str(gold_path),
                 "n_groups": len(groups),
                 "n_rows": sum(len(r) for r in groups.values()),
                 "labels": dict(Counter(r["label"] for rs in groups.values() for r in rs)),
                 "annotation": "both annotators and the adjudicator were language models; see data/README.md",
                 "split": {"seed": SEED, "dev_groups": len(dev_ids), "test_groups": len(test_ids),
                           "dev_gold_pairs_strict": n_pairs(dev_ids, False),
                           "dev_gold_pairs_lenient": n_pairs(dev_ids, True),
                           "test_gold_pairs_strict": n_pairs(test_ids, False),
                           "test_gold_pairs_lenient": n_pairs(test_ids, True)}},
        "metric": "pairwise precision/recall/F1 over the gold words of one split (metrics.pairwise_prf); "
                  "strict = only SAME pairs are true variants, lenient = SAME or INFLECTION",
        "tuning_on_dev": {
            "criterion": "strict F1 (ties: V-drop for plain H-Key; larger theta for clusters)",
            "hkey_vowel_policy": hkey_vowels,
            "clusters": {"vowels": best_v, "theta": best_th},
            "kmeans_ablation": {"vowels": "drop", "theta_small_blocks": drop_th,
                                "large_block_min_words": 10, "k": "knee of RSS, K = 1..block/2"},
            "theta_grid": THETA_GRID, "grid": grid,
            "note_vnorm": "with the V-norm key, all words in a block share one similarity form, so every "
                          "cosine is 1 and theta has no effect: clusters on a V-norm key equal plain V-norm H-Key",
        },
        "dev": dev,
        "test": test,
        "not_run": {"soundexhindi": "not reproduced (needs the paper's full rule set; not attempted)",
                    "dhvani": "optional dependency, not installed (needs approval)"},
        "fit_on": "train split only", "code_commit": commit,
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    plot_theta(grid, best_v, best_th, fig_path)
    return report


def spread_labels(ys: list[float], gap: float) -> list[float]:
    """Nudge label heights apart so that no two are closer than `gap`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def plot_theta(grid, best_v, best_th, path: Path) -> None:
    """theta sensitivity on dev (V-drop key): strict P, R, F1 against theta.

    Only the V-drop key is drawn: with the V-norm key every word in a block
    has the same similarity form, so every cosine is 1 and theta has no
    effect (the V-norm grid is flat; it is kept in the JSON).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, rule = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
    series = [("precision", "Precision", "#2a78d6", "o"),
              ("recall", "Recall", "#eb6834", "s"),
              ("f1", "F1", "#1baf7a", "^")]
    rows = grid["drop"]
    xs = [r["theta"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.0), facecolor=surface)
    ax.set_facecolor(surface)
    ends = []
    for key, name, colour, marker in series:
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, color=colour, linewidth=2, marker=marker, markersize=4.5,
                markeredgecolor=surface, markeredgewidth=0.8, label=name)
        ends.append(ys[-1])
    for (key, name, _, _), y in zip(series, spread_labels(ends, 0.05)):
        ax.annotate(name, (xs[-1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=ink)
    best = max(rows, key=lambda r: (r["f1"], r["theta"]))
    ax.axvline(best["theta"], color=muted, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    tag = "selected, " if best_v == "drop" else ""
    right_half = best["theta"] > (xs[0] + xs[-1]) / 2
    ax.annotate(f"{tag}best dev F1 at θ = {best['theta']:.2f}", (best["theta"], 0.97),
                xytext=(-4 if right_half else 4, 0), textcoords="offset points", fontsize=7,
                color=ink, va="top", ha="right" if right_half else "left")
    ax.set_xlabel("θ (cosine threshold on the similarity form)", fontsize=8, color=ink)
    ax.set_ylabel("Dev pairwise score (strict)", fontsize=8, color=ink)
    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.13)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=rule, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=ink, labelsize=7.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7.5,
              frameon=False, labelcolor=ink, handlelength=2.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates", help="write data/gold/candidates.tsv")
    c.add_argument("--out", type=Path, default=CANDIDATES_TSV)
    c.add_argument("--force", action="store_true")
    r = sub.add_parser("run", help="run E1 on the adjudicated gold")
    r.add_argument("--gold", type=Path, default=GOLD_TSV)
    r.add_argument("--results", type=Path, default=RESULTS_JSON)
    r.add_argument("--figure", type=Path, default=THETA_FIG)
    gs = sub.add_parser("gold-stats", help="describe the gold file only (no system is scored)")
    gs.add_argument("--gold", type=Path, default=GOLD_TSV)
    gs.add_argument("--out", type=Path, default=ROOT / "results" / "e1_gold_stats.json")
    pl = sub.add_parser("plot", help="redraw figures/e1_theta.png from results/e1_variants.json (no rescoring)")
    pl.add_argument("--results", type=Path, default=RESULTS_JSON)
    pl.add_argument("--figure", type=Path, default=THETA_FIG)
    args = ap.parse_args()

    if args.cmd == "gold-stats":
        groups = load_gold(args.gold)
        rows = [r for rs in groups.values() for r in rs]
        disputed = [r for r in rows if r["label_a"] != r["label_b"]]
        sizes = Counter(len(rs) for rs in groups.values())
        stats = {
            "file": args.gold.resolve().relative_to(ROOT).as_posix(),
            "n_groups": len(groups), "n_rows": len(rows),
            "candidates_per_group": {str(k): v for k, v in sorted(sizes.items())},
            "labels": dict(Counter(r["label"] for r in rows)),
            "labels_annotator_a": dict(Counter(r["label_a"] for r in rows)),
            "labels_annotator_b": dict(Counter(r["label_b"] for r in rows)),
            "n_disputed": len(disputed),
            "adjudicated_as_a": sum(r["label"] == r["label_a"] for r in disputed),
            "adjudicated_as_b": sum(r["label"] == r["label_b"] for r in disputed),
            "adjudicated_as_neither": sum(r["label"] not in (r["label_a"], r["label_b"]) for r in disputed),
            "groups_with_a_same_candidate": sum(any(r["label"] == "SAME" for r in rs) for rs in groups.values()),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(json.dumps(stats, indent=2))
        return
    if args.cmd == "plot":
        with open(args.results, encoding="utf-8") as f:
            tune = json.load(f)["tuning_on_dev"]
        plot_theta(tune["grid"], tune["clusters"]["vowels"], tune["clusters"]["theta"], args.figure)
        print(f"redrew {args.figure} from {args.results}")
        return

    if args.cmd == "candidates":
        rows = make_candidates()
        write_candidates(rows, args.out, args.force)
        n_groups = len({r["group_id"] for r in rows})
        print(f"wrote {args.out}: {n_groups} groups, {len(rows)} candidate rows")
    elif args.cmd == "run":
        rep = run_e1(args.gold, args.results, args.figure)
        tune = rep["tuning_on_dev"]
        print(f"dev choices: plain H-Key V-{tune['hkey_vowel_policy']}; clusters V-{tune['clusters']['vowels']} "
              f"theta={tune['clusters']['theta']}  ({rep['seconds']} s)")
        print(f"{'system':22s} {'strict P':>9s} {'R':>7s} {'F1':>7s}   {'lenient P':>9s} {'R':>7s} {'F1':>7s}")
        for name, s in rep["test"].items():
            st, le = s["strict"], s["lenient"]
            print(f"{name:22s} {st['precision']:9.3f} {st['recall']:7.3f} {st['f1']:7.3f}   "
                  f"{le['precision']:9.3f} {le['recall']:7.3f} {le['f1']:7.3f}")
        print(f"wrote {args.results} and {args.figure}")


if __name__ == "__main__":
    main()
