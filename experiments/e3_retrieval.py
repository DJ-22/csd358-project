"""E3: does normalisation help find a tweet from a respelled query?

    python experiments/e3_retrieval.py       # -> results/e3_retrieval.json, figures/e3_recall_at_k.png

Known-item retrieval. Each of the 100 queries in data/gold/e3_queries.tsv is a
respelling of words from one target test tweet (rules and provenance in
data/README.md). Every system indexes all 20,000 SentiMix tweets under its
normalisation mode, ranks them by lnc.ltc cosine against the query normalised
the same way, and is scored on where the target tweet lands.

Systems: raw, soundex, hkey (V-norm, main), hkey_clusters (V-drop key,
theta 0.85, fitted on train), hkey_vdrop (ablation). Nothing is tuned here;
the design was fixed before the first run. With one relevant tweet per query,
AP equals the reciprocal rank, so MAP equals MRR.
"""

import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglishkey import data, normalize                                  # noqa: E402
from hinglishkey.index import build_index                                 # noqa: E402
from hinglishkey.metrics import (average_precision, precision_at_k,      # noqa: E402
                                 recall_at_k, reciprocal_rank)
from hinglishkey.rank import Ranker, top_k                                # noqa: E402

QUERIES_TSV = ROOT / "data" / "gold" / "e3_queries.tsv"
QUERIES_SHA256 = "ab5c38097c7bee261947ec66869c321fcd7a78f5e82c371de6346627f39a7327"   # frozen file
RESULTS_JSON = ROOT / "results" / "e3_retrieval.json"
FIGURE = ROOT / "figures" / "e3_recall_at_k.png"
SYSTEMS = ("raw", "soundex", "hkey", "hkey_clusters", "hkey_vdrop")
CURVE_K = list(range(1, 21))
LABELS = {"raw": "raw", "soundex": "Soundex", "hkey": "H-Key (V-norm)",
          "hkey_clusters": "H-Key + clusters", "hkey_vdrop": "H-Key V-drop (ablation)"}
COLOURS = {"raw": ("#2a78d6", "o"), "soundex": ("#eb6834", "s"), "hkey": ("#1baf7a", "^"),
           "hkey_clusters": ("#eda100", "D"), "hkey_vdrop": ("#e87ba4", "v")}


def load_queries() -> list[tuple[str, str]]:
    raw = QUERIES_TSV.read_bytes()
    if hashlib.sha256(raw).hexdigest() != QUERIES_SHA256:
        sys.exit(f"{QUERIES_TSV} differs from the frozen version; E3 refuses to run on changed queries")
    with open(QUERIES_TSV, encoding="utf-8", newline="") as f:
        return [(r["tweet_id"].strip(), r["respelled_query"].strip())
                for r in csv.DictReader(f, delimiter="\t")]


def run() -> dict:
    t_start = time.perf_counter()
    queries = load_queries()
    splits = {s: data.load_split(s) for s in ("train", "dev", "test")}
    docs = [(tid, text) for s in ("train", "dev", "test") for tid, text, _ in splits[s]]
    test_ids = {tid for tid, _, _ in splits["test"]}
    missing = [tid for tid, _ in queries if tid not in test_ids]
    if missing:
        sys.exit(f"query targets not in the test split: {missing}")
    normalize.fit_variants([text for _, text, _ in splits["train"]])

    text_count = Counter(text for _, text in docs)
    target_text = {tid: text for tid, text in docs}
    n_dup_targets = sum(text_count[target_text[tid]] > 1 for tid, _ in queries)

    results, per_query = {}, {}
    for mode in SYSTEMS:
        index = build_index(docs, mode)
        ranker = Ranker(index)
        rows = []
        for tid, q in queries:
            terms = index.terms_of(q)
            ranked = [index.doc_ids[d] for d, _ in top_k(ranker.scores(terms), index.n_docs)]
            rank = ranked.index(tid) + 1 if tid in ranked else None
            rows.append({"tweet_id": tid, "rank": rank,
                         "n_query_terms": len(terms),
                         "n_query_terms_in_index": sum(index.df(t) > 0 for t in set(terms)),
                         "rr": reciprocal_rank(ranked, tid),
                         "p1": precision_at_k(ranked, [tid], 1),
                         "ap": average_precision(ranked, [tid]),
                         "r_at": [recall_at_k(ranked, [tid], k) for k in CURVE_K]})
        n = len(rows)
        mean = lambda key: sum(r[key] for r in rows) / n                     # noqa: E731
        results[mode] = {
            "MRR": mean("rr"), "P@1": mean("p1"),
            "R@5": sum(r["r_at"][4] for r in rows) / n,
            "R@10": sum(r["r_at"][9] for r in rows) / n,
            "MAP": mean("ap"),
            "recall_at_k": [sum(r["r_at"][k - 1] for r in rows) / n for k in CURVE_K],
            "n_not_retrieved": sum(r["rank"] is None for r in rows),
            "median_rank_found": sorted(r["rank"] for r in rows if r["rank"])[
                len([r for r in rows if r["rank"]]) // 2] if any(r["rank"] for r in rows) else None,
            "vocab_size": len(index.postings),
        }
        per_query[mode] = [{k: r[k] for k in ("tweet_id", "rank", "n_query_terms", "n_query_terms_in_index")}
                           for r in rows]

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {
        "question": "Does normalisation improve known-item retrieval with respelled queries?",
        "queries": {"file": "data/gold/e3_queries.tsv", "sha256": QUERIES_SHA256, "n": len(queries),
                    "provenance": "machine-generated by a language model under written rules, reviewed by "
                                  "the authors; see data/README.md"},
        "corpus": {"docs": len(docs), "splits": "train + dev + test", "relevant": "the target tweet_id only",
                   "targets_with_duplicate_text_in_corpus": n_dup_targets},
        "settings": {"hkey": dict(normalize.HKEY_OPTS),
                     "hkey_clusters": {"theta": normalize._variants.theta,
                                       "key_opts": normalize._variants.key_opts, "fit_on": "train"},
                     "hkey_vdrop": "ablation (agreed before the first E3 run)",
                     "ranking": "lnc.ltc cosine, full ranking of docs with score > 0, ties to lower doc number"},
        "note": "one relevant doc per query, so AP = reciprocal rank and MAP = MRR",
        "results": results,
        "per_query": per_query,
        "code_commit": commit,
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    plot(results, FIGURE)
    return report


def spread_labels(ys: list[float], gap: float, top: float = 1.0) -> list[float]:
    """Label heights at least `gap` apart, pushed down if the stack passes `top`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    for a, b in zip(reversed(order), list(reversed(order))[1:]):   # from the top down
        out[a] = min(out[a], top)
        if out[a] - out[b] < gap:
            out[b] = out[a] - gap
    return out


def plot(results: dict, path: Path) -> None:
    """R@k for k = 1..20, one line per system, direct labels at the right end."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, rule = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
    fig, ax = plt.subplots(figsize=(5.6, 3.3), facecolor=surface)
    ax.set_facecolor(surface)
    ends = []
    for mode in SYSTEMS:
        colour, marker = COLOURS[mode]
        ys = results[mode]["recall_at_k"]
        ax.plot(CURVE_K, ys, color=colour, linewidth=2, marker=marker, markersize=4,
                markeredgecolor=surface, markeredgewidth=0.8, label=LABELS[mode],
                linestyle=(0, (4, 2)) if mode == "hkey_vdrop" else "-")
        ends.append(ys[-1])
    for mode, y in zip(SYSTEMS, spread_labels(ends, 0.045)):
        ax.annotate(LABELS[mode], (CURVE_K[-1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=ink)
    ax.set_xlabel("k (rank cut-off)", fontsize=8, color=ink)
    ax.set_ylabel("Recall@k (target tweet in top k)", fontsize=8, color=ink)
    ax.set_xlim(0.5, CURVE_K[-1] + 6.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=rule, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=ink, labelsize=7.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7, frameon=False,
              labelcolor=ink, handlelength=2.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=surface)
    plt.close(fig)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        with open(RESULTS_JSON, encoding="utf-8") as f:
            plot(json.load(f)["results"], FIGURE)
        print(f"redrew {FIGURE} from {RESULTS_JSON}")
        return
    rep = run()
    print(f"E3: {rep['queries']['n']} queries over {rep['corpus']['docs']} tweets ({rep['seconds']} s)")
    print(f"{'system':16s} {'MRR':>6s} {'P@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MAP':>6s}  not found")
    for mode, r in rep["results"].items():
        print(f"{mode:16s} {r['MRR']:6.3f} {r['P@1']:6.3f} {r['R@5']:6.3f} {r['R@10']:6.3f} "
              f"{r['MAP']:6.3f}  {r['n_not_retrieved']}")
    print(f"wrote {RESULTS_JSON} and {FIGURE}")


if __name__ == "__main__":
    main()
