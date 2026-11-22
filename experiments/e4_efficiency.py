"""E4: what does normalisation cost and save?

    python experiments/e4_efficiency.py     # -> results/e4_efficiency.json, figures/e4_vocab_growth.png

For every mode, over all 20,000 SentiMix tweets (train, dev, test, in that
order): vocabulary size, index size (postings, positions, pickled bytes),
build time, query latency (the 100 E3 queries, ranked top-10) and the
vocabulary growth curve with a Heaps' law fit V = k * T^b (L5/IIR 5.1),
fitted by least squares on log10 V against log10 T.
Latencies depend on the machine; the JSON records which one.
"""

import csv
import json
import math
import pickle
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglishkey import data, normalize                    # noqa: E402
from hinglishkey.index import PositionalIndex               # noqa: E402
from hinglishkey.rank import Ranker                         # noqa: E402

RESULTS_JSON = ROOT / "results" / "e4_efficiency.json"
FIGURE = ROOT / "figures" / "e4_vocab_growth.png"
QUERIES_TSV = ROOT / "data" / "gold" / "e3_queries.tsv"
SYSTEMS = ("raw", "soundex", "hkey", "hkey_clusters", "hkey_vdrop")
LABELS = {"raw": "raw", "soundex": "Soundex", "hkey": "H-Key (V-norm)",
          "hkey_clusters": "H-Key + clusters", "hkey_vdrop": "H-Key V-drop (ablation)"}
COLOURS = {"raw": ("#2a78d6", "o"), "soundex": ("#eb6834", "s"), "hkey": ("#1baf7a", "^"),
           "hkey_clusters": ("#eda100", "D"), "hkey_vdrop": ("#e87ba4", "v")}
STEP = 500          # record the vocabulary after every 500 tweets
REPEATS = 5         # timed runs per query; the median is kept


def heaps_fit(tokens: list[int], vocab: list[int]) -> dict:
    """Least-squares line through (log10 T, log10 V): log V = log k + b log T."""
    xs = [math.log10(t) for t in tokens]
    ys = [math.log10(v) for v in vocab]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    log_k = my - b * mx
    ss_res = sum((y - (log_k + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"k": 10 ** log_k, "b": b, "r2": 1 - ss_res / ss_tot}


def run() -> dict:
    t0 = time.perf_counter()
    train = data.load_split("train")
    normalize.fit_variants([text for _, text, _ in train])
    fit_seconds = time.perf_counter() - t0
    docs = [(tid, text) for s in ("train", "dev", "test") for tid, text, _ in data.load_split(s)]
    with open(QUERIES_TSV, encoding="utf-8", newline="") as f:
        queries = [r["respelled_query"] for r in csv.DictReader(f, delimiter="\t")]

    results = {}
    for mode in SYSTEMS:
        index = PositionalIndex(mode)
        growth_t, growth_v, n_tokens = [], [], 0
        t0 = time.perf_counter()
        for i, (tid, text) in enumerate(docs, start=1):
            terms = index.terms_of(text)
            index.add(tid, terms)
            n_tokens += len(terms)
            if i % STEP == 0 or i == len(docs):
                growth_t.append(n_tokens)
                growth_v.append(len(index.postings))
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ranker = Ranker(index)
        ranker_s = time.perf_counter() - t0

        latencies = []
        for q in queries:
            runs = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                ranker.search(index.terms_of(q), k=10)
                runs.append(time.perf_counter() - t0)
            latencies.append(statistics.median(runs) * 1000)
        latencies.sort()
        results[mode] = {
            "vocab_size": len(index.postings),
            "n_tokens": n_tokens,
            "n_postings": index.n_postings(),
            "n_positions": index.n_positions(),
            "pickle_bytes": len(pickle.dumps(index.__dict__, protocol=pickle.HIGHEST_PROTOCOL)),
            "build_seconds": round(build_s, 3),
            "ranker_setup_seconds": round(ranker_s, 3),
            "query_ms": {"median": round(statistics.median(latencies), 3),
                         "p95": round(latencies[int(0.95 * (len(latencies) - 1))], 3),
                         "max": round(latencies[-1], 3)},
            "growth": {"tokens": growth_t, "vocab": growth_v},
            "heaps": heaps_fit(growth_t, growth_v),
        }
    raw_v = results["raw"]["vocab_size"]
    for mode in SYSTEMS:
        results[mode]["vocab_vs_raw"] = results[mode]["vocab_size"] / raw_v

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {
        "question": "Efficiency of raw vs normalised indexes",
        "corpus": {"docs": len(docs), "order": "train, dev, test"},
        "latency_protocol": f"100 E3 queries, ranked top-10 (tokenise + normalise + score + heap), "
                            f"median of {REPEATS} runs per query, then median/p95/max over queries",
        "cluster_fit_seconds_on_train": round(fit_seconds, 2),
        "machine": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor()},
        "results": results,
        "code_commit": commit,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    plot(results, FIGURE)
    return report


def plot(results: dict, path: Path) -> None:
    """Vocabulary growth on log-log axes (Heaps' law), one line per mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, rule = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
    fig, ax = plt.subplots(figsize=(5.6, 3.3), facecolor=surface)
    ax.set_facecolor(surface)
    ends = []
    for mode in SYSTEMS:
        colour, marker = COLOURS[mode]
        g = results[mode]["growth"]
        ax.plot(g["tokens"], g["vocab"], color=colour, linewidth=2, marker=marker, markersize=3.5,
                markevery=8, markeredgecolor=surface, markeredgewidth=0.8, label=LABELS[mode],
                linestyle=(0, (4, 2)) if mode == "hkey_vdrop" else "-")
        ends.append((g["tokens"][-1], g["vocab"][-1]))
    # direct labels in log space, nudged apart
    logs = [math.log10(v) for _, v in ends]
    order = sorted(range(len(logs)), key=lambda i: logs[i])
    placed = list(logs)
    for a, b in zip(order, order[1:]):
        if placed[b] - placed[a] < 0.07:
            placed[b] = placed[a] + 0.07
    for mode, (t, _), y in zip(SYSTEMS, ends, placed):
        h = results[mode]["heaps"]
        ax.annotate(f"{LABELS[mode]}  (b = {h['b']:.2f})", (t, 10 ** y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7, color=ink)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tokens seen, T (log scale)", fontsize=8, color=ink)
    ax.set_ylabel("Vocabulary size, V (log scale)", fontsize=8, color=ink)
    xmin = min(results[m]["growth"]["tokens"][0] for m in SYSTEMS)
    xmax = max(t for t, _ in ends)
    ax.set_xlim(xmin * 0.8, xmax * 4.5)
    ax.grid(axis="y", which="major", color=rule, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=ink, labelsize=7.5, which="both")
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
    print(f"{'mode':15s} {'vocab':>7s} {'vs raw':>7s} {'postings':>9s} {'MB':>6s} {'build s':>8s} "
          f"{'q med ms':>9s} {'p95':>6s} {'max':>6s} {'Heaps b':>8s}")
    for m, r in rep["results"].items():
        print(f"{m:15s} {r['vocab_size']:7d} {r['vocab_vs_raw']:7.3f} {r['n_postings']:9d} "
              f"{r['pickle_bytes'] / 1e6:6.2f} {r['build_seconds']:8.2f} {r['query_ms']['median']:9.2f} "
              f"{r['query_ms']['p95']:6.2f} {r['query_ms']['max']:6.2f} {r['heaps']['b']:8.3f}")
    print(f"cluster fit on train: {rep['cluster_fit_seconds_on_train']} s; wrote {RESULTS_JSON} and {FIGURE}")


if __name__ == "__main__":
    main()
