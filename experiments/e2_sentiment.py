"""E2: does spelling normalisation improve sentiment classification?

    {NB, Rocchio, kNN} x {raw, soundex, hkey, hkey_clusters} on SentiMix Hinglish.

Protocol
- Train on the official train split only (including the variant clusters for hkey_clusters).
- The only hyperparameter, k for kNN, is chosen on dev by macro F1 (ties -> the smaller k).
  The grid starts with the L13 slide values {1, 3, 9, 15}; because the best k was at the edge
  of that grid, it is extended with {21, 31, 45, 61}. NB and Rocchio have no hyperparameters.
- Test is predicted once per (classifier, mode), with the k chosen on dev.
- Reported: macro P/R/F1 (main metric), weighted P/R/F1 (the official SemEval-2020 Task 9
  metric, used for the comparison with published systems), accuracy, per-class P/R/F1 and
  confusion matrices, on dev and test.
- Significance: paired bootstrap of the test macro-F1 difference between each normalised mode
  and raw, for the same classifier (1000 resamples of the test set, seed 42).

Outputs
    results/e2_sentiment.json
    figures/e2_f1_bar.png        test macro F1, classifier x mode
    figures/e2_confusion.png     test confusion matrices, classifier x mode

Run from the repo root:  python experiments/e2_sentiment.py
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglishkey import normalize  # noqa: E402
from hinglishkey.classify.knn import K_VALUES, KNN  # noqa: E402
from hinglishkey.classify.nb import MultinomialNB  # noqa: E402
from hinglishkey.classify.rocchio import Rocchio  # noqa: E402
from hinglishkey.data import LABELS, load_split  # noqa: E402
from hinglishkey.metrics import accuracy, confusion_matrix, precision_recall_f1  # noqa: E402

SEED = 42
K_GRID = K_VALUES + (21, 31, 45, 61)     # L13 slide values, extended because 15 was the best (edge)
N_BOOTSTRAP = 1000
CLASSIFIERS = ("nb", "rocchio", "knn")
CLASSIFIER_NAMES = {"nb": "Naive Bayes", "rocchio": "Rocchio", "knn": "kNN"}
MODE_COLOURS = {"raw": "#2a78d6", "soundex": "#eb6834", "hkey": "#1baf7a", "hkey_clusters": "#eda100"}

# Published SemEval-2020 Task 9 Hinglish results, weighted F1 (%) on the official test set.
# Source: Patwa et al. (2020), Table 3 (systems ranked by "Weighted Average F1").
PUBLISHED = [
    {"system": "KK2018 (rank 1, XLM-R)", "paper": "Patwa et al. 2020, Table 3",
     "weighted_f1": 75.0, "f1_positive": 79.9, "f1_neutral": 68.9, "f1_negative": 76.9},
    {"system": "Reed / gopalanvinay (rank 4, bagged BERT-large)", "paper": "Patwa et al. 2020, Table 3; Gopalan and Hopkins 2020",
     "weighted_f1": 71.3, "f1_positive": 77.5, "f1_neutral": 64.3, "f1_negative": 73.0},
    {"system": "Organisers' M-BERT baseline (rank 45)", "paper": "Patwa et al. 2020, Table 3",
     "weighted_f1": 65.4, "f1_positive": 70.7, "f1_neutral": 58.1, "f1_negative": 68.3},
]


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    """All E2 metrics for one prediction run (metrics.py, L6)."""
    macro = precision_recall_f1(y_true, y_pred, "macro")
    weighted = precision_recall_f1(y_true, y_pred, "weighted")
    return {
        "macro": {k: macro[k] for k in ("precision", "recall", "f1")},
        "weighted": {k: weighted[k] for k in ("precision", "recall", "f1")},
        "accuracy": accuracy(y_true, y_pred),
        "per_class": {c: macro["per_class"].get(c, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
                      for c in LABELS},
        "confusion": {"labels": list(LABELS), "rows_true_cols_pred": confusion_matrix(y_true, y_pred, LABELS)},
    }


def macro_f1_batch(true_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Macro F1 for many resampled test sets at once (one per row), same formula as metrics.py.

    true_idx, pred_idx: integer class ids, shape (n_resamples, n_docs).
    Per class c: tp = #(true=c and pred=c), fp = #(pred=c) - tp, fn = #(true=c) - tp.
    """
    f1s = []
    for c in range(n_classes):
        tp = ((true_idx == c) & (pred_idx == c)).sum(axis=1)
        pred_c = (pred_idx == c).sum(axis=1)
        true_c = (true_idx == c).sum(axis=1)
        p = np.divide(tp, pred_c, out=np.zeros(len(tp)), where=pred_c > 0)
        r = np.divide(tp, true_c, out=np.zeros(len(tp)), where=true_c > 0)
        f1s.append(np.divide(2 * p * r, p + r, out=np.zeros(len(tp)), where=(p + r) > 0))
    return np.mean(f1s, axis=0)


def paired_bootstrap(y_true, pred_base, pred_new, resamples: np.ndarray) -> dict:
    """Paired bootstrap test of delta = macroF1(new) - macroF1(base) on the test set.

    Each row of `resamples` is a test set of the same size drawn WITH replacement. Both systems
    are scored on the SAME resampled tweets, so the difference compares like with like.
    Returns the observed delta, the 95% percentile interval of the resampled deltas, and the
    one-sided p-value = fraction of resamples where the new system is not better (delta <= 0).
    """
    index = {c: i for i, c in enumerate(LABELS)}
    t = np.array([index[c] for c in y_true])
    a = np.array([index[c] for c in pred_base])
    b = np.array([index[c] for c in pred_new])
    observed = (macro_f1_batch(t[None, :], b[None, :], len(LABELS))
                - macro_f1_batch(t[None, :], a[None, :], len(LABELS)))[0]
    deltas = (macro_f1_batch(t[resamples], b[resamples], len(LABELS))
              - macro_f1_batch(t[resamples], a[resamples], len(LABELS)))
    return {"delta_macro_f1": float(observed),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_one_sided": float(np.mean(deltas <= 0)),
            "n_resamples": int(len(resamples))}


def run_mode(mode: str, texts: dict, labels: dict) -> dict:
    """Train all three classifiers on one normalisation mode; tune k on dev; predict test once."""
    docs = {s: [normalize.normalize_tokens(normalize.tokenize(t), mode) for t in texts[s]] for s in texts}
    vocab = len({t for d in docs["train"] for t in d})
    out = {"train_vocabulary": vocab}

    for name in CLASSIFIERS:
        t0 = time.perf_counter()
        if name == "nb":
            model = MultinomialNB().fit(docs["train"], labels["train"])
        elif name == "rocchio":
            model = Rocchio().fit(docs["train"], labels["train"])
        else:
            model = KNN().fit(docs["train"], labels["train"])
        fit_seconds = time.perf_counter() - t0

        entry = {}
        if name == "knn":
            sweep = {k: evaluate(labels["dev"], model.predict(docs["dev"], k)) for k in K_GRID}
            best_k = max(K_GRID, key=lambda k: (sweep[k]["macro"]["f1"], -k))
            best_lecture_k = max(K_VALUES, key=lambda k: (sweep[k]["macro"]["f1"], -k))
            model.k = best_k
            entry["k"] = best_k
            entry["k_best_in_lecture_grid"] = best_lecture_k
            entry["dev_k_sweep_macro_f1"] = {str(k): sweep[k]["macro"]["f1"] for k in K_GRID}
            entry["dev"] = sweep[best_k]
        else:
            entry["dev"] = evaluate(labels["dev"], model.predict(docs["dev"]))

        t0 = time.perf_counter()
        test_pred = model.predict(docs["test"])
        entry["test_predict_seconds"] = time.perf_counter() - t0
        entry["fit_seconds"] = fit_seconds
        entry["test"] = evaluate(labels["test"], test_pred)
        entry["_test_pred"] = test_pred            # used for the bootstrap, removed before saving
        out[name] = entry
    return out


def plot_f1_bars(results: dict, path: Path) -> None:
    modes = normalize.MODES
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.2
    x = np.arange(len(CLASSIFIERS))
    for i, mode in enumerate(modes):
        values = [100 * results[mode][c]["test"]["macro"]["f1"] for c in CLASSIFIERS]
        bars = ax.bar(x + (i - 1.5) * width, values, width * 0.9, label=mode, color=MODE_COLOURS[mode],
                      edgecolor="#fcfcfb", linewidth=1)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7, color="#52514e")
    ax.set_xticks(x, [CLASSIFIER_NAMES[c] for c in CLASSIFIERS])
    ax.set_ylabel("Test macro F1 (%)", color="#52514e")
    ax.set_ylim(0, 100)
    ax.set_title("E2: sentiment classification on SentiMix test, by normalisation mode", fontsize=10)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#898781")
    ax.tick_params(colors="#52514e")
    ax.legend(title="mode", fontsize=8, title_fontsize=8, frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="#fcfcfb")
    plt.close(fig)


def plot_confusions(results: dict, path: Path) -> None:
    modes = normalize.MODES
    fig, axes = plt.subplots(len(CLASSIFIERS), len(modes), figsize=(11, 8.2))
    short = ["neg", "neu", "pos"]
    for r, clf in enumerate(CLASSIFIERS):
        for c, mode in enumerate(modes):
            ax = axes[r][c]
            m = np.array(results[mode][clf]["test"]["confusion"]["rows_true_cols_pred"])
            ax.imshow(m, cmap="Blues", vmin=0, vmax=m.sum(axis=1).max())
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8,
                            color="#ffffff" if m[i, j] > 0.55 * m.sum(axis=1).max() else "#0b0b0b")
            ax.set_xticks(range(3), short, fontsize=7)
            ax.set_yticks(range(3), short, fontsize=7)
            if r == len(CLASSIFIERS) - 1:
                ax.set_xlabel("predicted", fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{CLASSIFIER_NAMES[clf]}\ntrue", fontsize=8)
            if r == 0:
                ax.set_title(mode, fontsize=9)
    fig.suptitle("E2: test confusion matrices (rows = true class, columns = predicted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    random.seed(SEED)          # nothing in E2 is random; seeds fixed by project rule
    np.random.seed(SEED)
    started = time.perf_counter()

    splits = {s: load_split(s) for s in ("train", "dev", "test")}
    texts = {s: [text for _, text, _ in rows] for s, rows in splits.items()}
    labels = {s: [label for _, _, label in rows] for s, rows in splits.items()}

    normalize.fit_variants(texts["train"])     # clusters learned from train only

    results = {}
    for mode in normalize.MODES:
        print(f"running mode {mode} ...", flush=True)
        results[mode] = run_mode(mode, texts, labels)

    # Paired bootstrap: the same 1000 resampled test sets for every comparison.
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(0, len(labels["test"]), size=(N_BOOTSTRAP, len(labels["test"])))
    for clf in CLASSIFIERS:
        base = results["raw"][clf]["_test_pred"]
        for mode in normalize.MODES[1:]:
            results[mode][clf]["test_vs_raw_bootstrap"] = paired_bootstrap(
                labels["test"], base, results[mode][clf]["_test_pred"], resamples)
    for mode in normalize.MODES:
        for clf in CLASSIFIERS:
            del results[mode][clf]["_test_pred"]

    # Reference point for the L6 accuracy discussion: always predict the most frequent train class.
    majority = max(LABELS, key=labels["train"].count)
    majority_baseline = {"predicts": majority,
                         "dev": evaluate(labels["dev"], [majority] * len(labels["dev"])),
                         "test": evaluate(labels["test"], [majority] * len(labels["test"]))}

    clusters = normalize._variants
    output = {
        "experiment": "E2",
        "question": "Does spelling normalisation improve sentiment classification?",
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "seed": SEED,
            "dataset": "SentiMix Hinglish (SemEval-2020 Task 9), official splits",
            "split_sizes": {s: len(rows) for s, rows in splits.items()},
            "modes": list(normalize.MODES),
            "classifiers": list(CLASSIFIERS),
            "features": {"nb": "term counts of normalised tokens (multinomial)",
                         "rocchio": "ltc tf-idf, cosine-normalised; nearest centroid by Euclidean distance",
                         "knn": "ltc tf-idf, cosine similarity; majority vote, ties -> class of nearest neighbour"},
            "knn_k_grid": list(K_GRID),
            "knn_k_grid_lecture": list(K_VALUES),
            "tuning": "k chosen on dev by macro F1 (ties -> smaller k); test predicted once",
            "significance": f"paired bootstrap, {N_BOOTSTRAP} resamples of the test set, seed {SEED}; "
                            "delta = macro F1(mode) - macro F1(raw), same classifier; "
                            "p = one-sided fraction of resamples with delta <= 0",
            "hkey_opts": dict(normalize.HKEY_OPTS),
            "variant_clusters": {"theta": clusters.theta, "key_opts": clusters.key_opts},
            "hkey_settings_source": "frozen; chosen on the E1 dev groups by strict pairwise F1 "
                                    "(results/e1_variants.json), before this run",
            "main_metric": "macro F1",
            "comparison_metric": "weighted F1 (support-weighted mean of per-class F1), as in Patwa et al. 2020 Sec. 3.1",
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "results": results,
        "majority_baseline": majority_baseline,
        "published_reference": PUBLISHED,
        "runtime_seconds": None,
    }
    output["runtime_seconds"] = time.perf_counter() - started

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "e2_sentiment.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    plot_f1_bars(results, ROOT / "figures" / "e2_f1_bar.png")
    plot_confusions(results, ROOT / "figures" / "e2_confusion.png")

    print(f"\n{'classifier':10s} {'mode':14s} {'k':>3s} {'dev mF1':>8s} {'test mF1':>9s} {'test wF1':>9s} {'test acc':>9s}")
    for clf in CLASSIFIERS:
        for mode in normalize.MODES:
            e = results[mode][clf]
            print(f"{clf:10s} {mode:14s} {str(e.get('k', '-')):>3s} {100*e['dev']['macro']['f1']:8.2f} "
                  f"{100*e['test']['macro']['f1']:9.2f} {100*e['test']['weighted']['f1']:9.2f} "
                  f"{100*e['test']['accuracy']:9.2f}")
    print("\npaired bootstrap vs raw (test macro F1):")
    for clf in CLASSIFIERS:
        for mode in normalize.MODES[1:]:
            b = results[mode][clf]["test_vs_raw_bootstrap"]
            print(f"  {clf:8s} {mode:14s} delta {100*b['delta_macro_f1']:+5.2f}  "
                  f"95% CI [{100*b['ci95'][0]:+5.2f}, {100*b['ci95'][1]:+5.2f}]  p {b['p_one_sided']:.3f}")
    print(f"\nwrote results/e2_sentiment.json, figures/e2_f1_bar.png, figures/e2_confusion.png "
          f"({output['runtime_seconds']:.1f} s)")


if __name__ == "__main__":
    main()
