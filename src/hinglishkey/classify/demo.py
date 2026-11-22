"""Precomputed sentiment models for the demo, and the explanations the Sentiment tab shows.

    python -m hinglishkey.classify.demo        # writes artifacts/sentiment_models.pkl

The models are trained exactly as in E2: on the TRAIN split only, for every demo mode
(the four main modes plus the hkey_vdrop ablation). kNN uses the k that E2 chose on dev for
that mode, read from results/e2_sentiment.json, so the demo and the reported results agree.

Explanations (what each classifier's decision is made of):
- NB:      per token, log P(t|c) for every class c (L12). The class score is
           log P(c) + the sum of these values, so a token "votes" for the class where it is largest.
- Rocchio: per term, w_{t,d} * mu_c(t), the term's share of the dot product mu(c).v(d) (L13).
           The squared distance is |mu(c)|^2 - 2 mu(c).v(d) + 1, so a larger share pulls the
           document towards class c.
- kNN:     the votes among the k nearest training tweets and the nearest ones themselves (L13).
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from hinglishkey import normalize
from hinglishkey.classify.knn import KNN
from hinglishkey.classify.nb import MultinomialNB
from hinglishkey.classify.rocchio import Rocchio

ROOT = Path(__file__).resolve().parents[3]
MODELS_FILE = "sentiment_models.pkl"
DEMO_MODES = normalize.MODES + normalize.ABLATION_MODES
CLASSIFIERS = ("nb", "rocchio", "knn")


def build_models(out_dir: Path, e2_results: Path) -> Path:
    """Train every (classifier, mode) pair on TRAIN and pickle them into one bundle."""
    from hinglishkey.data import load_split

    e2 = json.loads(Path(e2_results).read_text(encoding="utf-8"))["results"]
    train = load_split("train")
    texts = [text for _, text, _ in train]
    labels = [label for _, _, label in train]
    normalize.fit_variants(texts)                      # clusters from train only, as in E2

    models, k_used = {}, {}
    for mode in DEMO_MODES:
        docs = [normalize.normalize_tokens(normalize.tokenize(t), mode) for t in texts]
        k_used[mode] = int(e2[mode]["knn"]["k"])
        models[("nb", mode)] = MultinomialNB().fit(docs, labels)
        models[("rocchio", mode)] = Rocchio().fit(docs, labels)
        models[("knn", mode)] = KNN(k=k_used[mode]).fit(docs, labels)

    bundle = {"models": models, "k": k_used, "train_ids": [tid for tid, _, _ in train],
              "variants": normalize._variants, "modes": list(DEMO_MODES)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / MODELS_FILE
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


_cache: dict = {}


def load_models(artifacts_dir: Path) -> dict:
    """Load the bundle written by build_models. Raises FileNotFoundError if it is missing.

    Cached per directory, so every rerun of the app shares one copy.
    Mode hkey_clusters needs the fitted variant clusters; if nothing has set them yet, the
    copy saved in the bundle (fitted on the same train split) is used.
    """
    key = str(Path(artifacts_dir).resolve())
    if key not in _cache:
        with open(Path(artifacts_dir) / MODELS_FILE, "rb") as f:
            _cache[key] = pickle.load(f)
    bundle = _cache[key]
    if normalize._variants is None:
        normalize._variants = bundle["variants"]
    return bundle


def nb_evidence(model: MultinomialNB, tokens: list[str]) -> list[tuple[str, dict[str, float]]]:
    """(term, {class: log P(term|class)}) for every token occurrence the model knows, in order."""
    return [(t, {c: model.log_cond[c][t] for c in model.classes}) for t in tokens if t in model.vocab]


def rocchio_contributions(model: Rocchio, tokens: list[str]) -> list[tuple[str, dict[str, float]]]:
    """(term, {class: w_{t,d} * mu_c(t)}) for every term of the document vector v(d).

    Summed over the terms, these give mu(c).v(d) for each class.
    """
    v = model.vectorizer.transform_one(tokens)
    return [(t, {c: w * model.centroids[c].get(t, 0.0) for c in model.classes}) for t, w in v.items()]


def knn_votes(model: KNN, tokens: list[str]) -> tuple[Counter, list[tuple[int, float]]]:
    """Vote counts among the k nearest training docs, and those neighbours (id, cosine)."""
    nearest = model.neighbours(tokens)
    if nearest[0][1] == 0.0:                           # zero vector: no neighbour is similar
        return Counter(), []
    return Counter(model.labels[i] for i, _ in nearest), nearest


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    path = build_models(ROOT / "artifacts", ROOT / "results" / "e2_sentiment.json")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.1f} s)")
