# HinglishKey

**A phonetic equivalence-class index for spelling-robust retrieval and classification of code-mixed Hinglish text** (CSD358 Information Retrieval course project).

Romanised Hindi has no standard spelling: *bahut*, *bohot*, *bht* and *bhot* are the same word. HinglishKey maps spelling variants to shared index terms. It uses H-Key, a Soundex-style phonetic key for Hinglish, plus variant clusters learned from the corpus. These terms are then used for retrieval and for sentiment classification. Every algorithm is written from scratch in Python; scikit-learn is used only to cross-check results in the tests.

## Setup

Python 3.12, CPU only.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The SentiMix Hinglish data (SemEval-2020 Task 9, CC BY 4.0) is not committed. To download and check it, see [`data/README.md`](data/README.md).

## Reproducing the experiments

Each command writes its numbers to `results/` and its figures to `figures/`. Run them from the repo root.

| Experiment                  | Command                                                                                      | Output                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| E1 variant grouping         | `python experiments/e1_variants.py run`                                                    | `results/e1_variants.json`, `figures/e1_theta.png`                                       |
| E2 sentiment classification | `python experiments/e2_sentiment.py`                                                       | `results/e2_sentiment.json`, `figures/e2_f1_bar.png`, `figures/e2_confusion.png`       |
| E3 known-item retrieval     | `python experiments/e3_retrieval.py` then `python experiments/e3_retrieval.py bootstrap` | `results/e3_retrieval.json`, `results/e3_bootstrap.json`, `figures/e3_recall_at_k.png` |
| E4 efficiency               | `python experiments/e4_efficiency.py`                                                      | `results/e4_efficiency.json`, `figures/e4_vocab_growth.png`                              |

How the gold files in `data/gold/` were made, and their provenance, is described in [`data/README.md`](data/README.md).

## Running the demo

The demo is a Streamlit web app. It runs offline on a laptop CPU, with these tabs: Search, Phrase search, Sentiment, Variant explorer and Results.

1. Build the precomputed files once (about 30 s; they go to `artifacts/`, which git ignores):

   ```powershell
   $env:PYTHONPATH = "src"
   python -m hinglishkey.index            # search indexes for every mode + the variant model
   python -m hinglishkey.classify.demo    # sentiment models (NB, Rocchio, kNN for every mode)
   ```
2. Start the app:

   ```powershell
   streamlit run app/streamlit_app.py
   ```

   The first start loads everything (about 5 s). After that, each interaction takes well under 300 ms. If the artifacts are missing, the app says so and shows the commands above; it never shows a traceback.

Rebuild the artifacts after re-running E2 (the kNN k comes from `results/e2_sentiment.json`) or after any change to the normalisation code.

## Tests

```powershell
python -m pytest -q
```

The suite includes agreement checks with scikit-learn for the three classifiers and the metrics, and a headless smoke test of the demo app (skipped until the artifacts are built).

## Repository layout

```
src/hinglishkey/      data loading, metrics, tokenisation, H-Key, variant clusters,
                      normalisation, positional index, ranking; classify/ (NB, Rocchio, kNN)
experiments/          one script per experiment (E1-E4) and the gold-merge script
app/                  streamlit_app.py (shell) and tabs/ (one module per tab)
data/                 raw/ (downloaded, not committed), gold/ (gold files), README.md
results/, figures/    experiment outputs read by the report and the Results tab
report/               LaTeX report sections
tests/                pytest suite
```
