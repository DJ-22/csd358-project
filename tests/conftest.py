"""Make `import hinglishkey` work when running `pytest` from the repo root (src/ layout),
and share the SentiMix fixtures used by the classifier cross-checks."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def sentimix_raw():
    """Train and dev as token lists in mode "raw": (X_train, y_train, X_dev, y_dev)."""
    from hinglishkey import data
    from hinglishkey.normalize import normalize_tokens, tokenize

    if not (data.RAW_DIR / data.SPLIT_FILES["train"]).exists():
        pytest.skip("SentiMix raw files not downloaded (see data/README.md)")

    def prepare(split):
        rows = data.load_split(split)
        return [normalize_tokens(tokenize(text), "raw") for _, text, _ in rows], [label for _, _, label in rows]

    return (*prepare("train"), *prepare("dev"))


@pytest.fixture(scope="session")
def sentimix_raw_tfidf(sentimix_raw):
    """The same data as scipy sparse matrices of OUR ltc tf-idf vectors, for feeding scikit-learn.

    scikit-learn's own TfidfVectorizer uses a different formula (natural log, idf + 1), so the
    cross-checks give both implementations identical input vectors and compare only the classifier.
    """
    from scipy.sparse import csr_matrix

    from hinglishkey.classify import TfidfVectorizer

    X_train, _, X_dev, _ = sentimix_raw
    vectorizer = TfidfVectorizer().fit(X_train)
    column = {t: j for j, t in enumerate(sorted(vectorizer.idf))}

    def to_csr(docs):
        rows, cols, vals = [], [], []
        for i, v in enumerate(vectorizer.transform(docs)):
            for t, w in v.items():
                rows.append(i)
                cols.append(column[t])
                vals.append(w)
        return csr_matrix((vals, (rows, cols)), shape=(len(docs), len(column)))

    return to_csr(X_train), to_csr(X_dev)
