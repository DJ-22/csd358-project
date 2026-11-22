"""Sentiment tab: classify a tweet and show what the decision is made of.

The app shell calls render(). The models come from artifacts/sentiment_models.pkl
(build it with: python -m hinglishkey.classify.demo). They are trained on the SentiMix
train split only, exactly as in E2.
"""

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hinglishkey.classify.demo import knn_votes, load_models, nb_evidence, rocchio_contributions  # noqa: E402
from hinglishkey.data import load_split                     # noqa: E402
from hinglishkey.normalize import normalize_tokens          # noqa: E402
from hinglishkey.tokenize import tokenize                   # noqa: E402

ARTIFACTS = ROOT / "artifacts"
CLASSES = ("negative", "neutral", "positive")
COLOURS = {"negative": "#e03131", "neutral": "#7c818a", "positive": "#2f9e44"}   # red / grey / green
CLASSIFIERS = {"Naive Bayes": "nb", "Rocchio": "rocchio", "kNN": "knn"}
MODES = {
    "Raw (no normalisation)": "raw",
    "H-Key (V-norm), default": "hkey",
    "H-Key + clusters": "hkey_clusters",
    "H-Key V-drop (ablation)": "hkey_vdrop",
    "Soundex (baseline)": "soundex",
}
EXAMPLES = ["bohot accha laga yaar, thank you", "kya bakwas movie thi, paisa barbaad",
            "kal India ka match hai", "tum log kabhi nahi sudhroge"]
MAX_CHARS = 500


@st.cache_data(show_spinner=False)
def e2_scores(mtime: float) -> dict:
    """{(classifier, mode): test macro F1} from results/e2_sentiment.json (mtime keys the cache)."""
    results = json.loads((ROOT / "results" / "e2_sentiment.json").read_text(encoding="utf-8"))["results"]
    return {(c, m): r[c]["test"]["macro"]["f1"] for m, r in results.items() for c in CLASSIFIERS.values()}


@st.cache_resource(show_spinner=False)
def train_texts() -> list[str]:
    return [text for _, text, _ in load_split("train")]


def bars(values: dict[str, float], fmt, scale: float) -> str:
    """Horizontal bars, one per class, in the sentiment colours; width = value / scale."""
    rows = []
    for c in CLASSES:
        v = values.get(c, 0.0)
        width = 0 if scale <= 0 else max(0.0, min(1.0, v / scale)) * 100
        rows.append(f'<div class="hk-bar-row"><span class="hk-bar-label">{c}</span>'
                    f'<span class="hk-bar-track"><div class="hk-bar-fill {c}" style="width:{width:.1f}%">'
                    f'</div></span><span class="hk-bar-value">{fmt(v)}</span></div>')
    return "".join(rows)


def per_term_chart(items: list[tuple[str, dict[str, float]]], centre: bool) -> pd.DataFrame:
    """One row per token occurrence, one column per class (numbered labels keep repeats apart)."""
    rows = {}
    for i, (term, vals) in enumerate(items, start=1):
        if centre:
            mean = sum(vals.values()) / len(vals)
            vals = {c: v - mean for c, v in vals.items()}
        rows[f"{i}. {term}"] = [vals[c] for c in CLASSES]
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(CLASSES))


def show_nb(model, terms: list[str]) -> None:
    proba = model.predict_proba(terms)
    st.markdown("**Class probabilities** P(c | tweet)")
    st.markdown(bars(proba, lambda v: f"{v:.2f}", 1.0), unsafe_allow_html=True)
    evidence = nb_evidence(model, terms)
    if not evidence:
        st.info("None of these words was seen in training under this mode, so Naive Bayes "
                "falls back to the class priors P(c).")
        return
    st.markdown("**Per-token evidence** log P(t | c)")
    st.caption("Each bar is log P(t|c) minus its mean over the three classes: a bar to the right means "
               "the word is more likely in that class than on average. The class score is "
               "log P(c) plus the sum of log P(t|c) over the words.")
    df = per_term_chart(evidence, centre=True)
    st.bar_chart(df, horizontal=True, stack=False, color=[COLOURS[c] for c in CLASSES],
                 height=min(80 + 34 * len(df), 520), x_label="log P(t|c) - mean over classes", y_label="")
    with st.expander("Exact values: log P(c) and log P(t|c)"):
        exact = per_term_chart(evidence, centre=False)
        prior = pd.DataFrame([[model.log_prior[c] for c in CLASSES]], index=["log P(c), prior"],
                             columns=list(CLASSES))
        total = pd.DataFrame([[model.log_scores(terms)[c] for c in CLASSES]], index=["total score"],
                             columns=list(CLASSES))
        st.dataframe(pd.concat([prior, exact, total]).round(3), width="stretch")


def show_rocchio(model, terms: list[str]) -> None:
    d = model.distances(terms)
    st.markdown("**Squared distance to each class centroid** |mu(c) - v(d)|^2 (nearest wins)")
    st.markdown(bars(d, lambda v: f"{v:.3f}", max(d.values()) or 1.0), unsafe_allow_html=True)
    contrib = rocchio_contributions(model, terms)
    if not contrib:
        st.info("None of these words has a tf-idf weight under this mode (unseen in training), so the "
                "tweet is the zero vector and the smallest centroid is nearest.")
        return
    st.markdown("**Per-term pull towards each centroid** w(t,d) x mu_c(t)")
    st.caption("Summed over the words, these give mu(c).v(d); the squared distance is "
               "|mu(c)|^2 - 2 mu(c).v(d) + 1, so a longer bar pulls the tweet towards that class.")
    df = per_term_chart(contrib, centre=False)
    st.bar_chart(df, horizontal=True, stack=False, color=[COLOURS[c] for c in CLASSES],
                 height=min(80 + 34 * len(df), 520), x_label="w(t,d) x mu_c(t)", y_label="")


def show_knn(model, terms: list[str]) -> None:
    votes, nearest = knn_votes(model, terms)
    if not nearest:
        st.info(f"No training tweet shares a word with this one under this mode, so kNN answers the "
                f"most frequent training class ({model.majority}).")
        return
    st.markdown(f"**Votes among the k = {model.k} nearest training tweets** (cosine on tf-idf)")
    st.markdown(bars(dict(votes), lambda v: f"{int(v)}", float(model.k)), unsafe_allow_html=True)
    st.markdown("**The 5 nearest training tweets**")
    texts = train_texts()
    rows = [{"cosine": round(score, 3), "label": model.labels[i], "tweet": texts[i]} for i, score in nearest[:5]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"tweet": st.column_config.TextColumn(width="large")})


def render() -> None:
    st.subheader("Sentiment")
    st.caption("Type a Hinglish tweet. The classifier predicts negative, neutral or positive and shows "
               "the evidence behind the decision. All models are trained on the SentiMix training split.")
    try:
        bundle = load_models(ARTIFACTS)
    except FileNotFoundError:
        st.warning("The sentiment models are not built yet. Run "
                   "`python -m hinglishkey.classify.demo` once (with `src` on PYTHONPATH), then reload.")
        return

    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"hk_sent_ex_{ex}", width="stretch"):
            st.session_state["hk_sent_text"] = ex
    text = st.text_area("Tweet", key="hk_sent_text", height=80, max_chars=MAX_CHARS,
                        placeholder="e.g. bohot accha laga yaar")
    c1, c2 = st.columns([2, 3])
    clf_label = c1.radio("Classifier", list(CLASSIFIERS), horizontal=True, key="hk_sent_clf")
    mode_label = c2.selectbox("Normalisation", list(MODES), index=1, key="hk_sent_mode")
    clf, mode = CLASSIFIERS[clf_label], MODES[mode_label]

    if not text.strip():
        st.info("Type a tweet above, or click an example.")
        return
    try:
        words = tokenize(text)
        if not words:
            st.info("The tweet has no words to classify (only symbols, links, mentions or emoji).")
            return
        terms = normalize_tokens(words, mode)
        model = bundle["models"][(clf, mode)]
        chips = "".join(
            f'<span class="hk-chip">{html.escape(w)} → {html.escape(t)}</span>' for w, t in zip(words, terms))
        st.markdown(f"Words → terms under {html.escape(mode_label.split(',')[0])}: {chips}",
                    unsafe_allow_html=True)

        label = model.predict_one(terms)
        score_note = ""
        try:
            f1 = e2_scores((ROOT / "results" / "e2_sentiment.json").stat().st_mtime)[(clf, mode)]
            score_note = f" · this configuration scored {100 * f1:.1f} macro F1 on the SentiMix test set (E2)"
        except Exception:
            pass
        st.markdown(f'Prediction: <span class="hk-badge {label}">{label}</span>'
                    f'<span style="opacity:0.7;font-size:0.85rem">{html.escape(score_note)}</span>',
                    unsafe_allow_html=True)

        if clf == "nb":
            show_nb(model, terms)
        elif clf == "rocchio":
            show_rocchio(model, terms)
        else:
            show_knn(model, terms)
    except Exception:                                   # never show a traceback in the demo
        st.error("Sorry, this tweet could not be classified. Please try different words.")
