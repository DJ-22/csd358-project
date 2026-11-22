"""Search tab: ranked tweet search that works for any spelling.

The app shell calls render(). Results come from the precomputed indexes in
artifacts/ (build them with: python -m hinglishkey.index).
"""

import html
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hinglishkey.index import load_demo                    # noqa: E402
from hinglishkey.normalize import normalize_tokens          # noqa: E402
from hinglishkey.tokenize import align_tokens, tokenize     # noqa: E402

ARTIFACTS = ROOT / "artifacts"
MODES = {
    "H-Key (V-norm), default": "hkey",
    "H-Key + clusters": "hkey_clusters",
    "H-Key V-drop (ablation)": "hkey_vdrop",
    "Soundex (baseline)": "soundex",
}
EXAMPLES = ["bohot acha laga", "kya baat hai", "pyar zindgi", "thik hai yaar"]
TOP_K = 10
STYLE = """
<style>
.hk-match { background: var(--hk-match, #fde68a); color: inherit; padding: 0 2px; border-radius: 3px; }
.hk-tweet { font-size: 0.95rem; line-height: 1.45; margin: 0.1rem 0 0.15rem 0; overflow-wrap: anywhere; }
.hk-meta { font-size: 0.78rem; opacity: 0.7; margin-bottom: 0.6rem; }
.hk-chip { display: inline-block; font-family: monospace; font-size: 0.82rem; padding: 1px 6px;
           margin: 0 6px 4px 0; border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; }
</style>
"""


def highlight(text: str, mode: str, wanted: set[str]) -> str:
    """Tweet as HTML with every word whose term (under `mode`) is in `wanted` marked."""
    parts = []
    tokens = tokenize(text)
    terms = normalize_tokens(tokens, mode)
    for word, positions in align_tokens(text):
        safe = html.escape(word)
        if any(terms[p] in wanted for p in positions):
            safe = f'<mark class="hk-match">{safe}</mark>'
        parts.append(safe)
    return " ".join(parts)


def show_results(demo, mode: str, query: str, title: str) -> None:
    index, ranker = demo.indexes[mode], demo.rankers[mode]
    t0 = time.perf_counter()
    terms = index.terms_of(query)
    hits = ranker.search(terms, TOP_K)
    ms = (time.perf_counter() - t0) * 1000
    st.markdown(f"**{title}**")
    st.caption(f"{len(hits)} shown · searched {index.n_docs:,} tweets in {ms:.0f} ms")
    if not hits:
        st.info("No tweet shares a term with this query in this index.")
        return
    wanted = set(terms)
    for rank, (d, score) in enumerate(hits, start=1):
        tid = index.doc_ids[d]
        st.markdown(f'<div class="hk-tweet">{rank}. {highlight(demo.texts[tid], mode, wanted)}</div>'
                    f'<div class="hk-meta">score {score:.3f} · tweet {html.escape(tid)} · '
                    f'{demo.splits[tid]} · {demo.labels[tid]}</div>', unsafe_allow_html=True)


def render() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    st.subheader("Search")
    st.caption("Type a query in any spelling. Each index maps spelling variants to one term, "
               "so 'bohot' can find 'bahut'. Matched words are highlighted.")
    try:
        demo = load_demo(ARTIFACTS)
    except FileNotFoundError:
        st.warning("The search indexes are not built yet. Run `python -m hinglishkey.index` once, then reload.")
        return

    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"hk_search_ex_{ex}", width="stretch"):
            st.session_state["hk_search_q"] = ex
    query = st.text_input("Query", key="hk_search_q", placeholder="e.g. bohot acha laga")
    c1, c2 = st.columns([3, 2])
    label = c1.radio("Normalisation", list(MODES), horizontal=True, key="hk_search_mode")
    compare = c2.toggle("Compare with the raw index", value=True, key="hk_search_compare")
    mode = MODES[label]

    if not query.strip():
        st.info("Type a query above, or click an example.")
        return
    try:
        words = tokenize(query)
        if not words:
            st.info("The query has no words to search for (only symbols, links or mentions).")
            return
        chips = "".join(f'<span class="hk-chip">{html.escape(w)} → {html.escape(t)}</span>'
                        for w, t in zip(words, normalize_tokens(words, mode)))
        st.markdown(f"Query terms under {label.split(',')[0]}: {chips}", unsafe_allow_html=True)
        if compare:
            left, right = st.columns(2)
            with left:
                show_results(demo, mode, query, label.split(",")[0])
            with right:
                show_results(demo, "raw", query, "Raw index (no normalisation)")
        else:
            show_results(demo, mode, query, label.split(",")[0])
    except Exception:                                   # never show a traceback in the demo
        st.error("Sorry, this query could not be processed. Please try different words.")
