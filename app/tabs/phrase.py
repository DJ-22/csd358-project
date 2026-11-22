"""Phrase tab: exact-phrase search that tolerates spelling variants.

The words must appear next to each other, in order, but each word may be
spelled any way that maps to the same term (positional intersection over
the normalised index). The app shell calls render().
"""

import html
import sys
import time
from collections import Counter
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
EXAMPLES = ["bahut accha", "kya baat hai", "thik hai", "modi ji"]
SHOW = 15
SPELL_SAMPLE = 300     # matches used for the 'spellings matched' summary
STYLE = """
<style>
.hk-match { background: var(--hk-match, #fde68a); color: inherit; padding: 0 2px; border-radius: 3px; }
.hk-tweet { font-size: 0.95rem; line-height: 1.45; margin: 0.1rem 0 0.15rem 0; overflow-wrap: anywhere; }
.hk-meta { font-size: 0.78rem; opacity: 0.7; margin-bottom: 0.6rem; }
.hk-chip { display: inline-block; font-family: monospace; font-size: 0.82rem; padding: 1px 6px;
           margin: 0 6px 4px 0; border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; }
</style>
"""


def marked(text: str, starts: list[int], length: int) -> tuple[str, list[str]]:
    """Tweet HTML with each phrase occurrence marked, plus the spellings matched."""
    covered = {s + i for s in starts for i in range(length)}
    parts, spelled, current = [], [], []
    for word, positions in align_tokens(text):
        safe = html.escape(word)
        if positions and all(p in covered for p in positions):
            safe = f'<mark class="hk-match">{safe}</mark>'
            current.append(word.lower())
        elif current:
            spelled.append(" ".join(current))
            current = []
        parts.append(safe)
    if current:
        spelled.append(" ".join(current))
    return " ".join(parts), spelled


def phrase_matches(demo, mode: str, query: str):
    index = demo.indexes[mode]
    terms = index.terms_of(query)
    return terms, index.phrase(terms)


def render() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    st.subheader("Phrase search")
    st.caption("The words must appear together and in order, but each may be spelled differently. "
               "Try 'bahut accha' with H-Key V-drop to find 'bohot acha'.")
    try:
        demo = load_demo(ARTIFACTS)
    except FileNotFoundError:
        st.warning("The search indexes are not built yet. Run `python -m hinglishkey.index` once, then reload.")
        return

    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"hk_phrase_ex_{ex}", width="stretch"):
            st.session_state["hk_phrase_q"] = ex
    query = st.text_input("Phrase", key="hk_phrase_q", placeholder="e.g. bahut accha")
    label = st.radio("Normalisation", list(MODES), horizontal=True, key="hk_phrase_mode")
    mode = MODES[label]

    if not query.strip():
        st.info("Type a phrase of two or more words above, or click an example.")
        return
    try:
        words = tokenize(query)
        if not words:
            st.info("The phrase has no words to search for (only symbols, links or mentions).")
            return
        t0 = time.perf_counter()
        terms, hits = phrase_matches(demo, mode, query)
        _, raw_hits = phrase_matches(demo, "raw", query)
        ms = (time.perf_counter() - t0) * 1000
        chips = "".join(f'<span class="hk-chip">{html.escape(w)} → {html.escape(t)}</span>'
                        for w, t in zip(words, normalize_tokens(words, mode)))
        st.markdown(f"Phrase terms under {label.split(',')[0]}: {chips}", unsafe_allow_html=True)

        index = demo.indexes[mode]
        a, b, c = st.columns(3)
        a.metric(f"Tweets matched ({label.split(',')[0]})", f"{len(hits):,}")
        b.metric("Tweets matched (raw spelling only)", f"{len(raw_hits):,}")
        c.metric("Search time", f"{ms:.0f} ms")
        if not hits:
            st.info("No tweet contains this phrase in this index. Try another spelling mode.")
            return

        rows, spellings = [], Counter()
        for i, (d, starts) in enumerate(hits[:SPELL_SAMPLE]):
            tid = index.doc_ids[d]
            tweet_html, spelled = marked(demo.texts[tid], starts, len(terms))
            spellings.update(spelled)
            if i < SHOW:
                rows.append((tid, tweet_html))
        sample = f" (in the first {SPELL_SAMPLE:,} matches)" if len(hits) > SPELL_SAMPLE else ""
        st.markdown(f"**Spellings matched{sample}:** " + ", ".join(
            f"{html.escape(s)} ({n})" for s, n in spellings.most_common(8)), unsafe_allow_html=True)
        st.caption(f"Showing {len(rows)} of {len(hits):,} tweets.")
        for tid, tweet_html in rows:
            st.markdown(f'<div class="hk-tweet">{tweet_html}</div>'
                        f'<div class="hk-meta">tweet {html.escape(tid)} · {demo.splits[tid]} · '
                        f'{demo.labels[tid]}</div>', unsafe_allow_html=True)
    except Exception:                                   # never show a traceback in the demo
        st.error("Sorry, this phrase could not be processed. Please try different words.")
