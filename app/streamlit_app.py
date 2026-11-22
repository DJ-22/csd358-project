"""HinglishKey demo: the app shell.

    streamlit run app/streamlit_app.py

Before the first run, build the precomputed artifacts (about 30 s, CPU only):

    PYTHONPATH=src python -m hinglishkey.index            # search indexes + variant model
    PYTHONPATH=src python -m hinglishkey.classify.demo    # sentiment models

The shell sets the page up (title bar, sidebar with dataset statistics, shared styling),
loads every artifact once per process (st.cache_resource) and calls each tab's render().
Only the open tab runs on each interaction. Any error inside a tab is caught here, so the
demo never shows a traceback.
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
for p in (ROOT / "src", APP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hinglishkey.classify.demo import load_models  # noqa: E402
from hinglishkey.index import load_demo             # noqa: E402
from tabs import phrase, results, search, sentiment, variants  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
BUILD_HELP = ("Build the demo files once (about 30 s), then reload:\n\n"
              "```\n$env:PYTHONPATH = \"src\"\n"
              "python -m hinglishkey.index\n"
              "python -m hinglishkey.classify.demo\n```")

# Shared colours. Sentiment: green / grey / red. Matched variants: --hk-match (used by every tab).
THEMES = {
    "light": {"match": "#fde68a", "pos": "#2f9e44", "neu": "#7c818a", "neg": "#e03131",
              "track": "rgba(0,0,0,0.07)"},
    "dark": {"match": "#8a6a12", "pos": "#51cf66", "neu": "#a5abb4", "neg": "#ff6b6b",
             "track": "rgba(255,255,255,0.10)"},
}


def shared_style() -> str:
    try:
        mode = "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        mode = "light"
    t = THEMES[mode]
    return f"""
<style>
:root {{ --hk-match: {t['match']}; --hk-pos: {t['pos']}; --hk-neu: {t['neu']}; --hk-neg: {t['neg']};
         --hk-track: {t['track']}; }}
.block-container {{ padding-top: 4rem; padding-bottom: 2rem; }}
/* p.hk-* plus !important: Streamlit's own markdown paragraph style would otherwise win */
p.hk-title {{ font-size: 2.1rem !important; font-weight: 700; margin: 0 !important; line-height: 1.35 !important;
              padding-bottom: 0.1rem; }}
p.hk-tagline {{ font-size: 0.98rem !important; opacity: 0.75; margin: 0.15rem 0 0.4rem 0 !important; }}
.hk-chip {{ display: inline-block; font-family: monospace; font-size: 0.82rem; padding: 1px 6px;
            margin: 0 6px 4px 0; border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; }}
.hk-chip-off {{ opacity: 0.5; text-decoration: line-through; }}
.hk-badge {{ display: inline-block; font-weight: 700; font-size: 1.05rem; padding: 3px 12px;
             border-radius: 6px; color: #ffffff; }}
.hk-badge.positive {{ background: var(--hk-pos); }}
.hk-badge.neutral {{ background: var(--hk-neu); }}
.hk-badge.negative {{ background: var(--hk-neg); }}
.hk-bar-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.18rem 0; font-size: 0.9rem; }}
.hk-bar-label {{ width: 5.6rem; flex-shrink: 0; }}
.hk-bar-track {{ flex: 1; background: var(--hk-track); border-radius: 4px; height: 0.95rem; }}
.hk-bar-fill {{ height: 100%; border-radius: 4px; }}
.hk-bar-fill.positive {{ background: var(--hk-pos); }}
.hk-bar-fill.neutral {{ background: var(--hk-neu); }}
.hk-bar-fill.negative {{ background: var(--hk-neg); }}
.hk-bar-value {{ width: 4.2rem; text-align: right; font-variant-numeric: tabular-nums; flex-shrink: 0; }}
</style>
"""


@st.cache_resource(show_spinner="Loading the search indexes (first start only)...")
def get_demo():
    return load_demo(ARTIFACTS)          # raises FileNotFoundError if not built; failures are not cached


@st.cache_resource(show_spinner="Loading the sentiment models (first start only)...")
def get_models():
    return load_models(ARTIFACTS)


@st.cache_resource(show_spinner=False)
def warm_libraries() -> bool:
    import altair  # noqa: F401  (Streamlit charts import it lazily; pay that once at startup)
    return True


def load_artifacts() -> list[str]:
    """Warm both caches before the first click. Returns the names of missing artifact sets."""
    warm_libraries()
    try:
        results.warm()                   # screen-sized copies of the report figures
    except Exception:
        pass                             # the Results tab shows its own message per figure
    missing = []
    for name, loader in (("search indexes", get_demo), ("sentiment models", get_models)):
        try:
            loader()
        except FileNotFoundError:
            missing.append(name)
        except Exception:
            missing.append(f"{name} (could not be read; rebuild them)")
    if "search indexes" not in missing:
        try:                             # the Variant explorer's lookup tables, built once per process
            demo = get_demo()
            variants.classes(demo)
            variants.key_groups(demo, "norm")
            variants.key_groups(demo, "drop")
        except Exception:
            pass                         # the tab builds them itself on first use
    return missing


def read_json(name: str) -> dict | None:
    try:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))
    except Exception:
        return None


def sidebar(missing: list[str]) -> None:
    with st.sidebar:
        st.markdown("### Dataset")
        st.caption("SentiMix Hinglish (SemEval-2020 Task 9), official splits. CC BY 4.0.")
        e2 = read_json("e2_sentiment.json")
        if e2:
            sizes = e2["config"]["split_sizes"]
            st.metric("Tweets", f"{sum(sizes.values()):,}")
            st.caption(" · ".join(f"{s} {n:,}" for s, n in sizes.items()))
        e4 = read_json("e4_efficiency.json")
        if e4:
            st.markdown("### Vocabulary")
            st.caption(f"Distinct index terms over all {e4['corpus']['docs']:,} tweets (E4).")
            names = {"raw": "Raw", "hkey": "H-Key (V-norm)", "hkey_clusters": "H-Key + clusters",
                     "hkey_vdrop": "H-Key V-drop", "soundex": "Soundex"}
            rows = [{"Mode": names[m], "Terms": f"{r['vocab_size']:,}",
                     "vs raw": f"{100 * r['vocab_vs_raw']:.0f}%"}
                    for m, r in e4["results"].items() if m in names]
            st.dataframe(rows, hide_index=True, width="stretch")
        if missing:
            st.warning("Missing: " + ", ".join(missing) + ".\n\n" + BUILD_HELP)
        st.caption("Runs offline on a CPU. Every model is precomputed from the training split.")


def run_tab(render) -> None:
    try:
        render()
    except Exception:                    # last line of defence: never show a traceback
        st.error("Sorry, something went wrong in this tab. Please try again or switch tabs.")


def main() -> None:
    st.set_page_config(page_title="HinglishKey", page_icon=":material/translate:", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(shared_style(), unsafe_allow_html=True)
    st.markdown('<p class="hk-title">HinglishKey</p>'
                '<p class="hk-tagline">Spelling-robust search and sentiment classification for '
                'code-mixed Hinglish tweets, using phonetic equivalence classes (H-Key).</p>',
                unsafe_allow_html=True)
    missing = load_artifacts()
    sidebar(missing)

    pages = [("Search", search.render), ("Phrase search", phrase.render),
             ("Sentiment", sentiment.render), ("Variant explorer", variants.render),
             ("Results", results.render)]
    tabs = st.tabs([name for name, _ in pages], key="hk_nav", on_change="rerun")
    for tab, (_, render) in zip(tabs, pages):
        if tab.open:                     # lazy: only the selected tab runs
            with tab:
                run_tab(render)


main()
