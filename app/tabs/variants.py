"""Variant explorer tab: one word's codes and its spelling-variant class.

Shows the word's H-Key (V-norm and V-drop), its Soundex code, the variant
class it belongs to (learned from the training tweets) with each member's
frequency, and the training words that share each key. The app shell calls
render().
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hinglishkey.hkey import hkey                          # noqa: E402
from hinglishkey.index import load_demo                    # noqa: E402
from hinglishkey.normalize import soundex                   # noqa: E402
from hinglishkey.tokenize import tokenize                   # noqa: E402

ARTIFACTS = ROOT / "artifacts"
EXAMPLES = ["bahut", "accha", "pyaar", "zindagi", "thik"]
MAX_ROWS = 15

def _cache(demo) -> dict:
    """A dict stored on the shared demo object, so it survives reruns and reimports."""
    if not hasattr(demo, "tab_cache"):
        demo.tab_cache = {}
    return demo.tab_cache


def classes(demo) -> dict[str, list[str]]:
    """Variant classes (canonical -> members), computed once per process."""
    cache = _cache(demo)
    if "classes" not in cache:
        cache["classes"] = demo.variants.classes()
    return cache["classes"]


def key_groups(demo, vowels: str) -> dict[str, list[str]]:
    """H-Key (given vowel rule) -> training words with that key, most frequent first."""
    cache = _cache(demo)
    if ("key_groups", vowels) not in cache:
        groups: dict[str, list[str]] = {}
        for w in demo.train_freq:
            if w.isascii() and w.isalpha():
                groups.setdefault(hkey(w, vowels=vowels), []).append(w)
        for ws in groups.values():
            ws.sort(key=lambda w: (-demo.train_freq[w], w))
        cache[("key_groups", vowels)] = groups
    return cache[("key_groups", vowels)]


def word_table(demo, words: list[str], target: str) -> list[dict]:
    return [{"word": w, "frequency in train": demo.train_freq[w], "": "◀ this word" if w == target else ""}
            for w in words[:MAX_ROWS]]


def render() -> None:
    st.subheader("Variant explorer")
    st.caption("See how one word is keyed and which spellings it is grouped with.")
    try:
        demo = load_demo(ARTIFACTS)
    except FileNotFoundError:
        st.warning("The indexes are not built yet. Run `python -m hinglishkey.index` once, then reload.")
        return

    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"hk_var_ex_{ex}", width="stretch"):
            st.session_state["hk_var_q"] = ex
    raw = st.text_input("Word", key="hk_var_q", placeholder="e.g. bahut")
    classes(demo)                                   # warm the caches before the first lookup
    key_groups(demo, "norm")
    key_groups(demo, "drop")

    if not raw.strip():
        st.info("Type one word above, or click an example.")
        return
    try:
        toks = tokenize(raw)
        if not toks:
            st.info("That has no letters to look up.")
            return
        word = toks[0]
        if len(toks) > 1:
            st.caption(f"Showing the first word only: {word}")
        if not (word.isascii() and word.isalpha()):
            st.info("H-Key works on romanised words (a–z). Numbers and other scripts have no key.")
            return

        norm_key, drop_key = hkey(word, vowels="norm"), hkey(word, vowels="drop")
        a, b, c = st.columns(3)
        a.metric("H-Key (V-norm)", norm_key)
        b.metric("H-Key (V-drop)", drop_key)
        c.metric("Soundex", soundex(word))

        canon = demo.variants.canonical(word)
        members = classes(demo).get(canon, [canon])
        seen = demo.train_freq[word] > 0
        st.markdown(f"**Variant class: `{canon}`** "
                    f"({len(members)} spelling{'s' if len(members) != 1 else ''}, learned from the training tweets)")
        if not seen:
            st.caption(f"'{word}' never occurs in the training tweets. "
                       + ("It is assigned to its closest variant class." if canon != word
                          else "No training spelling is close enough, so it forms its own class."))
        st.dataframe(word_table(demo, members, word), hide_index=True, width="stretch")
        if len(members) > MAX_ROWS:
            st.caption(f"Showing the {MAX_ROWS} most frequent of {len(members)} spellings.")

        left, right = st.columns(2)
        same_norm = key_groups(demo, "norm").get(norm_key, [])
        same_drop = key_groups(demo, "drop").get(drop_key, [])
        with left:
            st.markdown(f"**Same V-norm key** `{norm_key}`: {len(same_norm)} word(s)")
            if same_norm:
                st.dataframe(word_table(demo, same_norm, word), hide_index=True, width="stretch")
        with right:
            st.markdown(f"**Same V-drop key** `{drop_key}`: {len(same_drop)} word(s)")
            if same_drop:
                st.dataframe(word_table(demo, same_drop, word), hide_index=True, width="stretch")
        st.caption("V-drop keys are coarser: they join more true variants (e.g. bahut/bohot) "
                   "but also more unrelated words. The variant class keeps only V-drop key-mates "
                   "whose spellings are similar enough (cosine ≥ 0.85).")
    except Exception:                                   # never show a traceback in the demo
        st.error("Sorry, this word could not be looked up. Please try another.")
