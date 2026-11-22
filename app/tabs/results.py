"""Results tab: headline tables and figures, read live from results/*.json and figures/*.png.

The app shell calls render(). Nothing here is hard-coded: re-run an experiment script
(python experiments/e<N>_*.py) and reload the page to see its new numbers.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODE_NAMES = {"raw": "Raw", "soundex": "Soundex (baseline)", "hkey": "H-Key (V-norm)",
              "hkey_clusters": "H-Key + clusters", "hkey_vdrop": "H-Key V-drop (ablation)"}
E1_NAMES = {"exact": "Exact match", "soundex": "Soundex", "hkey_vdrop": "H-Key V-drop",
            "hkey_vnorm": "H-Key V-norm", "hkey_clusters": "H-Key + clusters",
            "hkey_kmeans_ablation": "H-Key k-means (ablation)"}
CLF_NAMES = {"nb": "Naive Bayes", "rocchio": "Rocchio", "knn": "kNN"}


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(name: str) -> dict:
    """Parsed results/<name>; re-read whenever the file changes. Raises FileNotFoundError."""
    path = RESULTS / name
    return _load(str(path), path.stat().st_mtime)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


@st.cache_data(show_spinner=False, max_entries=32)
def _screen_png(path: str, mtime: float, max_width: int = 1600) -> bytes:
    """A screen-sized PNG copy of a 300 dpi report figure (the file itself is not changed)."""
    import io

    from PIL import Image

    with Image.open(path) as img:
        if img.width > max_width:
            img = img.resize((max_width, round(img.height * max_width / img.width)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def warm() -> None:
    """Prepare the screen copies of all figures (called once by the shell at startup)."""
    for path in FIGURES.glob("*.png"):
        _screen_png(str(path), path.stat().st_mtime)


def figure(name: str, caption: str) -> None:
    path = FIGURES / name
    if path.exists():
        st.image(_screen_png(str(path), path.stat().st_mtime), caption=caption, width="stretch")
    else:
        st.info(f"Figure `figures/{name}` not found; run its experiment script to create it.")


def section(title: str, script: str, body) -> None:
    """Run one experiment's section; a missing or unexpected file never breaks the tab."""
    try:
        body()
    except FileNotFoundError:
        st.info(f"No results yet for {title}. Run `python {script}` to create them.")
    except (KeyError, TypeError, ValueError):
        st.warning(f"The {title} results file has an unexpected format. Re-run `python {script}`.")


def e1() -> None:
    d = load("e1_variants.json")
    view = st.radio("Gold view", ["strict (SAME only)", "lenient (SAME + INFLECTION)"],
                    horizontal=True, key="hk_res_e1_view")
    key = "strict" if view.startswith("strict") else "lenient"
    rows = [{"System": E1_NAMES.get(s, s), "Precision": f"{r[key]['precision']:.3f}",
             "Recall": f"{r[key]['recall']:.3f}", "F1": f"{r[key]['f1']:.3f}",
             "Predicted pairs": r[key]["n_pred_pairs"]}
            for s, r in d["test"].items()]
    st.markdown(f"**Pairwise P / R / F1 on the test groups** ({key})")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Settings were tuned on the dev groups only. Precision is a lower bound: pairs that "
               "cross gold groups count as false positives.")
    try:
        a = load("e1_agreement.json")
        st.caption(f"Gold: {a['n_items']:,} candidate pairs in {a['n_groups']} groups; "
                   "how it was made is described in data/README.md.")
    except FileNotFoundError:
        pass
    figure("e1_theta.png", "E1: pairwise F1 as the clustering threshold theta varies (dev)")


def e2() -> None:
    d = load("e2_sentiment.json")
    res = d["results"]
    ablations = set(d["config"].get("ablation_modes", []))
    rows = []
    for c in CLF_NAMES:
        for m in res:
            r = res[m][c]
            b = r.get("test_vs_raw_bootstrap")
            rows.append({
                "Classifier": CLF_NAMES[c], "Mode": MODE_NAMES.get(m, m),
                "Macro F1": pct(r["test"]["macro"]["f1"]), "Weighted F1": pct(r["test"]["weighted"]["f1"]),
                "Accuracy": pct(r["test"]["accuracy"]),
                "Δ macro F1 vs raw": "–" if b is None else f"{100 * b['delta_macro_f1']:+.2f}",
                "95% interval": "–" if b is None else f"[{100 * b['ci95'][0]:+.2f}, {100 * b['ci95'][1]:+.2f}]",
                "p": "–" if b is None else ("<0.001" if b["p_one_sided"] == 0 else f"{b['p_one_sided']:.3f}"),
            })
    st.markdown("**Test set (3,000 tweets), %**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    n_boot = next((res[m][c]["test_vs_raw_bootstrap"]["n_resamples"] for m in res for c in CLF_NAMES
                   if "test_vs_raw_bootstrap" in res[m][c]), None)
    st.caption("Macro F1 is the main metric; weighted F1 is the official SemEval metric. p is one-sided "
               f"(paired bootstrap{f', {n_boot:,} resamples' if n_boot else ''}), "
               "not corrected for multiple comparisons."
               + (" Ablation modes: " + ", ".join(MODE_NAMES.get(m, m) for m in ablations) + "." if ablations else ""))
    left, right = st.columns(2)
    with left:
        figure("e2_f1_bar.png", "E2: test macro F1 by classifier and normalisation mode")
    with right:
        figure("e2_confusion.png", "E2: test confusion matrices (rows = true class)")
    st.markdown("**Comparison with published SemEval-2020 Task 9 Hinglish systems** (weighted F1, %)")
    pub = [{"System": p["system"], "Weighted F1": f"{p['weighted_f1']:.1f}", "Source": p["paper"]}
           for p in d["published_reference"]]
    best = max(((c, m) for c in CLF_NAMES for m in res if m not in ablations),
               key=lambda cm: res[cm[1]][cm[0]]["dev"]["macro"]["f1"])
    pub.append({"System": f"Ours: {CLF_NAMES[best[0]]} on {MODE_NAMES.get(best[1], best[1])}",
                "Weighted F1": pct(res[best[1]][best[0]]["test"]["weighted"]["f1"]), "Source": "E2"})
    st.dataframe(pd.DataFrame(pub), hide_index=True, width="stretch")
    st.caption("Our row is the configuration with the best dev macro F1 (chosen without the test set).")


def e3() -> None:
    d = load("e3_retrieval.json")
    rows = [{"Index": MODE_NAMES.get(m, m), "MRR": f"{r['MRR']:.3f}", "P@1": f"{r['P@1']:.2f}",
             "R@5": f"{r['R@5']:.2f}", "R@10": f"{r['R@10']:.2f}", "MAP": f"{r['MAP']:.3f}",
             "Not found": r["n_not_retrieved"]}
            for m, r in d["results"].items()]
    st.markdown(f"**Known-item retrieval: {d['queries']['n']} respelled queries over "
                f"{d['corpus']['docs']:,} tweets**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("One relevant tweet per query, so MAP equals MRR.")
    try:
        b = load("e3_bootstrap.json")
        brow = [{"Comparison": f"{MODE_NAMES.get(x['a'], x['a'])} − {MODE_NAMES.get(x['b'], x['b'])}",
                 "Δ MRR": f"{x['delta_mrr']:+.3f}", "95% interval": f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]",
                 "p": "<0.001" if x["p"] == 0 else f"{x['p']:.3f}", "Test": x["sided"]}
                for x in b["vs_raw"] + b["pairs"]]
        st.markdown("**Paired bootstrap** (comparisons fixed before computing)")
        st.dataframe(pd.DataFrame(brow), hide_index=True, width="stretch")
        mc = b["multiple_comparisons"]
        st.caption(f"{b['n_resamples']:,} resamples of the queries, seed {b['seed']}. Uncorrected; the "
                   f"Bonferroni threshold for all {mc['n_tests']} tests is {mc['bonferroni_alpha_all']:.4f}.")
    except FileNotFoundError:
        pass
    figure("e3_recall_at_k.png", "E3: recall at k for each index")


def e4() -> None:
    d = load("e4_efficiency.json")
    rows = [{"Mode": MODE_NAMES.get(m, m), "Vocabulary": f"{r['vocab_size']:,}",
             "vs raw": f"{100 * r['vocab_vs_raw']:.1f}%", "Postings": f"{r['n_postings']:,}",
             "Index MB": f"{r['pickle_bytes'] / 1e6:.2f}",
             "Query ms (median / p95 / max)": f"{r['query_ms']['median']:.1f} / {r['query_ms']['p95']:.1f} / "
                                              f"{r['query_ms']['max']:.1f}",
             "Heaps b": f"{r['heaps']['b']:.3f}"}
            for m, r in d["results"].items()]
    st.markdown(f"**Efficiency over all {d['corpus']['docs']:,} tweets**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Latency: {d['latency_protocol']}")
    figure("e4_vocab_growth.png", "E4: vocabulary growth (Heaps' law)")


def files() -> None:
    paths = sorted(RESULTS.glob("*.json"))
    if not paths:
        st.info("No results files yet.")
        return
    st.dataframe(pd.DataFrame([{"File": f"results/{p.name}"} for p in paths]), hide_index=True, width="stretch")
    choice = st.selectbox("Show a raw results file", [p.name for p in paths], index=None,
                          placeholder="Choose a file", key="hk_res_raw")
    if choice:
        st.json(load(choice), expanded=1)


def render() -> None:
    st.subheader("Results")
    st.caption("Headline numbers and figures, read live from results/ and figures/. "
               "Re-run an experiment script and reload to update them.")
    parts = [("E1 · Variant grouping", "experiments/e1_variants.py", e1),
             ("E2 · Sentiment", "experiments/e2_sentiment.py", e2),
             ("E3 · Retrieval", "experiments/e3_retrieval.py", e3),
             ("E4 · Efficiency", "experiments/e4_efficiency.py", e4),
             ("All files", "", files)]
    tabs = st.tabs([name for name, _, _ in parts], key="hk_res_tabs", on_change="rerun")
    for tab, (name, script, body) in zip(tabs, parts):
        if tab.open:
            with tab:
                section(name.split(" · ")[-1], script, body)
