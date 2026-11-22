"""Smoke test of the demo app, run headless: every tab opens and the Sentiment tab handles
every classifier and an awkward input without an exception or an error box.
Skipped when the demo artifacts have not been built."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
TABS = ["Search", "Phrase search", "Sentiment", "Variant explorer", "Results"]


@pytest.fixture(scope="module")
def app():
    if not (ROOT / "artifacts" / "sentiment_models.pkl").exists() or \
            not (ROOT / "artifacts" / "index_raw.pkl").exists():
        pytest.skip("demo artifacts not built (python -m hinglishkey.index / hinglishkey.classify.demo)")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("tab", TABS)
def test_every_tab_opens(app, tab):
    app.session_state["hk_nav"] = tab
    app.run()
    assert_clean(app)
    assert [s.value for s in app.subheader][0] == tab


@pytest.mark.parametrize("clf", ["Naive Bayes", "Rocchio", "kNN"])
@pytest.mark.parametrize("text", ["bohot accha laga yaar", "!!! 😂 https://t.co/x", "zzqqxx"])
def test_sentiment_inputs(app, clf, text):
    app.session_state["hk_nav"] = "Sentiment"
    app.session_state["hk_sent_clf"] = clf
    app.session_state["hk_sent_mode"] = "H-Key (V-norm), default"
    app.session_state["hk_sent_text"] = text
    app.run()
    assert_clean(app)
