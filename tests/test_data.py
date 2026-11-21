"""Tests for data.py: parser on a synthetic file + checks on the real SentiMix release."""

from collections import Counter

import pytest

from hinglishkey import data
from hinglishkey.data import LABELS, fix_mojibake, load_split, parse_conll, read_test_labels

SAMPLE = (
    "meta\t101\tpositive\n"
    "bahut\tHin\n"
    "accha\tHin\n"
    "movie\tEng\n"
    "\n"
    "meta\t102\tNegative\n"
    "bohot\tHin\n"
    "bura\tHin\n"
    "\n"
)


def test_parse_conll_synthetic(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    assert parse_conll(p) == [("101", ["bahut", "accha", "movie"], "positive"),
                              ("102", ["bohot", "bura"], "negative")]


def test_parse_test_file_and_labels(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("meta\t7\nacha\tHin\n\nmeta\t8\nok\tEng\n", encoding="utf-8")
    assert parse_conll(p) == [("7", ["acha"], None), ("8", ["ok"], None)]
    lab = tmp_path / "labels.txt"
    lab.write_text("Uid,Sentiment\n7,positive\n8,neutral\n", encoding="utf-8")
    assert read_test_labels(lab) == {"7": "positive", "8": "neutral"}


def test_fix_mojibake():
    assert fix_mojibake("ðŸ˜…") == "😅"
    assert fix_mojibake("â€¦") == "…"
    assert fix_mojibake("â\x9d¤") == "❤"          # 0x9D is undefined in cp1252
    for clean in ("bahut", "accha", "😅", "नमस्ते", "café"):
        assert fix_mojibake(clean) == clean       # real text is left alone


def test_bad_split_name():
    with pytest.raises(ValueError):
        load_split("validation")


# --- Real data: sizes and label counts from Patwa et al. (2020), Table 1 ---------------

EXPECTED = {
    "train": {"positive": 4634, "neutral": 5264, "negative": 4102},
    "dev": {"positive": 982, "neutral": 1128, "negative": 890},
    "test": {"positive": 1000, "neutral": 1100, "negative": 900},
}

needs_data = pytest.mark.skipif(not (data.RAW_DIR / data.SPLIT_FILES["train"]).exists(),
                                reason="SentiMix raw files not downloaded (see data/README.md)")


@needs_data
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_split_sizes_match_overview_paper(split):
    rows = load_split(split)
    counts = Counter(label for _, _, label in rows)
    assert dict(counts) == EXPECTED[split]
    assert all(label in LABELS for _, _, label in rows)
    assert all(text.strip() for _, text, _ in rows)


@needs_data
def test_tweet_ids_unique_and_splits_disjoint():
    ids = {s: [uid for uid, _, _ in load_split(s)] for s in ("train", "dev", "test")}
    for s, lst in ids.items():
        assert len(lst) == len(set(lst)), f"duplicate ids in {s}"
    assert not set(ids["train"]) & set(ids["dev"])
    assert not set(ids["train"]) & set(ids["test"])
    assert not set(ids["dev"]) & set(ids["test"])


@needs_data
def test_test_split_has_real_emoji_after_repair():
    # The raw test file is double-encoded; after loading it should look like train (real emoji).
    test_text = " ".join(text for _, text, _ in load_split("test"))
    assert "😅" in test_text or "😂" in test_text
    assert "ðŸ" not in test_text
