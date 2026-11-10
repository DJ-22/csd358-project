"""SentiMix Hinglish loader.

Source: SemEval-2020 Task 9 (Patwa et al. 2020), Zenodo record 3974927, CC BY 4.0.
See data/README.md for the download steps. The raw files are NOT committed.

Raw format (CoNLL-style, one token per line, tweets separated by blank lines):
    meta    <uid>   <label>        <- train/dev; the test file has "meta <uid>" only
    <token> <lang_tag>             <- lang_tag in {Hin, Eng, O, EMT}
Test labels live in a separate CSV with header "Uid,Sentiment".

Shared by all experiments, so LABELS and load_split's signature must stay fixed.
"""

from __future__ import annotations

import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

LABELS = ("negative", "neutral", "positive")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "Semeval_2020_task9_data" / "Hinglish"
SPLIT_FILES = {
    "train": "Hinglish_train_14k_split_conll.txt",
    "dev": "Hinglish_dev_3k_split_conll.txt",
    "test": "Hinglish_test_unalbelled_conll_updated.txt",   # "unalbelled" is the official file name
}
TEST_LABELS_FILE = "Hinglish_test_labels.txt"


def parse_conll(path: Path) -> list[tuple[str, list[str], str | None]]:
    """Parse a SentiMix CoNLL file into (uid, tokens, label-or-None) per tweet."""
    tweets = []
    uid, label, tokens = None, None, []

    def flush():
        if uid is not None:
            tweets.append((uid, tokens, label))

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue                               # blank line between tweets
            parts = line.split("\t")
            if parts[0] == "meta":                     # start of a new tweet
                flush()
                uid = parts[1].strip()
                label = parts[2].strip().lower() if len(parts) > 2 and parts[2].strip() else None
                tokens = []
            elif uid is not None and parts[0] != "":
                tokens.append(parts[0])                # keep the token, drop the language tag
    flush()
    return tweets


def fix_mojibake(token: str) -> str:
    """Undo double encoding: UTF-8 bytes that were wrongly decoded as cp1252 ("ðŸ˜…" -> "😅").

    Only the official TEST file has this problem (train/dev contain real emoji). Without the
    repair, every test emoji/Devanagari token would be a junk string never seen in training.
    Bytes that cp1252 leaves undefined (0x81, 0x8D, 0x8F, 0x90, 0x9D) survive as control
    characters U+0080..U+009F, so they are mapped straight back to their byte value.
    If the result is not valid UTF-8, the token was not mojibake and is returned unchanged.
    """
    raw = bytearray()
    for ch in token:
        try:
            raw += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:
                raw.append(ord(ch))
            else:
                return token
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return token


def read_test_labels(path: Path) -> dict[str, str]:
    """Read the "Uid,Sentiment" CSV released after the competition."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row["Uid"].strip(): row["Sentiment"].strip().lower() for row in csv.DictReader(f)}


@lru_cache(maxsize=None)
def _load(split: str, raw_dir: Path) -> tuple[tuple[str, str, str], ...]:
    tweets = parse_conll(raw_dir / SPLIT_FILES[split])
    if split == "test":
        gold = read_test_labels(raw_dir / TEST_LABELS_FILE)
        tweets = [(uid, [fix_mojibake(t) for t in tokens], gold.get(uid)) for uid, tokens, _ in tweets]

    rows = []
    for uid, tokens, label in tweets:
        if label not in LABELS:
            raise ValueError(f"{split}: tweet {uid} has label {label!r}, expected one of {LABELS}")
        rows.append((uid, " ".join(tokens), label))
    return tuple(rows)


def load_split(split: str) -> list[tuple[str, str, str]]:
    """split in {"train","dev","test"}; returns (tweet_id, text, label).

    text = the tweet's tokens joined by single spaces, as in the official release, except that
    the test split's double-encoded characters are repaired (fix_mojibake). No other
    normalisation happens here; that is normalize.py's job.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {tuple(SPLIT_FILES)}, got {split!r}")
    if not (RAW_DIR / SPLIT_FILES[split]).exists():
        raise FileNotFoundError(f"{RAW_DIR / SPLIT_FILES[split]} not found; see data/README.md to download SentiMix")
    return list(_load(split, RAW_DIR))


if __name__ == "__main__":
    # Prints split sizes and label distribution (the numbers recorded in data/README.md).
    for name in SPLIT_FILES:
        rows = load_split(name)
        counts = Counter(label for _, _, label in rows)
        print(f"{name:5s} n={len(rows):5d}  " + "  ".join(f"{c}={counts[c]}" for c in LABELS))
