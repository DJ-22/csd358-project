"""Merge two annotators' variant labels, report agreement, prepare adjudication.

    python experiments/merge_gold.py data/gold/variants_gold_A.tsv data/gold/variants_gold_B.tsv

1. Checks that both files contain exactly the rows of candidates.tsv, in
   order, each with a label in {SAME, INFLECTION, DIFFERENT}.
2. Prints observed agreement and Cohen's kappa (L6 inter-annotator
   agreement), for the 3 labels and for the two binary views used by E1:
   strict (SAME vs rest) and lenient (SAME or INFLECTION vs DIFFERENT).
3. Writes data/gold/variants_gold.tsv: agreed rows keep the label,
   disagreements get an empty label for joint adjudication.
   Never overwrites an existing gold file (adjudication would be lost).
4. Writes the agreement numbers to results/e1_agreement.json.
5. Writes label-only copies of both inputs (variants_labels_A.tsv and
   variants_labels_B.tsv: group_id, headword, candidate, label, note) next
   to the gold file. They hold no tweet text, so they can be committed, and
   this script accepts them as inputs to reproduce the kappa values.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data" / "gold"
LABELS = ("SAME", "INFLECTION", "DIFFERENT")
KEY_COLUMNS = ("group_id", "headword", "candidate")
OUT_COLUMNS = ["group_id", "headword", "candidate", "label", "label_a", "label_b",
               "note_a", "note_b"]
LABEL_ONLY_COLUMNS = ["group_id", "headword", "candidate", "label", "note"]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa = (P(A) - P(E)) / (1 - P(E))   (L6, IIR section 8.1).

    P(A) = observed agreement = fraction of items with the same label.
    P(E) = chance agreement = sum over labels k of p_a(k) * p_b(k), where
           p_x(k) is the fraction of items annotator x gave label k.
    If P(E) = 1 (both annotators used one single label), kappa is 1.0.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    n = len(a)
    p_agree = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    p_chance = sum(count_a[k] * count_b[k] for k in set(a) | set(b)) / (n * n)
    if p_chance == 1:
        return 1.0
    return (p_agree - p_chance) / (1 - p_chance)


def agreement(a: list[str], b: list[str]) -> dict:
    return {"observed": sum(x == y for x, y in zip(a, b)) / len(a),
            "kappa": cohen_kappa(a, b)}


def strict(labels):
    return ["SAME" if x == "SAME" else "NOT_SAME" for x in labels]


def lenient(labels):
    return ["VARIANT" if x in ("SAME", "INFLECTION") else "DIFFERENT" for x in labels]


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_labels(path: Path, reference: list[dict]) -> list[dict]:
    """Rows of one annotator's file, checked against candidates.tsv."""
    rows = read_tsv(path)
    problems = []
    if len(rows) != len(reference):
        problems.append(f"{len(rows)} rows, expected {len(reference)}")
    for i, (row, ref) in enumerate(zip(rows, reference), start=2):   # line 1 = header
        if any((row.get(c) or "").strip() != ref[c] for c in KEY_COLUMNS):
            problems.append(f"line {i}: row differs from candidates.tsv "
                            f"({row.get('group_id')}, {row.get('headword')}, {row.get('candidate')})")
        label = (row.get("label") or "").strip().upper()
        if label not in LABELS:
            problems.append(f"line {i}: label {row.get('label')!r} is not one of {LABELS}")
        row["label"] = label
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        sys.exit(f"{path}:\n  {shown}{more}")
    return rows


def write_label_only(rows: list[dict], path: Path) -> None:
    """Copy of one annotator's labels without the tweet-context columns."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_ONLY_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({"group_id": r["group_id"], "headword": r["headword"],
                        "candidate": r["candidate"], "label": r["label"],
                        "note": (r.get("note") or "").strip()})


def main():
    ap = argparse.ArgumentParser(description="Merge two variant-label files and report Cohen's kappa.")
    ap.add_argument("file_a", type=Path)
    ap.add_argument("file_b", type=Path)
    ap.add_argument("--candidates", type=Path, default=GOLD_DIR / "candidates.tsv")
    ap.add_argument("--out", type=Path, default=GOLD_DIR / "variants_gold.tsv")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "e1_agreement.json")
    args = ap.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} already exists; it may hold adjudicated labels, so it is not overwritten. "
                 f"Move it away first if you really want to rebuild it.")

    reference = read_tsv(args.candidates)
    rows_a = read_labels(args.file_a, reference)
    rows_b = read_labels(args.file_b, reference)
    a = [r["label"] for r in rows_a]
    b = [r["label"] for r in rows_b]

    report = {
        "n_items": len(a),
        "n_groups": len({r["group_id"] for r in reference}),
        "three_labels": agreement(a, b),
        "strict_binary": agreement(strict(a), strict(b)),
        "lenient_binary": agreement(lenient(a), lenient(b)),
        "label_counts_a": {k: a.count(k) for k in LABELS},
        "label_counts_b": {k: b.count(k) for k in LABELS},
        "confusion_a_rows_b_cols": {x: {y: sum(1 for p, q in zip(a, b) if (p, q) == (x, y))
                                        for y in LABELS} for x in LABELS},
        "n_disagreements": sum(x != y for x, y in zip(a, b)),
        "files": {"a": str(args.file_a), "b": str(args.file_b)},
    }

    print(f"items: {report['n_items']} in {report['n_groups']} groups")
    for view in ("three_labels", "strict_binary", "lenient_binary"):
        r = report[view]
        print(f"{view:15s} observed agreement {r['observed']:.3f}   Cohen's kappa {r['kappa']:.3f}")
    print("confusion (rows = A, cols = B):")
    print(" " * 12 + "".join(f"{y:>12s}" for y in LABELS))
    for x in LABELS:
        print(f"{x:>12s}" + "".join(f"{report['confusion_a_rows_b_cols'][x][y]:12d}" for y in LABELS))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for ra, rb in zip(rows_a, rows_b):
            w.writerow({"group_id": ra["group_id"], "headword": ra["headword"],
                        "candidate": ra["candidate"],
                        "label": ra["label"] if ra["label"] == rb["label"] else "",
                        "label_a": ra["label"], "label_b": rb["label"],
                        "note_a": (ra.get("note") or "").strip(),
                        "note_b": (rb.get("note") or "").strip()})
    label_only = {}
    for name, rows, src in (("A", rows_a, args.file_a), ("B", rows_b, args.file_b)):
        path = args.out.parent / f"variants_labels_{name}.tsv"
        if path.resolve() != src.resolve():          # inputs may already be the label-only copies
            write_label_only(rows, path)
        label_only[name] = path
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out} ({report['n_disagreements']} rows to adjudicate), "
          f"{label_only['A'].name}, {label_only['B'].name} and {args.results}")


if __name__ == "__main__":
    main()
