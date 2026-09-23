"""Flag caption-level issues in a sampled CFPE audit; never approve visual labels."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def review(row):
    flags = []
    roles = ("reference", "positive", "cf_a", "cf_b")
    captions = {r: row.get(r + "_caption", "").lower() for r in roles}
    if row["object"] == "flower":
        for role in roles[:3]:
            if re.search(r"\bflowers?\s+(pots?|pottery|vases?|beds?)\b", captions[role]):
                flags.append(("flower_compound", role))
        if re.search(r"\b(roses?|tulips?|lilies|lily|irises|iris)\b", captions["cf_b"]):
            flags.append(("cf_b_flower_subtype_needs_review", "cf_b"))
    if row["object"] == "iris" and any(re.search(r"\b(eye|eyes|pupil)\b", c) for c in captions.values()):
        flags.append(("iris_word_sense", "tuple"))
    for role, caption in captions.items():
        if re.search(r"\b(tree lights?|chicken eggs?|bird cages?|dog collars?)\b", caption):
            flags.append(("other_noun_compound_needs_review", role))
    media = {r: bool(re.search(r"\b(vector|illustration|cartoon|typography|drawing|seamless pattern)\b", c))
             for r, c in captions.items()}
    if media["reference"] != media["positive"]:
        flags.append(("possible_media_mismatch", "reference_positive"))
    return flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    with a.audit.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    counters = {s: Counter() for s in {r["split"] for r in rows}}
    selected = []
    for csv_line, row in enumerate(rows, 2):
        flags = review(row)
        for category in {k for k, _ in flags}:
            counters[row["split"]][category] += 1
        if flags:
            selected.append(dict(audit_csv_line=csv_line, split=row["split"],
                source_csv_line=row["source_csv_line"], flags=flags, row=row))
    summary = dict(source=str(a.audit.resolve()), source_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest(),
        audit_rows=len(rows), rows_by_split=dict(Counter(r["split"] for r in rows)),
        attributes_by_split={s: dict(Counter(r["attribute_type"] for r in rows if r["split"] == s))
                             for s in counters},
        flagged_rows=len(selected), flagged_rows_by_split=dict(Counter(r["split"] for r in selected)),
        flags_by_split={s: dict(c) for s, c in counters.items()},
        visual_audit_completed=False,
        limitations=["Automated caption screening of all audit rows; not manual visual approval.",
            "Flags overlap and are not measured label error rates. Unflagged rows are not approved.",
            "The audit has equal train/validation samples, not the same proportions as the full dataset."],
        flagged_examples=selected)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "caption_review.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {k: v for k, v in summary.items() if k != "flagged_examples"}
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"Report: {a.output_dir / 'caption_review.json'}")


if __name__ == "__main__":
    main()
