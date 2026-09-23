"""Read-only diagnostics for existing pseudo-edit splits; no GPU required."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROLES = ("reference", "positive", "cf_a", "cf_b")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*ROLES, "modification", "attribute_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns: {path}")
        rows = list(reader)
    if not rows or any(not row.get(key, "").strip() for row in rows for key in required):
        raise ValueError(f"Empty manifest or incomplete rows: {path}")
    return rows


def identity(name, image_root):
    path = (image_root / name).resolve()
    if not path.is_relative_to(image_root):
        raise ValueError(f"Image outside image root: {name}")
    return str(path)


def flags(row):
    """Heuristic review flags, not automatic proof of incorrect visual labels."""
    found = []
    if row["reference"] == row["cf_a"]:
        found.append("cf_a_is_reference")
    for role, target in (("reference", row.get("before", "")),
                         ("positive", row.get("after", ""))):
        caption = row.get(f"{role}_caption", "").lower()
        if target in {"long", "short"} and re.search(
            r"\b" + re.escape(target) + r"\s+(enough|time|ago|while|period)\b", caption
        ):
            found.append(f"{role}_possible_temporal_size_word")
        if re.search(r"\b(no|not|without)\b", caption):
            found.append(f"{role}_negation_needs_review")
    return found


def diagnose(train, val, image_root):
    role_sets = {
        split: {role: {identity(row[role], image_root) for row in rows} for role in ROLES}
        for split, rows in (("train", train), ("val", val))
    }
    train_images = set().union(*role_sets["train"].values())
    val_images = set().union(*role_sets["val"].values())
    overlap = train_images & val_images
    summary = {}
    for split, rows in (("train", train), ("val", val)):
        references = Counter(identity(row["reference"], image_root) for row in rows)
        tuple_keys = [tuple(identity(row[r], image_root) for r in ROLES) +
                      (row["modification"],) for row in rows]
        summary[split] = {
            "tuples": len(rows),
            "unique_images": len(set().union(*role_sets[split].values())),
            "unique_references": len(references),
            "largest_reference_share_percent": 100 * max(references.values()) / len(rows),
            "top_10_references": references.most_common(10),
            "attribute_counts": dict(Counter(row["attribute_type"] for row in rows)),
            "duplicate_tuples": len(tuple_keys) - len(set(tuple_keys)),
            "cf_a_equals_reference": sum(identity(r["cf_a"], image_root) ==
                                          identity(r["reference"], image_root) for r in rows),
            "positive_equals_negative": sum(any(identity(r["positive"], image_root) ==
                identity(r[role], image_root) for role in ("cf_a", "cf_b")) for r in rows),
            "review_flag_counts": dict(Counter(flag for r in rows for flag in flags(r))),
        }
    touched = sum(any(identity(row[role], image_root) in train_images for role in ROLES)
                  for row in val)
    missing = sorted(p for p in train_images | val_images if not Path(p).is_file())
    return {
        "splits": summary,
        "cross_split": {
            "shared_image_paths": len(overlap),
            "validation_tuples_with_any_train_image": touched,
            "validation_tuples_with_any_train_image_percent": 100 * touched / len(val),
            "overlap_by_role": {a: {b: len(role_sets["train"][a] & role_sets["val"][b])
                for b in ROLES} for a in ROLES},
            "overlap_examples": sorted(overlap)[:20],
        },
        "missing_images": len(missing),
        "missing_examples": missing[:20],
        "limitations": [
            "Overlap checks resolved file paths, not duplicate image content under different names.",
            "Caption flags are review hints, not verified visual errors.",
            "Using the reference as CF-A is reported separately; it is not automatically invalid.",
            "Path overlap does not establish its effect on the B1/B4 performance difference.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    image_root = args.image_root.expanduser().resolve()
    if not image_root.is_dir():
        parser.error("Image root does not exist")
    train = read_rows(args.data_dir / "train.csv")
    val = read_rows(args.data_dir / "val.csv")
    print("Checking both splits and image paths...", flush=True)
    report = diagnose(train, val, image_root)
    report["inputs"] = {"data_dir": str(args.data_dir.resolve()), "image_root": str(image_root),
                        "audit_seed": args.seed}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rng = random.Random(args.seed)
    population = [(split, index + 2, row) for split, rows in (("train", train), ("val", val))
                  for index, row in enumerate(rows)]
    fields = ["split", "source_csv_line", "review_flags", *train[0].keys(),
              "review_positive_P_E", "review_cf_a_P_not_E", "review_cf_b_not_P_E", "review_notes"]
    fields = list(dict.fromkeys(fields))
    samples = rng.sample(population, min(1000, len(population)))
    with (args.output_dir / "manual_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, line, row in samples:
            writer.writerow({**row, "split": split, "source_csv_line": line,
                             "review_flags": ";".join(flags(row))})
    with (args.output_dir / "flagged_examples.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        count = 0
        for split, line, row in population:
            hints = [x for x in flags(row) if x != "cf_a_is_reference"]
            if hints:
                writer.writerow({**row, "split": split, "source_csv_line": line,
                                 "review_flags": ";".join(hints)})
                count += 1
                if count >= 500:
                    break
    print(json.dumps({"splits": report["splits"], "cross_split": report["cross_split"],
                      "missing_images": report["missing_images"]}, indent=2))
    print(f"Reports: {args.output_dir.resolve()}")
    print("Manual audit remains pending. No data or checkpoints were modified.")


if __name__ == "__main__":
    main()
