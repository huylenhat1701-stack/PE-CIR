"""Mine Preserve/Edit and CF-A/CF-B tuples from an image-caption manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ATTRIBUTE_VALUES = {
    "color": ("black", "white", "red", "blue", "green", "yellow", "brown", "pink", "purple", "orange", "gray", "grey"),
    "pattern": ("plain", "striped", "floral", "plaid", "checked", "dotted", "printed"),
    "material": ("wooden", "metal", "leather", "cotton", "silk", "denim", "glass", "plastic", "wool"),
    "size_shape": ("small", "large", "long", "short", "round", "square", "wide", "narrow", "tall"),
}
OBJECTS = (
    "dress", "shirt", "top", "jacket", "coat", "pants", "shoe", "boot", "bag", "hat",
    "car", "bike", "bicycle", "motorcycle", "bus", "truck", "boat", "plane",
    "dog", "cat", "horse", "bird", "chair", "table", "sofa", "bed", "lamp", "bottle",
    "cup", "plate", "phone", "computer", "building", "house", "flower", "tree", "food",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--max-tuples", type=int, default=50_000)
    parser.add_argument("--min-tuples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-small", action="store_true")
    return parser


def _terms(text: str, vocabulary: tuple[str, ...]) -> tuple[str, ...]:
    normalized = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower())} "
    return tuple(term for term in vocabulary if f" {term} " in normalized)


def annotate(row: dict[str, str]) -> dict[str, object]:
    caption = row["caption"].strip()
    attributes = {name: _terms(caption, values) for name, values in ATTRIBUTE_VALUES.items()}
    objects = _terms(caption, OBJECTS)
    return {"image": row["image"].strip(), "caption": caption, "attributes": attributes, "objects": objects}


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    if args.max_tuples <= 0 or args.min_tuples <= 0:
        raise ValueError("max-tuples and min-tuples must be positive")
    with args.source_manifest.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "caption"}.issubset(reader.fieldnames or ()):
            raise ValueError("Source manifest must contain image and caption columns")
        records = [annotate(row) for row in reader if row.get("image") and row.get("caption")]
    if not records:
        raise ValueError("Source manifest contains no captioned images")

    by_object: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    by_attribute: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        objects = record["objects"]
        if objects:
            by_object[objects].append(record)
        for kind, values in record["attributes"].items():
            for value in values:
                by_attribute[(kind, value)].append(record)

    tuples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    groups = list(by_object.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
        for reference in group:
            for positive in group:
                if reference["image"] == positive["image"]:
                    continue
                for kind in ATTRIBUTE_VALUES:
                    reference_values = reference["attributes"][kind]
                    positive_values = positive["attributes"][kind]
                    if len(reference_values) != 1 or len(positive_values) != 1:
                        continue
                    before, after = reference_values[0], positive_values[0]
                    if before == after:
                        continue
                    cf_a_pool = [
                        item for item in group
                        if before in item["attributes"][kind] and item["image"] != positive["image"]
                    ]
                    cf_b_pool = [
                        item for item in by_attribute[(kind, after)]
                        if item["objects"] != reference["objects"]
                    ]
                    if not cf_a_pool or not cf_b_pool:
                        continue
                    cf_a = rng.choice(cf_a_pool)
                    cf_b = rng.choice(cf_b_pool)
                    key = (reference["image"], positive["image"], kind)
                    if key in seen:
                        continue
                    seen.add(key)
                    tuples.append({
                        "reference": reference["image"],
                        "positive": positive["image"],
                        "cf_a": cf_a["image"],
                        "cf_b": cf_b["image"],
                        "modification": f"change the {kind.replace('_', ' ')} from {before} to {after}",
                        "attribute_type": kind,
                        "before": before,
                        "after": after,
                        "reference_caption": reference["caption"],
                        "positive_caption": positive["caption"],
                        "cf_a_caption": cf_a["caption"],
                        "cf_b_caption": cf_b["caption"],
                    })
                    if len(tuples) >= args.max_tuples:
                        break
                if len(tuples) >= args.max_tuples:
                    break
            if len(tuples) >= args.max_tuples:
                break
        if len(tuples) >= args.max_tuples:
            break

    if len(tuples) < args.min_tuples and not args.allow_small:
        raise RuntimeError(
            f"Only {len(tuples)} high-confidence tuples were mined; need at least {args.min_tuples}. "
            "Add more captioned CC3M shards or use --allow-small only for a smoke test."
        )
    fields = list(tuples[0]) if tuples else [
        "reference", "positive", "cf_a", "cf_b", "modification", "attribute_type"
    ]
    for path, rows in ((args.output, tuples), (args.audit, tuples[: max(500, min(len(tuples), 1000))])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps({
        "source_manifest": str(args.source_manifest.resolve()),
        "seed": args.seed,
        "max_tuples": args.max_tuples,
        "min_tuples": args.min_tuples,
        "records_scanned": len(records),
        "tuples_mined": len(tuples),
        "attribute_priority": list(ATTRIBUTE_VALUES),
    }, indent=2), encoding="utf-8")
    print(f"Mined {len(tuples)} tuples")
    print(f"Training manifest: {args.output.resolve()}")
    print(f"Audit rows: {min(len(tuples), max(500, min(len(tuples), 1000)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
