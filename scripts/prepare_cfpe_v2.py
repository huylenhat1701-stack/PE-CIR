"""Prepare conservative, image-disjoint pseudo-edit candidates for manual review.

Uses the existing CC3M images. Does not train or overwrite previous datasets.
Caption rules provide hypotheses, not verified visual labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

KINDS = ("color", "pattern", "material")
VOCAB = {
    "color": "black white red blue green yellow brown pink purple orange gray grey".split(),
    "pattern": "plain striped floral plaid checked dotted printed".split(),
    "material": "wooden metal leather cotton silk denim glass plastic wool".split(),
}
OBJECTS = set("dress shirt jacket coat pants shoe boot bag hat car bike bicycle motorcycle bus truck boat plane dog cat horse bird chair table sofa bed lamp bottle cup plate phone computer building house flower tree".split())
TOKEN_KIND = {v: k for k, values in VOCAB.items() for v in values}
ROLES = ("reference", "positive", "cf_a", "cf_b")
FIELDS = [*ROLES, "modification", "attribute_type", "before", "after", "object",
          "preserved_attributes", "cf_b_changed_attribute", "reference_caption",
          "positive_caption", "cf_a_caption", "cf_b_caption"]


def normalize(text):
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def annotate(caption):
    """Only accept a contiguous attribute phrase attached to one named object."""
    words = normalize(caption).split()
    if any(w in {"no", "not", "without", "neither", "never"} for w in words):
        return None
    # Simple regular plurals only; unknown words are left untouched.
    words = [w[:-1] if w.endswith("s") and w[:-1] in OBJECTS else w for w in words]
    positions = [i for i, word in enumerate(words) if word in OBJECTS]
    if len(positions) != 1:
        return None
    pos = positions[0]
    # Avoid using the object as a modifier: e.g. "red floral dog collar".
    if pos + 1 < len(words) and words[pos + 1] in {
        "collar", "leash", "toy", "print", "picture", "photo", "statue", "logo",
        "costume", "shaped", "design", "pattern", "sleeve", "sleeves",
    }:
        return None
    attrs = {}
    index = pos - 1
    while index >= 0 and words[index] in TOKEN_KIND:
        word = words[index]
        kind = TOKEN_KIND[word]
        if kind in attrs:
            return None
        attrs[kind] = "gray" if word == "grey" else word
        index -= 1
    if len(attrs) < 2:
        return None
    # Reject disjunctions/conjoined attributes not consumed by the phrase.
    if index >= 0 and (words[index] in TOKEN_KIND or words[index] in {"and", "or"}):
        return None
    return (words[pos], *(attrs.get(k, "") for k in KINDS))


def split_for(digest, seed, val_fraction):
    number = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest(), 16)
    return "val" if number / 2**256 < val_fraction else "train"


def read_sources(manifest, image_root, seed, val_fraction, annotator=None):
    annotator = annotator or annotate
    pools = {"train": [], "val": []}
    counts = Counter()
    seen_hashes, seen_captions = set(), set()
    with manifest.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"image", "caption"}.issubset(reader.fieldnames or []):
            raise ValueError("Source CSV requires image and caption columns")
        # Stable input order makes duplicate representative selection repeatable.
        rows = sorted(reader, key=lambda r: (r.get("image", ""), r.get("caption", "")))
    for number, row in enumerate(rows, 1):
        counts["source_rows"] += 1
        name = row.get("image", "").strip()
        caption = row.get("caption", "").strip()
        if not name or not caption:
            counts["empty_row"] += 1
            continue
        path = (image_root / name).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError(f"Unsafe image path: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as image:
            digest = hashlib.file_digest(image, "sha256").hexdigest()
        # Deduplicate even before caption selection/splitting.
        if digest in seen_hashes:
            counts["duplicate_image_bytes"] += 1
        else:
            seen_hashes.add(digest)
            text_key = normalize(caption)
            if text_key in seen_captions:
                counts["duplicate_normalized_caption"] += 1
            else:
                seen_captions.add(text_key)
                sig = annotator(caption)
                if sig is None:
                    counts["caption_rejected"] += 1
                else:
                    split = split_for(digest, seed, val_fraction)
                    pools[split].append(dict(image=path.relative_to(image_root).as_posix(),
                                             caption=caption, sha256=digest, signature=sig))
                    counts[f"eligible_{split}_images"] += 1
        if number % 2000 == 0:
            print(f"Checked {number}/{len(rows)} source images", flush=True)
    return pools, dict(counts)


def mine(records, seed, max_tuples, per_reference):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in records:
        buckets[r["signature"]].append(r)
    signatures = sorted(buckets)
    by_edit = defaultdict(list)
    for sig in signatures:
        for index, value in enumerate(sig[1:], 1):
            if value:
                by_edit[(sig[0], index, value)].append(sig)
    rows, seen, usage = [], set(), Counter()
    references = sorted(records, key=lambda r: r["image"])
    rng.shuffle(references)
    # Round-robin prevents a few early references from consuming the full quota.
    for round_id in range(per_reference):
        for ref in references:
            sig = ref["signature"]
            indices = [i for i in range(1, 4) if sig[i]]
            rng.shuffle(indices)
            indices.sort(key=lambda i: usage[KINDS[i - 1]])
            added = False
            for index in indices:
                afters = [v for v in VOCAB[KINDS[index - 1]] if v != "grey" and v != sig[index]]
                rng.shuffle(afters)
                for after in afters:
                    pos_sig = list(sig)
                    pos_sig[index] = after
                    pos_sig = tuple(pos_sig)
                    positives = buckets.get(pos_sig, [])
                    cf_as = [r for r in buckets[sig] if r["image"] != ref["image"]]
                    if not positives or not cf_as:
                        continue
                    # Same object and edited attribute; exactly one known preserve
                    # attribute differs. Missing attributes are not evidence of change.
                    b_sigs = []
                    for bs in by_edit[(sig[0], index, after)]:
                        differences = [j for j in range(1, 4) if j != index and bs[j] != sig[j]]
                        if len(differences) == 1 and all(bool(bs[j]) == bool(sig[j]) for j in range(1, 4)):
                            b_sigs.append((bs, differences[0]))
                    if not b_sigs:
                        continue
                    candidates = [p for p in positives if (ref["image"], p["image"], index) not in seen]
                    if not candidates:
                        continue
                    positive = rng.choice(candidates)
                    cf_a = rng.choice(cf_as)
                    b_sig, changed = rng.choice(b_sigs)
                    cf_b = rng.choice(buckets[b_sig])
                    seen.add((ref["image"], positive["image"], index))
                    kind = KINDS[index - 1]
                    preserve = {KINDS[j - 1]: sig[j] for j in range(1, 4) if j != index and sig[j]}
                    row = dict(zip(ROLES, [r["image"] for r in (ref, positive, cf_a, cf_b)]))
                    row.update(modification=f"change the {kind} of the {sig[0]} from {sig[index]} to {after}",
                               attribute_type=kind, before=sig[index], after=after, object=sig[0],
                               preserved_attributes=json.dumps(preserve, sort_keys=True),
                               cf_b_changed_attribute=KINDS[changed - 1])
                    for role, record in zip(ROLES, (ref, positive, cf_a, cf_b)):
                        row[f"{role}_caption"] = record["caption"]
                    rows.append(row)
                    usage[kind] += 1
                    added = True
                    break
                if added:
                    break
            if len(rows) >= max_tuples:
                rng.shuffle(rows)
                return rows
        print(f"Mining round {round_id + 1}/{per_reference}: {len(rows)} candidates", flush=True)
    rng.shuffle(rows)
    return rows


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(annotator=None, miner=None, profile="strict_two_attributes_v2", extra_limitations=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-per-reference", type=int, default=4)
    p.add_argument("--max-train", type=int, default=50000)
    p.add_argument("--max-val", type=int, default=5000)
    a = p.parse_args()
    if not 0 < a.val_fraction < 1 or min(a.max_per_reference, a.max_train, a.max_val) < 1:
        p.error("Require 0 < val-fraction < 1 and positive limits")
    if a.output_dir.exists():
        p.error("Choose a new output directory; existing data will not be overwritten")
    root = a.image_root.expanduser().resolve()
    if not root.is_dir():
        p.error("Image root does not exist")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    print("Hashing images and selecting conservative captions. No GPU needed.", flush=True)
    pools, counts = read_sources(a.source_manifest, root, a.seed, a.val_fraction, annotator)
    result = {}
    for split in ("train", "val"):
        print(f"Mining {split}: {len(pools[split])} eligible source images", flush=True)
        result[split] = (miner or mine)(pools[split], a.seed,
            a.max_train if split == "train" else a.max_val, a.max_per_reference)
        write_csv(a.output_dir / f"{split}_candidates.csv", result[split], FIELDS)
        write_csv(a.output_dir / f"{split}_source.csv",
                  [{k: r[k] for k in ("image", "caption", "sha256")} for r in pools[split]],
                  ["image", "caption", "sha256"])
    sets = {s: {r[k] for r in result[s] for k in ROLES} for s in result}
    hashes = {s: {r["sha256"] for r in pools[s]} for s in pools}
    assert not sets["train"] & sets["val"]
    assert not hashes["train"] & hashes["val"]
    rng = random.Random(a.seed)
    audit = []
    for split, rows in result.items():
        for index in rng.sample(range(len(rows)), min(500, len(rows))):
            audit.append({"split": split, "source_csv_line": index + 2, **rows[index]})
    write_csv(a.output_dir / "manual_audit.csv", audit,
              ["split", "source_csv_line", *FIELDS, "positive_ok", "cf_a_ok", "cf_b_ok", "notes"])
    report = dict(status="CANDIDATES_ONLY_MANUAL_AUDIT_REQUIRED", dataset_profile=profile,
                  source_counts=counts,
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  source_manifest_sha256=hashlib.sha256(a.source_manifest.read_bytes()).hexdigest(),
                  shared_image_paths=0, shared_exact_image_hashes=0,
                  screening_count_target_met=len(result["train"]) >= 20000 and len(result["val"]) >= 500,
                  splits={s: dict(tuples=len(rows), unique_images=len(sets[s]),
                      unique_references=len({r["reference"] for r in rows}),
                      attribute_counts=dict(Counter(r["attribute_type"] for r in rows))) for s, rows in result.items()},
                  limitations=["Caption constraints are not verified visual semantics.",
                      "SHA256 removes byte-identical files, not resized/reencoded or near-duplicate images.",
                      "Pilot edits cover color/pattern/material only; size/shape and object replacement edits are excluded.",
                      "No minimum yield is guaranteed; do not relax quality checks just to fill a quota.",
                      "Existing checkpoints have seen the previous split; retrain both B1 and B4 after approving new data."])
    report["limitations"].extend(extra_limitations or [])
    (a.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"DONE: {a.output_dir.resolve()}. Review report before training.", flush=True)


if __name__ == "__main__":
    main()
