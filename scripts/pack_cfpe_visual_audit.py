"""Package 40 targeted/random audit tuples with small image previews for review."""
import argparse
import csv
import io
import json
import random
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageOps
from review_cfpe_captions import review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.image_root.expanduser().resolve()
    with args.audit.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(0)
    priority = [i for i, row in enumerate(rows)
                if any(k != "possible_media_mismatch" for k, _ in review(row))]
    selected = rng.sample(priority, min(20, len(priority)))
    others = [i for i in range(len(rows)) if i not in selected]
    selected += rng.sample(others, min(40 - len(selected), len(others)))
    payload = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "x", ZIP_DEFLATED) as archive:
        saved = set()
        for number, index in enumerate(selected, 1):
            row = dict(rows[index])
            row["audit_csv_line"] = index + 2
            row["caption_flags"] = review(row)
            row["preview_paths"] = {}
            for role in ("reference", "positive", "cf_a", "cf_b"):
                path = (root / row[role]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"Unsafe path: {row[role]}")
                name = "images/" + path.relative_to(root).as_posix() + ".jpg"
                row["preview_paths"][role] = name
                if name not in saved:
                    with Image.open(path) as image:
                        preview = ImageOps.exif_transpose(image).convert("RGB")
                        preview.thumbnail((384, 384))
                        buffer = io.BytesIO()
                        preview.save(buffer, format="JPEG", quality=85)
                    archive.writestr(name, buffer.getvalue())
                    saved.add(name)
            payload.append(row)
            print(f"Packed {number}/{len(selected)} tuples", flush=True)
        archive.writestr("manifest.json", json.dumps(dict(
            selection="Up to 20 structural-caption flags, then random remaining rows; seed=0.",
            scope="Diagnostic sample, not a completed 500/200 visual audit or an error-rate estimate.",
            tuples=payload), ensure_ascii=False, indent=2))
    print(f"DONE: {args.output.resolve()}")


if __name__ == "__main__":
    main()
