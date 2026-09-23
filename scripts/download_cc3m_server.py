"""Download and prepare the CC3M screening subset used by the Colab notebook."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from prepare_cc3m_shard import prepare_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "datasets" / "cc3m")
    parser.add_argument("--num-shards", type=int, default=20)
    parser.add_argument("--images-per-shard", type=int, default=5000)
    args = parser.parse_args()
    if args.num_shards < 1 or args.images_per_shard < 1:
        parser.error("Shard and image counts must be positive")
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = "pixparse/cc3m-wds"
    metadata = root / "source_revision.json"
    if metadata.exists():
        revision = json.loads(metadata.read_text(encoding="utf-8"))["revision"]
    else:
        revision = HfApi().dataset_info(repo).sha
        metadata.write_text(json.dumps({"repo": repo, "revision": revision}, indent=2), encoding="utf-8")
    subset = root / f"subset_{args.images_per_shard}"
    image_root = subset / "images"
    manifests = subset / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    all_rows = []
    print(f"Dataset: {repo}, revision: {revision}", flush=True)
    for index in range(args.num_shards):
        shard = f"{index:04d}"
        manifest = manifests / f"{shard}.csv"
        rows = []
        if manifest.exists():
            with manifest.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        if not rows or not all((image_root / shard / row["image"]).is_file() for row in rows):
            filename = f"cc3m-train-{shard}.tar"
            print(f"[{index + 1}/{args.num_shards}] Download / extract {filename}", flush=True)
            archive = hf_hub_download(
                repo_id=repo, repo_type="dataset", filename=filename,
                revision=revision, local_dir=str(root / "archives"),
            )
            temporary = manifest.with_suffix(".tmp")
            prepare_shard(Path(archive), image_root / shard, temporary, args.images_per_shard)
            temporary.replace(manifest)
            with manifest.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            print(f"[{index + 1}/{args.num_shards}] Reuse {shard}: {len(rows)} images", flush=True)
        all_rows.extend({"image": f"{shard}/{row['image']}", "caption": row["caption"]} for row in rows)
    combined = subset / "cc3m_captioned.csv"
    temporary = combined.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "caption"])
        writer.writeheader()
        writer.writerows(all_rows)
    temporary.replace(combined)
    captioned = sum(bool(row["caption"].strip()) for row in all_rows)
    print(f"DONE: {len(all_rows)} images, {captioned} with captions", flush=True)
    print(f"Image root: {image_root}\nManifest: {combined}", flush=True)


if __name__ == "__main__":
    main()
