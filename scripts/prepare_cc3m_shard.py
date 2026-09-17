"""Extract a small image-only training set from one CC3M WebDataset shard."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to cc3m-train-XXXX.tar")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cc_data/train"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/cc/Train_GCC-training_output.csv"),
    )
    parser.add_argument("--max-images", type=int, default=1000)
    return parser


def _safe_basename(member: tarfile.TarInfo) -> str:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"Unsafe archive member path: {member.name}")
    return path.name


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _read_caption(archive: tarfile.TarFile, member: tarfile.TarInfo | None) -> str:
    if member is None:
        return ""
    source = archive.extractfile(member)
    if source is None:
        return ""
    with source:
        text = source.read(1024 * 1024).decode("utf-8", errors="replace").strip()
    if Path(member.name).suffix.lower() == ".json":
        try:
            payload = json.loads(text)
            for key in ("caption", "text", "txt"):
                if isinstance(payload.get(key), str):
                    return payload[key].strip()
        except (json.JSONDecodeError, AttributeError):
            return ""
    return text


def prepare_shard(
    archive_path: Path,
    output_dir: Path,
    manifest_path: Path,
    max_images: int,
) -> tuple[int, int]:
    """Extract at most ``max_images`` valid images and write their CSV manifest."""

    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    manifest_path = manifest_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"CC3M shard not found: {archive_path}")
    if max_images <= 0:
        raise ValueError("max_images must be greater than zero")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_rows: list[tuple[str, str]] = []
    invalid_images = 0

    with tarfile.open(archive_path, mode="r:") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        sidecars: dict[str, tarfile.TarInfo] = {}
        for member in members:
            name = _safe_basename(member)
            if Path(name).suffix.lower() in {".txt", ".json"}:
                sidecars.setdefault(Path(name).stem, member)
        for member in members:
            name = _safe_basename(member)
            if Path(name).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue

            destination = output_dir / name
            source = archive.extractfile(member)
            if source is None:
                invalid_images += 1
                continue
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)

            if not _is_valid_image(destination):
                destination.unlink(missing_ok=True)
                invalid_images += 1
                continue

            caption = _read_caption(archive, sidecars.get(Path(name).stem))
            extracted_rows.append((name, caption))
            if len(extracted_rows) >= max_images:
                break

    if not extracted_rows:
        raise RuntimeError("The shard did not contain any valid supported images")

    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "caption"])
        writer.writerows(extracted_rows)

    return len(extracted_rows), invalid_images


def main() -> int:
    args = build_parser().parse_args()
    extracted, invalid = prepare_shard(
        args.archive,
        args.output_dir,
        args.manifest,
        args.max_images,
    )
    print(f"Prepared {extracted} valid CC3M images")
    print(f"Skipped {invalid} invalid images")
    print(f"Image directory: {args.output_dir.resolve()}")
    print(f"Manifest: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
