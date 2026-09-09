"""Import authorized CIRR/NLVR2 raw images and verify the validation split."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

from pic2word.data import load_cirr_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted NLVR2 directory or ZIP archive")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cirr"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _relative_key(path: str | Path) -> str:
    return PurePosixPath(str(path).replace("\\", "/")).name


def _source_directory_files(source: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in source.rglob("*.png"):
        files.setdefault(_relative_key(path), path)
    return files


def _zip_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for info in archive.infolist():
        if not info.is_dir() and info.filename.lower().endswith(".png"):
            members.setdefault(_relative_key(info.filename), info.filename)
    return members


def _copy_from_directory(
    source: Path,
    destinations: list[Path],
    *,
    overwrite: bool,
) -> int:
    source_files = _source_directory_files(source)
    copied = 0
    for destination in destinations:
        if destination.is_file() and not overwrite:
            continue
        source_path = source_files.get(destination.name)
        if source_path is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied += 1
    return copied


def _copy_from_zip(
    source: Path,
    destinations: list[Path],
    *,
    overwrite: bool,
) -> int:
    copied = 0
    with zipfile.ZipFile(source) as archive:
        members = _zip_members(archive)
        for destination in destinations:
            if destination.is_file() and not overwrite:
                continue
            member = members.get(destination.name)
            if member is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as input_stream, destination.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
            copied += 1
    return copied


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"CIRR/NLVR2 source not found: {source}")

    dataset = load_cirr_split(args.dataset_root)
    destinations = dataset.candidate_paths
    if source.is_dir():
        copied = _copy_from_directory(source, destinations, overwrite=args.overwrite)
    elif zipfile.is_zipfile(source):
        copied = _copy_from_zip(source, destinations, overwrite=args.overwrite)
    else:
        raise ValueError("CIRR/NLVR2 source must be an extracted directory or ZIP archive")

    missing = dataset.missing_images()
    invalid = [path for path in destinations if path.is_file() and not _valid_image(path)]
    print(f"Imported {copied} CIRR validation images")
    print(f"Valid images: {len(destinations) - len(missing) - len(invalid)}/{len(destinations)}")
    if missing or invalid:
        raise RuntimeError(
            f"CIRR images are incomplete: {len(missing)} missing and {len(invalid)} invalid"
        )
    print("CIRR validation images are complete and readable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
