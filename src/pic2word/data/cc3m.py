"""Image-only CC3M dataset used to train the Pic2Word Mapping Network."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

IMAGE_COLUMN_CANDIDATES = (
    "image",
    "image_path",
    "path",
    "local_path",
    "filename",
    "file_name",
    "filepath",
)


class CC3MImageDataset(Dataset[Tensor]):
    """Load local CC3M images listed in a CSV or TSV manifest.

    Pic2Word trains its Mapping Network with images only, so captions and URLs in the
    original Conceptual Captions metadata are intentionally not returned.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        preprocess: Callable[[Image.Image], Tensor],
        *,
        image_column: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.image_root = Path(image_root).resolve()
        self.preprocess = preprocess

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"CC3M manifest not found: {self.manifest_path}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"CC3M image directory not found: {self.image_root}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be greater than zero when provided")

        self.image_paths = self._read_image_paths(image_column, max_samples)
        if not self.image_paths:
            raise ValueError(f"No image paths found in manifest: {self.manifest_path}")

    def _read_image_paths(
        self,
        requested_column: str | None,
        max_samples: int | None,
    ) -> list[Path]:
        delimiter = "\t" if self.manifest_path.suffix.lower() in {".tsv", ".txt"} else ","
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            image_column = requested_column or next(
                (candidate for candidate in IMAGE_COLUMN_CANDIDATES if candidate in fieldnames),
                None,
            )
            if image_column is None or image_column not in fieldnames:
                available = ", ".join(fieldnames) if fieldnames else "none"
                expected = requested_column or ", ".join(IMAGE_COLUMN_CANDIDATES)
                raise ValueError(
                    f"Cannot find an image-path column. Expected {expected}; available: {available}"
                )

            image_paths: list[Path] = []
            for row in reader:
                raw_path = (row.get(image_column) or "").strip()
                if not raw_path:
                    continue
                path = Path(raw_path)
                image_paths.append(path if path.is_absolute() else self.image_root / path)
                if max_samples is not None and len(image_paths) >= max_samples:
                    break
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        image_path = self.image_paths[index]
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                return self.preprocess(rgb_image)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot load training image: {image_path}") from error


def build_cc3m_dataloader(
    manifest_path: str | Path,
    image_root: str | Path,
    preprocess: Callable[[Image.Image], Tensor],
    *,
    image_column: str | None = None,
    max_samples: int | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = False,
    seed: int = 0,
) -> DataLoader[Tensor]:
    """Create the image-only DataLoader consumed by the Pic2Word trainer."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = CC3MImageDataset(
        manifest_path,
        image_root,
        preprocess,
        image_column=image_column,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
