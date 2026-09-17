"""Pseudo-edit tuples mined from image-caption data for B1/B4/B5 training."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

REQUIRED_COLUMNS = ("reference", "positive", "cf_a", "cf_b", "modification")


class PseudoEditDataset(Dataset[dict[str, Tensor | str]]):
    """Load reference/positive/CF-A/CF-B image tuples and modification text."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        preprocess: Callable[[Image.Image], Tensor],
        *,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.image_root = Path(image_root).resolve()
        self.preprocess = preprocess
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Pseudo-edit manifest not found: {self.manifest_path}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Pseudo-edit image root not found: {self.image_root}")
        with self.manifest_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
            if missing:
                raise ValueError(f"Pseudo-edit manifest is missing columns: {', '.join(missing)}")
            self.rows = [row for row in reader if all(row.get(column, "").strip() for column in REQUIRED_COLUMNS)]
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be greater than zero")
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError("Pseudo-edit manifest has no complete rows")

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, relative_path: str) -> Tensor:
        path = (self.image_root / relative_path).resolve()
        if not path.is_relative_to(self.image_root):
            raise ValueError(f"Pseudo-edit image escapes image_root: {relative_path}")
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB"))

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.rows[index]
        return {
            "reference": self._load(row["reference"]),
            "positive": self._load(row["positive"]),
            "cf_a": self._load(row["cf_a"]),
            "cf_b": self._load(row["cf_b"]),
            "modification": row["modification"],
            "attribute_type": row.get("attribute_type", "unknown"),
        }


def build_pseudo_edit_dataloader(
    manifest_path: str | Path,
    image_root: str | Path,
    preprocess: Callable[[Image.Image], Tensor],
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    max_samples: int | None = None,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    dataset = PseudoEditDataset(
        manifest_path, image_root, preprocess, max_samples=max_samples
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
