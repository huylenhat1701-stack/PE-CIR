"""Fashion-IQ validation annotations and image paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FashionIQQuery:
    category: str
    reference_id: str
    target_id: str
    captions: tuple[str, ...]

    @property
    def modification(self) -> str:
        return " and ".join(caption.strip().rstrip(".") for caption in self.captions)


@dataclass(frozen=True, slots=True)
class FashionIQSplit:
    root: Path
    queries: tuple[FashionIQQuery, ...]
    image_paths: dict[str, dict[str, Path]]

    def candidates(self, category: str) -> list[Path]:
        return list(self.image_paths[category].values())

    def path_for(self, category: str, image_id: str) -> Path:
        return self.image_paths[category][image_id]

    def missing_images(self) -> list[Path]:
        return [path for paths in self.image_paths.values() for path in paths.values() if not path.is_file()]


def _image_path(root: Path, image_id: str) -> Path:
    direct = root / "images" / image_id
    if direct.suffix:
        return direct.resolve()
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = direct.with_suffix(suffix)
        if candidate.is_file():
            return candidate.resolve()
    return direct.with_suffix(".png").resolve()


def load_fashioniq_split(
    root: str | Path,
    *,
    split: str = "val",
    categories: tuple[str, ...] = ("dress", "shirt", "toptee"),
) -> FashionIQSplit:
    root = Path(root).resolve()
    queries: list[FashionIQQuery] = []
    images: dict[str, dict[str, Path]] = {}
    for category in categories:
        split_path = root / "image_splits" / f"split.{category}.{split}.json"
        captions_path = root / "captions" / f"cap.{category}.{split}.json"
        if not split_path.is_file() or not captions_path.is_file():
            raise FileNotFoundError(f"Fashion-IQ annotations missing for {category}/{split}")
        image_ids = json.loads(split_path.read_text(encoding="utf-8"))
        captions = json.loads(captions_path.read_text(encoding="utf-8"))
        if not isinstance(image_ids, list) or not isinstance(captions, list):
            raise TypeError("Fashion-IQ split and captions must be JSON lists")
        images[category] = {image_id: _image_path(root, image_id) for image_id in image_ids}
        for entry in captions:
            query = FashionIQQuery(
                category=category,
                reference_id=str(entry["candidate"]),
                target_id=str(entry["target"]),
                captions=tuple(str(value) for value in entry["captions"]),
            )
            if query.reference_id not in images[category] or query.target_id not in images[category]:
                raise ValueError(f"Fashion-IQ query references an ID outside {category}/{split}")
            queries.append(query)
    return FashionIQSplit(root, tuple(queries), images)
