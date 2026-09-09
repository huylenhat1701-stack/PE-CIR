"""CIRR annotations and image-path loading for validation evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CIRRQuery:
    """One composed-image retrieval query from a CIRR caption split."""

    pair_id: int
    reference_id: str
    target_id: str
    caption: str
    group_members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CIRRSplit:
    """Resolved CIRR query annotations and candidate image paths."""

    root: Path
    split: str
    version: str
    queries: tuple[CIRRQuery, ...]
    image_paths: dict[str, Path]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Return all candidate IDs in the stable annotation order."""

        return tuple(self.image_paths)

    @property
    def candidate_paths(self) -> list[Path]:
        """Return paths aligned with :attr:`candidate_ids`."""

        return list(self.image_paths.values())

    def path_for(self, image_id: str) -> Path:
        """Resolve an image ID or raise a useful annotation error."""

        try:
            return self.image_paths[image_id]
        except KeyError as error:
            raise KeyError(f"CIRR image ID is absent from the {self.split} image split: {image_id}") from error

    def missing_images(self) -> list[Path]:
        """Return annotated paths whose raw image files are unavailable."""

        return [path for path in self.image_paths.values() if not path.is_file()]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"CIRR annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _required_string(entry: dict[str, Any], field: str, index: int) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CIRR caption entry {index} has an invalid {field!r}")
    return value.strip()


def load_cirr_split(
    root: str | Path,
    *,
    split: str = "val",
    version: str = "rc2",
) -> CIRRSplit:
    """Load the official ``captions`` and ``image_splits`` CIRR JSON files.

    ``root`` must contain ``captions/``, ``image_splits/`` and ``img_raw/``.
    Files are not decoded here; call :meth:`CIRRSplit.missing_images` before
    building the candidate index to diagnose an incomplete dataset download.
    """

    dataset_root = Path(root).resolve()
    captions_path = dataset_root / "captions" / f"cap.{version}.{split}.json"
    image_split_path = dataset_root / "image_splits" / f"split.{version}.{split}.json"
    caption_payload = _load_json(captions_path)
    image_payload = _load_json(image_split_path)

    if not isinstance(caption_payload, list):
        raise TypeError(f"CIRR captions must be a JSON list: {captions_path}")
    if not isinstance(image_payload, dict):
        raise TypeError(f"CIRR image split must be a JSON object: {image_split_path}")

    image_paths: dict[str, Path] = {}
    for image_id, relative_path in image_payload.items():
        if not isinstance(image_id, str) or not isinstance(relative_path, str):
            raise TypeError("Every CIRR image split entry must map a string ID to a string path")
        normalized_relative = relative_path.replace("\\", "/").removeprefix("./")
        image_paths[image_id] = (dataset_root / "img_raw" / normalized_relative).resolve()

    queries: list[CIRRQuery] = []
    for index, raw_entry in enumerate(caption_payload):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"CIRR caption entry {index} must be a JSON object")
        reference_id = _required_string(raw_entry, "reference", index)
        target_id = _required_string(raw_entry, "target_hard", index)
        caption = _required_string(raw_entry, "caption", index)
        pair_id = raw_entry.get("pairid")
        if not isinstance(pair_id, int):
            raise TypeError(f"CIRR caption entry {index} has an invalid 'pairid'")
        image_set = raw_entry.get("img_set")
        if not isinstance(image_set, dict) or not isinstance(image_set.get("members"), list):
            raise TypeError(f"CIRR caption entry {index} has invalid 'img_set.members'")
        group_members = tuple(image_set["members"])
        if not group_members or not all(isinstance(value, str) for value in group_members):
            raise ValueError(f"CIRR caption entry {index} has invalid group member IDs")

        missing_ids = [
            image_id
            for image_id in (reference_id, target_id, *group_members)
            if image_id not in image_paths
        ]
        if missing_ids:
            raise ValueError(
                f"CIRR caption entry {index} references IDs absent from the image split: "
                + ", ".join(missing_ids)
            )
        queries.append(
            CIRRQuery(
                pair_id=pair_id,
                reference_id=reference_id,
                target_id=target_id,
                caption=caption,
                group_members=group_members,
            )
        )

    if not queries:
        raise ValueError(f"CIRR caption split is empty: {captions_path}")
    if not image_paths:
        raise ValueError(f"CIRR image split is empty: {image_split_path}")
    return CIRRSplit(dataset_root, split, version, tuple(queries), image_paths)
