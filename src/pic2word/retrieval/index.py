"""Small PyTorch candidate index for composed image retrieval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional

from pic2word.models.clip_backbone import FrozenCLIPBackbone

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked candidate returned by a search."""

    path: Path
    score: float


class CandidateIndex:
    """Normalized candidate image features and their source paths."""

    def __init__(self, paths: list[Path], features: Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("Candidate features must have shape [images, embedding_dim]")
        if len(paths) != features.shape[0]:
            raise ValueError("Candidate path and feature counts must match")
        if not paths:
            raise ValueError("Candidate index cannot be empty")
        self.paths = tuple(path.resolve() for path in paths)
        self.features = functional.normalize(features.detach().cpu().float(), dim=-1)

    def search(
        self,
        query_feature: Tensor,
        *,
        top_k: int = 5,
        exclude_paths: set[Path] | None = None,
        allowed_paths: set[Path] | None = None,
    ) -> list[RetrievalResult]:
        """Rank candidates by cosine similarity to one composed query.

        ``allowed_paths`` restricts the search space, which is required for the
        CIRR group/subset Recall metric.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if query_feature.ndim == 2 and query_feature.shape[0] == 1:
            query_feature = query_feature[0]
        if query_feature.ndim != 1 or query_feature.shape[0] != self.features.shape[1]:
            raise ValueError("Query feature must match the candidate embedding dimension")

        query_feature = functional.normalize(query_feature.detach().cpu().float(), dim=-1)
        scores = self.features @ query_feature
        excluded = {path.resolve() for path in (exclude_paths or set())}
        allowed = None
        if allowed_paths is not None:
            allowed = {path.resolve() for path in allowed_paths}
        allowed_indices = [
            index
            for index, path in enumerate(self.paths)
            if path not in excluded and (allowed is None or path in allowed)
        ]
        if not allowed_indices:
            return []

        allowed_tensor = torch.tensor(allowed_indices, dtype=torch.long)
        allowed_scores = scores[allowed_tensor]
        result_count = min(top_k, allowed_scores.numel())
        values, positions = torch.topk(allowed_scores, k=result_count)
        return [
            RetrievalResult(
                path=self.paths[allowed_indices[int(position)]],
                score=float(value),
            )
            for value, position in zip(values, positions, strict=True)
        ]

    def save(self, path: str | Path) -> Path:
        """Save paths and CLIP candidate features for later searches."""

        index_path = Path(path).resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
        torch.save(
            {"paths": [str(path) for path in self.paths], "features": self.features},
            temporary_path,
        )
        temporary_path.replace(index_path)
        return index_path

    @classmethod
    def load(cls, path: str | Path) -> CandidateIndex:
        """Load an index created by :meth:`save`."""

        payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=True)
        return cls([Path(value) for value in payload["paths"]], payload["features"])


def find_candidate_images(directory: str | Path, *, limit: int | None = None) -> list[Path]:
    """Return sorted supported image paths from a local directory."""

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Candidate image directory not found: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("Candidate limit must be greater than zero")
        paths = paths[:limit]
    return paths


def build_candidate_index(
    backbone: FrozenCLIPBackbone,
    paths: list[Path],
    *,
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> CandidateIndex:
    """Encode local images with frozen CLIP and build a normalized search index."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    valid_paths: list[Path] = []
    feature_batches: list[Tensor] = []

    for start in range(0, len(paths), batch_size):
        image_tensors: list[Tensor] = []
        batch_paths: list[Path] = []
        for path in paths[start : start + batch_size]:
            try:
                with Image.open(path) as image:
                    image_tensors.append(backbone.preprocess(image.convert("RGB")))
                batch_paths.append(path)
            except (OSError, ValueError):
                continue
        if not image_tensors:
            continue
        image_batch = torch.stack(image_tensors)
        feature_batches.append(backbone.encode_image(image_batch, normalize=True).cpu())
        valid_paths.extend(batch_paths)
        if progress is not None:
            progress(min(start + batch_size, len(paths)), len(paths))

    if not feature_batches:
        raise RuntimeError("No valid candidate images could be encoded")
    return CandidateIndex(valid_paths, torch.cat(feature_batches, dim=0))
