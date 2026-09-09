"""Checkpoint import helpers for the official Pic2Word implementation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

OFFICIAL_MAPPING_KEYS = {
    "module.layers.0.0.weight": "hidden.0.weight",
    "module.layers.0.0.bias": "hidden.0.bias",
    "module.layers.1.0.weight": "hidden.3.weight",
    "module.layers.1.0.bias": "hidden.3.bias",
    "module.fc_out.weight": "output.weight",
    "module.fc_out.bias": "output.bias",
}


def convert_official_mapping_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename the official ``IM2TEXT`` parameters for :class:`MappingNetwork`."""

    missing = sorted(set(OFFICIAL_MAPPING_KEYS) - set(state))
    extra = sorted(set(state) - set(OFFICIAL_MAPPING_KEYS))
    if missing or extra:
        raise ValueError(
            "Unexpected official Mapping Network keys "
            f"(missing={missing or 'none'}, extra={extra or 'none'})"
        )
    converted = {
        target_key: state[source_key].detach().cpu().float()
        for source_key, target_key in OFFICIAL_MAPPING_KEYS.items()
    }
    expected_shapes = {
        "hidden.0.weight": (512, 768),
        "hidden.0.bias": (512,),
        "hidden.3.weight": (512, 512),
        "hidden.3.bias": (512,),
        "output.weight": (768, 512),
        "output.bias": (768,),
    }
    actual_shapes = {key: tuple(value.shape) for key, value in converted.items()}
    if actual_shapes != expected_shapes:
        raise ValueError(
            "Official checkpoint is not the expected Pic2Word ViT-L/14 Mapping Network: "
            f"{actual_shapes}"
        )
    return converted


def _numpy_safe_globals() -> list[Any]:
    # Old Pic2Word checkpoints use NumPy scalars for optimizer metadata. These
    # narrowly-scoped globals let torch read that metadata while weights_only=True
    # continues to reject arbitrary Python objects.
    from numpy._core.multiarray import scalar

    safe_globals: list[Any] = [
        (scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
    ]
    if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
        safe_globals.append(np.dtypes.Float64DType)
    return safe_globals


def load_official_mapping_checkpoint(path: str | Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Safely read and convert a full official Pic2Word training checkpoint."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Official Pic2Word checkpoint not found: {checkpoint_path}")
    with torch.serialization.safe_globals(_numpy_safe_globals()):
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict_img2text"), dict):
        raise TypeError("File is not an official Pic2Word checkpoint with state_dict_img2text")
    state = convert_official_mapping_state(payload["state_dict_img2text"])
    metadata = {
        "original_epoch": int(payload.get("epoch", -1)),
        "original_name": str(payload.get("name", "")),
    }
    return state, metadata


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a reproducibility fingerprint without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
