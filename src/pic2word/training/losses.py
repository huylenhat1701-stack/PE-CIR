"""Contrastive objectives used to train Pic2Word."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional


class ContrastiveLossOutput(NamedTuple):
    """Loss values and logits useful for logging and diagnostics."""

    total: Tensor
    image_to_text: Tensor
    text_to_image: Tensor
    logits: Tensor


def symmetric_contrastive_loss(
    image_features: Tensor,
    text_features: Tensor,
    *,
    logit_scale: Tensor | float = 1.0,
) -> ContrastiveLossOutput:
    """Compute the symmetric CLIP-style contrastive loss.

    Matching image/text pairs must appear at the same row index. Both feature sets are
    normalized here so callers cannot accidentally use raw embeddings for the loss.
    """

    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("image_features and text_features must both be rank-2 tensors")
    if image_features.shape != text_features.shape:
        raise ValueError(
            "image_features and text_features must have identical shapes; "
            f"received {tuple(image_features.shape)} and {tuple(text_features.shape)}"
        )
    if image_features.shape[0] == 0:
        raise ValueError("contrastive loss requires at least one sample")

    image_features = functional.normalize(image_features, dim=-1)
    text_features = functional.normalize(text_features, dim=-1)
    scale = torch.as_tensor(
        logit_scale,
        device=image_features.device,
        dtype=image_features.dtype,
    )
    logits = scale * image_features @ text_features.transpose(0, 1)
    labels = torch.arange(logits.shape[0], device=logits.device)

    image_to_text = functional.cross_entropy(logits, labels)
    text_to_image = functional.cross_entropy(logits.transpose(0, 1), labels)
    total = (image_to_text + text_to_image) / 2
    return ContrastiveLossOutput(total, image_to_text, text_to_image, logits)
