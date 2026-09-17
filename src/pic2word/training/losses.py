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


def retrieval_alignment_loss(
    queries: Tensor,
    positives: Tensor,
    *,
    logit_scale: Tensor | float = 1.0,
) -> Tensor:
    """In-batch Stage-1 retrieval loss for aligned query/positive rows."""

    if queries.ndim != 2 or queries.shape != positives.shape:
        raise ValueError("queries and positives must have identical [batch, dim] shapes")
    queries = functional.normalize(queries, dim=-1)
    positives = functional.normalize(positives, dim=-1)
    scale = torch.as_tensor(logit_scale, device=queries.device, dtype=queries.dtype)
    labels = torch.arange(queries.shape[0], device=queries.device)
    return functional.cross_entropy(scale * queries @ positives.transpose(0, 1), labels)


def factorization_loss(
    preserve: Tensor,
    composed_edit: Tensor,
    modification: Tensor,
    gates: Tensor,
) -> Tensor:
    """Keep Preserve separated from the requested edit and align the Edit branch.

    A small gate-balance term prevents all slots from collapsing into only one branch.
    """

    modification = functional.normalize(modification, dim=-1)
    preserve = functional.normalize(preserve, dim=-1)
    composed_edit = functional.normalize(composed_edit, dim=-1)
    preserve_leak = functional.cosine_similarity(preserve, modification, dim=-1).square().mean()
    edit_alignment = (1.0 - functional.cosine_similarity(composed_edit, modification, dim=-1)).mean()
    gate_balance = (gates.mean(dim=-1) - 0.5).square().mean()
    return preserve_leak + edit_alignment + 0.1 * gate_balance


def counterfactual_margin_loss(
    queries: Tensor,
    positives: Tensor,
    cf_a: Tensor,
    cf_b: Tensor,
    *,
    margin: float = 0.2,
) -> Tensor:
    """Require the positive to outrank both P+E- and P-E+ counterfactuals."""

    if margin < 0:
        raise ValueError("margin cannot be negative")
    tensors = [functional.normalize(value, dim=-1) for value in (queries, positives, cf_a, cf_b)]
    query, positive, negative_a, negative_b = tensors
    positive_score = (query * positive).sum(dim=-1)
    score_a = (query * negative_a).sum(dim=-1)
    score_b = (query * negative_b).sum(dim=-1)
    return (
        functional.relu(margin - positive_score + score_a)
        + functional.relu(margin - positive_score + score_b)
    ).mean()


def verifier_loss(
    positive_probabilities: Tensor,
    cf_a_probabilities: Tensor,
    cf_b_probabilities: Tensor,
) -> Tensor:
    """BCE loss for (p_P, p_E, p_V) on positive, CF-A, and CF-B examples."""

    expected = (
        torch.tensor((1.0, 1.0, 0.0), device=positive_probabilities.device),
        torch.tensor((1.0, 0.0, 1.0), device=positive_probabilities.device),
        torch.tensor((0.0, 1.0, 1.0), device=positive_probabilities.device),
    )
    losses = []
    probabilities = (positive_probabilities, cf_a_probabilities, cf_b_probabilities)
    for values, labels in zip(probabilities, expected, strict=True):
        labels = labels.to(dtype=values.dtype).expand_as(values)
        losses.append(functional.binary_cross_entropy(values, labels))
    return torch.stack(losses).mean()
