"""Utilities for inserting image-derived tokens into text token sequences."""

from __future__ import annotations

import torch
from torch import Tensor


def inject_pseudo_token(
    token_ids: Tensor,
    token_embeddings: Tensor,
    pseudo_tokens: Tensor,
    *,
    placeholder_token_id: int,
) -> Tensor:
    """Replace one placeholder embedding per sample with a pseudo-word token.

    Args:
        token_ids: Integer tensor shaped ``[batch, sequence]``.
        token_embeddings: Tensor shaped ``[batch, sequence, token_dim]``.
        pseudo_tokens: Tensor shaped ``[batch, token_dim]``.
        placeholder_token_id: Token ID used to mark the ``*`` position.

    Returns:
        A new embedding tensor. The input tensor is not modified.
    """

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [batch, sequence, token_dim]")
    if pseudo_tokens.ndim != 2:
        raise ValueError("pseudo_tokens must have shape [batch, token_dim]")
    if token_embeddings.shape[:2] != token_ids.shape:
        raise ValueError("token_ids and token_embeddings batch/sequence shapes must match")
    if token_embeddings.shape[0] != pseudo_tokens.shape[0]:
        raise ValueError("token_embeddings and pseudo_tokens batch sizes must match")
    if token_embeddings.shape[2] != pseudo_tokens.shape[1]:
        raise ValueError("token_embeddings and pseudo_tokens dimensions must match")

    placeholder_mask = token_ids.eq(placeholder_token_id)
    placeholder_counts = placeholder_mask.sum(dim=1)
    if not torch.all(placeholder_counts == 1):
        counts = placeholder_counts.detach().cpu().tolist()
        raise ValueError(f"Each sample must contain exactly one placeholder; counts={counts}")

    placeholder_positions = placeholder_mask.to(dtype=torch.int64).argmax(dim=1)
    batch_indices = torch.arange(token_ids.shape[0], device=token_ids.device)
    result = token_embeddings.clone()
    result[batch_indices, placeholder_positions] = pseudo_tokens.to(
        device=result.device,
        dtype=result.dtype,
    )
    return result
